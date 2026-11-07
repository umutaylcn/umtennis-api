"""Track FREE upcoming fixtures and apply their final results to current state."""

from __future__ import annotations

from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_ai.fixture_pipeline import build_upcoming_fixture_table, load_fixture_snapshot
from tennis_ai.inference import EnsemblePredictor
from tennis_ai.live_data import LiveTennisClient, TennisAPIError
from tennis_ai.prediction_history import PredictionHistoryStore
from tennis_ai.result_tracker import TrackedFixtureStore, collect_tracked_results
from tennis_ai.results_backfill import match_identity_key, match_identity_keys
from tennis_ai.state_engine import CurrentStateEngine, EloConfig


STATE_PATH = PROJECT_ROOT / "models" / "current_state.joblib"
BACKFILL_PATH = PROJECT_ROOT / "data" / "processed" / "atp_backfill_2026_current.pkl"
TRACKER_PATH = PROJECT_ROOT / "data" / "cache" / "tracked_fixtures.json"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "atp_model_data_1990_2026.pkl"
LOCK_PATH = PROJECT_ROOT / "data" / "cache" / "free_result_sync.lock"
PREDICTION_HISTORY_PATH = PROJECT_ROOT / "data" / "cache" / "prediction_history.json"


def selected_elo_config() -> EloConfig:
    payload = json.loads(
        (PROJECT_ROOT / "models" / "elo_config.json").read_text(encoding="utf-8")
    )
    names = {item.name for item in fields(EloConfig)}
    return EloConfig(**{key: payload[key] for key in names if key in payload})


def configure_request_budget(client: LiveTennisClient) -> int:
    """Keep a daily reserve and cap each update below the free minute limit."""
    try:
        usage = client.get_usage()
        remaining = int(usage["today"]["remaining_day"])
    except (TennisAPIError, KeyError, TypeError, ValueError) as exc:
        client.set_request_budget(client.requests_made)
        print(f"API usage unavailable; skipping quota-consuming calls: {exc}")
        return 0
    additional_budget = min(14, max(0, remaining - 10))
    client.set_request_budget(client.requests_made + additional_budget)
    print(f"API quota remaining: {remaining} | Additional request budget: {additional_budget}")
    return additional_budget


def atomic_joblib(value: object, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".joblib", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_pickle(frame: pd.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".pkl", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_pickle(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_state_and_backfill() -> tuple[CurrentStateEngine, pd.DataFrame]:
    config = selected_elo_config()
    if STATE_PATH.exists():
        state: CurrentStateEngine = joblib.load(STATE_PATH)
        state.elo_config = config
    else:
        state = CurrentStateEngine(config).bootstrap(pd.read_pickle(MODEL_DATA_PATH))
    if BACKFILL_PATH.exists():
        backfill = pd.read_pickle(BACKFILL_PATH)
    else:
        backfill = pd.DataFrame()
    return state, backfill


def semantic_unseen(results: pd.DataFrame, existing: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if results.empty:
        return results, []
    known = match_identity_keys(existing)
    duplicate_ids: list[int] = []
    keep: list[bool] = []
    for row in results.itertuples(index=False):
        duplicate = match_identity_key(row) in known
        keep.append(not duplicate)
        if duplicate:
            duplicate_ids.append(int(row.provider_match_id))
    return results.loc[keep].reset_index(drop=True), duplicate_ids


def mark_known_completed(store: TrackedFixtureStore, existing: pd.DataFrame) -> int:
    """Skip detail calls for fixtures already represented by the season CSV."""
    known = match_identity_keys(existing)
    duplicate_ids = []
    for record in store.pending():
        if not record.get("round"):
            continue
        identity = match_identity_key(SimpleNamespace(
            played_at_utc=record["start_time_utc"],
            tourney_name=record["tournament_name"],
            winner_name=record["p1_name"],
            loser_name=record["p2_name"],
            round=record["round"],
        ))
        if identity in known:
            duplicate_ids.append(int(record["match_id"]))
    if duplicate_ids:
        store.mark(duplicate_ids, "applied")
    return len(duplicate_ids)


def apply_results(
    state: CurrentStateEngine,
    existing: pd.DataFrame,
    new_results: pd.DataFrame,
) -> tuple[CurrentStateEngine, pd.DataFrame, str]:
    if new_results.empty:
        return state, existing, "no-op"
    combined = pd.concat([existing, new_results], ignore_index=True)
    combined = combined.sort_values(["played_at_utc", "provider_match_id"]).reset_index(drop=True)
    earliest = pd.to_datetime(new_results["played_at_utc"], utc=True).min()
    state_as_of = pd.to_datetime(state.state_as_of, utc=True) if state.state_as_of is not None else None
    if state_as_of is not None and earliest < state_as_of:
        state = CurrentStateEngine(selected_elo_config()).bootstrap(pd.read_pickle(MODEL_DATA_PATH))
        state.apply_backfill(combined)
        mode = "chronological rebuild"
    else:
        state.apply_backfill(new_results.sort_values(["played_at_utc", "provider_match_id"]))
        mode = "incremental"
    return state, combined, mode


def main() -> None:
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Another result sync is running or left a stale lock; refusing to publish stale data.")
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.close(lock_fd)
        client = LiveTennisClient.from_env(PROJECT_ROOT / ".env")
        configure_request_budget(client)

        # Freeze the last fixture snapshot before reading its result. This keeps
        # historical predictions honest: they are never recomputed post-match.
        state, existing = load_state_and_backfill()
        history = PredictionHistoryStore(PREDICTION_HISTORY_PATH)
        captured_before = history.capture(
            load_fixture_snapshot(PROJECT_ROOT), state, EnsemblePredictor(PROJECT_ROOT)
        )
        # Fetch fixtures first. A backlog of match-detail calls must not use up
        # the budget before tomorrow's matches can be captured.
        try:
            upcoming = build_upcoming_fixture_table(
                PROJECT_ROOT, client,
                historical_names=state.players.keys(),
                recent_names=(
                    name for name, player in state.players.items()
                    if player.last_season is not None and player.last_season >= pd.Timestamp.now(tz="UTC").year - 2
                ),
            )
            print(f"Upcoming fixtures tracked: {len(upcoming)}")
        except TennisAPIError as exc:
            upcoming = pd.DataFrame()
            print(f"Upcoming refresh skipped: {exc}")

        store = TrackedFixtureStore(TRACKER_PATH)
        already_completed = mark_known_completed(store, existing)
        detail_limit = min(12, client.remaining_request_budget or 0)
        results, terminal_ids, pending_ids, mismatch_ids = collect_tracked_results(
            store, client, limit=detail_limit
        )
        finalized = history.finalize(results)
        unseen, duplicate_ids = semantic_unseen(results, existing)
        state, combined, mode = apply_results(state, existing, unseen)

        if not unseen.empty:
            atomic_joblib(state, STATE_PATH)
            atomic_pickle(combined, BACKFILL_PATH)
        store.mark(
            [*duplicate_ids, *unseen.get("provider_match_id", pd.Series(dtype=int)).astype(int).tolist()],
            "applied",
        )
        store.mark(terminal_ids, "terminal")
        store.mark(mismatch_ids, "provider_mismatch")

        captured_after = history.capture(
            upcoming, state, EnsemblePredictor(PROJECT_ROOT)
        )

        print(f"Mode: {mode}")
        print(f"New completed matches applied: {len(unseen)}")
        print(f"Already known matches skipped: {len(duplicate_ids)}")
        print(f"Known completed fixtures skipped before API calls: {already_completed}")
        print(f"Terminal non-standard matches: {len(terminal_ids)}")
        print(f"Provider identity mismatches rejected: {len(mismatch_ids)}")
        print(f"Matches still pending: {len(store.pending())}")
        print(f"Predictions captured: {captured_before + captured_after}")
        print(f"Prediction results finalized: {finalized}")
        print(f"State as of UTC: {state.state_as_of}")
        print(f"Quota-consuming API requests made: {client.requests_made}")
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
