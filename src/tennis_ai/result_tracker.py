"""Persist upcoming fixtures and turn their FREE match-detail responses into results."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .live_data import LiveTennisClient, TennisAPIError
from .player_matching import normalize_player_name
from .results_backfill import (
    clean_tournament_name,
    tournament_draw_size,
    tournament_level,
)


TERMINAL_EVENT_STATUSES = {"retired", "walk over", "cancelled"}


def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class TrackedFixtureStore:
    """Small durable queue whose keys are stable Live Tennis match IDs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._fixtures = dict(payload.get("fixtures", {}))
        else:
            self._fixtures: dict[str, dict[str, Any]] = {}

    def track(self, fixtures: pd.DataFrame) -> int:
        if fixtures.empty:
            return 0
        added = 0
        for row in fixtures.itertuples(index=False):
            if not bool(row.identities_resolved):
                continue
            match_id = str(int(row.match_id))
            record = {
                "match_id": int(row.match_id),
                "start_time_utc": _json_value(row.start_time_utc),
                "tournament_name": str(row.tournament_name),
                "surface": str(row.surface).title(),
                "round": _json_value(row.round),
                "p1_name": str(row.p1_historical_name),
                "p2_name": str(row.p2_historical_name),
                "p1_id": _json_value(row.p1_id),
                "p2_id": _json_value(row.p2_id),
                "p1_rank": _json_value(row.p1_current_rank),
                "p2_rank": _json_value(row.p2_current_rank),
                "p1_rank_points": _json_value(row.p1_current_rank_points),
                "p2_rank_points": _json_value(row.p2_current_rank_points),
                "sync_status": "pending",
            }
            existing = self._fixtures.get(match_id)
            if existing is None:
                self._fixtures[match_id] = record
                added += 1
            elif existing.get("sync_status") == "pending":
                self._fixtures[match_id] = {**existing, **record}
        self.save()
        return added

    def pending(self, now: pd.Timestamp | None = None) -> list[dict[str, Any]]:
        cutoff = now or pd.Timestamp.now(tz="UTC")
        records: list[dict[str, Any]] = []
        for record in self._fixtures.values():
            if record.get("sync_status") != "pending":
                continue
            start = pd.to_datetime(record.get("start_time_utc"), utc=True, errors="coerce")
            if pd.notna(start) and start <= cutoff:
                records.append(record.copy())
        return sorted(
            records,
            key=lambda item: (item.get("start_time_utc") or "", int(item["match_id"])),
        )

    def mark(self, match_ids: list[int], status: str) -> None:
        for match_id in match_ids:
            record = self._fixtures.get(str(int(match_id)))
            if record is not None:
                record["sync_status"] = status
        self.save()

    def records(self, match_ids: list[int]) -> list[dict[str, Any]]:
        return [
            self._fixtures[str(int(match_id))].copy()
            for match_id in match_ids
            if str(int(match_id)) in self._fixtures
        ]

    def save(self) -> None:
        _atomic_json({"version": 1, "fixtures": self._fixtures}, self.path)


def _score_totals(detail: dict[str, Any], winner: int) -> tuple[int, int, int, int]:
    score = detail.get("score") or {}
    sets = score.get("sets") or [0, 0]
    games = score.get("games") or [[], []]
    if len(sets) < 2 or len(games) < 2:
        raise ValueError("Completed match has an incomplete score payload")
    winner_index = winner - 1
    loser_index = 1 - winner_index
    return (
        int(sets[winner_index]),
        int(sets[loser_index]),
        sum(int(value) for value in (games[winner_index] or []) if value is not None),
        sum(int(value) for value in (games[loser_index] or []) if value is not None),
    )


def _completed_row(record: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    winner = int(detail["winner"])
    winner_prefix = "p1" if winner == 1 else "p2"
    loser_prefix = "p2" if winner == 1 else "p1"
    tournament = clean_tournament_name(
        str(detail.get("tournament") or record["tournament_name"])
    )
    surface = str(detail.get("surface") or record["surface"]).title()
    round_code = detail.get("round_code") or record.get("round")
    winner_sets, loser_sets, winner_games, loser_games = _score_totals(detail, winner)
    played_at = detail.get("scheduled_time") or record["start_time_utc"]
    level = tournament_level(tournament)
    return {
        "provider_match_id": int(record["match_id"]),
        "played_at_utc": pd.to_datetime(played_at, utc=True),
        "tourney_id": f"{pd.Timestamp(played_at).year}-{tournament}",
        "tourney_name": tournament,
        "tourney_level": level,
        "draw_size": tournament_draw_size(tournament),
        "surface": surface,
        "round": round_code,
        "best_of": 5 if level == "G" else 3,
        "winner_name": record[f"{winner_prefix}_name"],
        "loser_name": record[f"{loser_prefix}_name"],
        "winner_sets": winner_sets,
        "loser_sets": loser_sets,
        "winner_games": winner_games,
        "loser_games": loser_games,
        "winner_rank": record.get(f"{winner_prefix}_rank"),
        "loser_rank": record.get(f"{loser_prefix}_rank"),
        "winner_rank_points": record.get(f"{winner_prefix}_rank_points"),
        "loser_rank_points": record.get(f"{loser_prefix}_rank_points"),
        "home_match_status": "provider_id",
        "away_match_status": "provider_id",
        "home_match_score": 1.0,
        "away_match_score": 1.0,
    }


def _detail_matches_record(record: dict[str, Any], detail: dict[str, Any]) -> bool:
    """Reject provider ID collisions before they can mutate the Elo state."""
    players = detail.get("players") or {}
    detail_profiles = [players.get("p1") or {}, players.get("p2") or {}]
    tracked_ids = {
        int(value)
        for value in (record.get("p1_id"), record.get("p2_id"))
        if value is not None
    }
    detail_ids = {
        int(profile["id"])
        for profile in detail_profiles
        if profile.get("id") is not None
    }
    if tracked_ids and detail_ids:
        if tracked_ids != detail_ids:
            return False
    else:
        tracked_names = {
            normalize_player_name(record["p1_name"]),
            normalize_player_name(record["p2_name"]),
        }
        detail_names = {
            normalize_player_name(profile.get("name") or "")
            for profile in detail_profiles
            if profile.get("name")
        }
        if detail_names and tracked_names != detail_names:
            return False

    tracked_time = pd.to_datetime(record.get("start_time_utc"), utc=True, errors="coerce")
    detail_time = pd.to_datetime(detail.get("scheduled_time"), utc=True, errors="coerce")
    if pd.notna(tracked_time) and pd.notna(detail_time):
        if abs(detail_time - tracked_time) > pd.Timedelta(days=2):
            return False
    return True


def collect_tracked_results(
    store: TrackedFixtureStore,
    client: LiveTennisClient,
    *,
    now: pd.Timestamp | None = None,
    limit: int = 80,
) -> tuple[pd.DataFrame, list[int], list[int], list[int]]:
    """Return normal, terminal, pending and provider-mismatch match IDs."""
    completed_rows: list[dict[str, Any]] = []
    terminal_ids: list[int] = []
    still_pending: list[int] = []
    mismatch_ids: list[int] = []
    for record in store.pending(now)[:limit]:
        match_id = int(record["match_id"])
        try:
            detail = client.get_match(match_id)
        except TennisAPIError:
            still_pending.append(match_id)
            continue
        if not _detail_matches_record(record, detail):
            mismatch_ids.append(match_id)
            continue
        status = str(detail.get("status") or "").casefold()
        event_status = str(detail.get("event_status") or "").casefold()
        if status == "cancelled" or event_status in TERMINAL_EVENT_STATUSES:
            terminal_ids.append(match_id)
        elif status == "completed" and detail.get("winner") in (1, 2):
            completed_rows.append(_completed_row(record, detail))
        else:
            still_pending.append(match_id)

    result = pd.DataFrame(completed_rows)
    if not result.empty:
        result = result.sort_values(["played_at_utc", "provider_match_id"]).reset_index(drop=True)
    return result, terminal_ids, still_pending, mismatch_ids
