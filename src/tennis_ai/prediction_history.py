"""Immutable pre-match predictions joined with final ATP results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from .player_matching import normalize_player_name
from .results_backfill import clean_tournament_name, player_identity_key


def _iso_utc(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _is_pre_match_record(record: dict[str, Any]) -> bool:
    """Only predictions actually captured before the scheduled start count."""
    start = pd.to_datetime(record.get("start_time_utc"), utc=True, errors="coerce")
    captured = pd.to_datetime(record.get("captured_at_utc"), utc=True, errors="coerce")
    state_as_of = pd.to_datetime(record.get("state_as_of_utc"), utc=True, errors="coerce")
    return bool(
        pd.notna(start)
        and pd.notna(captured)
        and captured < start
        and (pd.isna(state_as_of) or state_as_of < start)
    )


def _player_side(name: object, record: dict[str, Any]) -> str | None:
    """Match historical/provider name variants to the captured display side."""
    normalized = normalize_player_name(str(name or ""))
    if not normalized:
        return None
    names = {side: str(record[f"{side}_name"]) for side in ("p1", "p2")}
    checks = (
        lambda candidate: normalize_player_name(candidate) == normalized,
        lambda candidate: player_identity_key(candidate) == player_identity_key(name),
        lambda candidate: _compatible_identity_keys(player_identity_key(candidate), player_identity_key(name)),
        lambda candidate: set(normalize_player_name(candidate).split()) == set(normalized.split()),
    )
    for matches in checks:
        sides = [side for side, candidate in names.items() if matches(candidate)]
        if len(sides) == 1:
            return sides[0]
    return None


def _compatible_identity_keys(left: str, right: str) -> bool:
    """Accept a single first initial for a hyphenated first name, with surname fixed."""
    left_surname, _, left_initials = left.partition(":")
    right_surname, _, right_initials = right.partition(":")
    return bool(
        left_surname == right_surname
        and left_initials
        and right_initials
        and min(len(left_initials), len(right_initials)) == 1
        and left_initials[0] == right_initials[0]
    )


def _result_sides(record: dict[str, Any], winner: object, loser: object) -> tuple[str, str] | None:
    winner_side = _player_side(winner, record)
    loser_side = _player_side(loser, record)
    if winner_side is not None and loser_side is not None and winner_side == loser_side:
        return None
    if winner_side is None and loser_side is None:
        return None
    if winner_side is None:
        winner_side = "p2" if loser_side == "p1" else "p1"
    if loser_side is None:
        loser_side = "p2" if winner_side == "p1" else "p1"
    return winner_side, loser_side


def _predicted_side(record: dict[str, Any]) -> str:
    """The model's pick is the higher stored probability, never a name comparison."""
    return "p1" if record["p1_win_probability"] >= record["p2_win_probability"] else "p2"


def _display_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize legacy history at read time without changing saved probabilities."""
    sides = _result_sides(record, record.get("actual_winner"), record.get("actual_loser"))
    if sides is None:
        return None
    winner_side, loser_side = sides
    displayed = record.copy()
    if record.get("actual_played_at_utc"):
        displayed["scheduled_start_time_utc"] = record["start_time_utc"]
        displayed["start_time_utc"] = record["actual_played_at_utc"]
    displayed["predicted_side"] = _predicted_side(record)
    displayed["actual_side"] = winner_side
    displayed["predicted_winner"] = record[f"{displayed['predicted_side']}_name"]
    displayed["actual_winner"] = record[f"{winner_side}_name"]
    displayed["actual_loser"] = record[f"{loser_side}_name"]
    displayed["prediction_correct"] = (
        None if str(record.get("match_status", "")).casefold() == "defaulted"
        else displayed["predicted_side"] == winner_side
    )
    return displayed


class PredictionHistoryStore:
    """Durable store; a prediction is never overwritten after first capture."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._predictions = dict(payload.get("predictions", {}))
        else:
            self._predictions: dict[str, dict[str, Any]] = {}

    def capture(
        self, fixtures: pd.DataFrame, state: Any, predictor: Any,
        *, now_utc: datetime | None = None,
    ) -> int:
        captured = 0
        schedule_updated = False
        if fixtures.empty:
            return captured
        captured_at = pd.Timestamp(now_utc or datetime.now(timezone.utc))
        if captured_at.tzinfo is None:
            captured_at = captured_at.tz_localize("UTC")
        else:
            captured_at = captured_at.tz_convert("UTC")
        state_as_of = pd.to_datetime(state.state_as_of, utc=True, errors="coerce")
        for _, fixture in fixtures.iterrows():
            if not bool(fixture.get("identities_resolved", False)):
                continue
            start = pd.to_datetime(fixture.start_time_utc, utc=True, errors="coerce")
            if pd.isna(start) or captured_at >= start:
                continue
            if pd.notna(state_as_of) and state_as_of >= start:
                continue
            match_id = str(int(fixture.match_id))
            if match_id in self._predictions:
                record = self._predictions[match_id]
                if not record.get("actual_winner") and record.get("start_time_utc") != _iso_utc(start):
                    record["start_time_utc"] = _iso_utc(start)
                    schedule_updated = True
                continue
            prediction = predictor.predict_frame(state.build_feature_row(fixture))
            predicted_side = "p1" if prediction["p1_win_probability"] >= prediction["p2_win_probability"] else "p2"
            self._predictions[match_id] = {
                "match_id": int(fixture.match_id),
                "start_time_utc": _iso_utc(fixture.start_time_utc),
                "tournament_name": str(fixture.tournament_name),
                "surface": str(fixture.surface).title(),
                "round": str(fixture["round"]),
                "p1_name": str(fixture.p1_display_name),
                "p2_name": str(fixture.p2_display_name),
                "p1_win_probability": prediction["p1_win_probability"],
                "p2_win_probability": prediction["p2_win_probability"],
                "predicted_side": predicted_side,
                "predicted_winner": str(fixture[f"{predicted_side}_display_name"]),
                "confidence": prediction["confidence"],
                "captured_at_utc": _iso_utc(captured_at),
                "state_as_of_utc": _iso_utc(state.state_as_of),
                "actual_winner": None,
                "actual_loser": None,
                "match_status": None,
                "prediction_correct": None,
            }
            captured += 1
        if captured or schedule_updated:
            self.save()
        return captured

    def finalize(self, results: pd.DataFrame, *, reconcile_existing: bool = False) -> int:
        finalized = 0
        if results.empty:
            return finalized
        for row in results.itertuples(index=False):
            record = self._predictions.get(str(int(row.provider_match_id)))
            if record is None:
                result_time = pd.to_datetime(row.played_at_utc, utc=True, errors="coerce")
                for candidate in self._predictions.values():
                    if candidate.get("actual_winner") and not reconcile_existing:
                        continue
                    same_event = clean_tournament_name(str(candidate["tournament_name"])) == clean_tournament_name(str(row.tourney_name))
                    if not same_event:
                        continue
                    candidate_time = pd.to_datetime(candidate.get("start_time_utc"), utc=True, errors="coerce")
                    close_in_time = pd.notna(result_time) and pd.notna(candidate_time) and abs(result_time - candidate_time) <= pd.Timedelta(days=2)
                    if not close_in_time:
                        continue
                    winner_side = _player_side(row.winner_name, candidate)
                    loser_side = _player_side(row.loser_name, candidate)
                    if winner_side is not None and loser_side is not None and winner_side != loser_side:
                        record = candidate
                        break
            if record is None or (record.get("actual_winner") and not reconcile_existing):
                continue
            sides = _result_sides(record, row.winner_name, row.loser_name)
            if sides is None:
                continue
            winner_side, loser_side = sides
            predicted_side = _predicted_side(record)
            result_fields = {
                    "predicted_side": predicted_side,
                    "predicted_winner": record[f"{predicted_side}_name"],
                    "actual_side": winner_side,
                    "actual_winner": record[f"{winner_side}_name"],
                    "actual_loser": record[f"{loser_side}_name"],
                    "match_status": str(row.match_status),
                    "prediction_correct": None if str(row.match_status).casefold() == "defaulted" else predicted_side == winner_side,
                    "winner_sets": int(row.winner_sets),
                    "loser_sets": int(row.loser_sets),
                    "actual_played_at_utc": _iso_utc(getattr(row, "played_at_utc", None)),
            }
            if any(record.get(key) != value for key, value in result_fields.items()):
                record.update(result_fields)
                finalized += 1
        if finalized:
            self.save()
        return finalized

    def completed(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = [
            displayed for item in self._predictions.values()
            if item.get("actual_winner") and _is_pre_match_record(item)
            if (displayed := _display_record(item)) is not None
        ]
        rows.sort(key=lambda item: (item.get("start_time_utc") or "", item["match_id"]), reverse=True)
        return rows if limit is None else rows[: max(0, limit)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.path.parent, suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as handle:
            temporary = Path(handle.name)
            json.dump({"version": 1, "predictions": self._predictions}, handle, ensure_ascii=False, indent=2)
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
