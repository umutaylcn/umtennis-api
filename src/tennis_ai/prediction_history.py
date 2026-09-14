"""Immutable pre-match predictions joined with final ATP results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

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


class PredictionHistoryStore:
    """Durable store; a prediction is never overwritten after first capture."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._predictions = dict(payload.get("predictions", {}))
        else:
            self._predictions: dict[str, dict[str, Any]] = {}

    def capture(self, fixtures: pd.DataFrame, state: Any, predictor: Any) -> int:
        captured = 0
        if fixtures.empty:
            return captured
        for _, fixture in fixtures.iterrows():
            if not bool(fixture.get("identities_resolved", False)):
                continue
            match_id = str(int(fixture.match_id))
            if match_id in self._predictions:
                continue
            prediction = predictor.predict_frame(state.build_feature_row(fixture))
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
                "predicted_winner": prediction["predicted_winner"],
                "confidence": prediction["confidence"],
                "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "state_as_of_utc": _iso_utc(state.state_as_of),
                "actual_winner": None,
                "actual_loser": None,
                "match_status": None,
                "prediction_correct": None,
            }
            captured += 1
        if captured:
            self.save()
        return captured

    def finalize(self, results: pd.DataFrame) -> int:
        finalized = 0
        if results.empty:
            return finalized
        for row in results.itertuples(index=False):
            record = self._predictions.get(str(int(row.provider_match_id)))
            if record is None:
                result_players = {
                    player_identity_key(row.winner_name),
                    player_identity_key(row.loser_name),
                }
                result_time = pd.to_datetime(row.played_at_utc, utc=True, errors="coerce")
                for candidate in self._predictions.values():
                    if candidate.get("actual_winner"):
                        continue
                    candidate_players = {
                        player_identity_key(candidate["p1_name"]),
                        player_identity_key(candidate["p2_name"]),
                    }
                    candidate_time = pd.to_datetime(candidate.get("start_time_utc"), utc=True, errors="coerce")
                    same_event = clean_tournament_name(str(candidate["tournament_name"])) == clean_tournament_name(str(row.tourney_name))
                    close_in_time = pd.notna(result_time) and pd.notna(candidate_time) and abs(result_time - candidate_time) <= pd.Timedelta(days=2)
                    if candidate_players == result_players and same_event and close_in_time:
                        record = candidate
                        break
            if record is None or record.get("actual_winner"):
                continue
            winner = str(row.winner_name)
            record.update(
                {
                    "actual_winner": winner,
                    "actual_loser": str(row.loser_name),
                    "match_status": str(row.match_status),
                    "prediction_correct": record.get("predicted_winner") == winner,
                    "winner_sets": int(row.winner_sets),
                    "loser_sets": int(row.loser_sets),
                }
            )
            finalized += 1
        if finalized:
            self.save()
        return finalized

    def completed(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = [item.copy() for item in self._predictions.values() if item.get("actual_winner")]
        rows.sort(key=lambda item: (item.get("start_time_utc") or "", item["match_id"]), reverse=True)
        return rows[: max(0, limit)]

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
