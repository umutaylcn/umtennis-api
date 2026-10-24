from pathlib import Path
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.api import PredictionService


class FakeHistory:
    def __init__(self, record):
        self.record = record

    def completed(self, limit):
        return [self.record.copy()]


def test_previous_round_is_verified_only_for_display():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        backfill_path = root / "data" / "processed" / "atp_backfill_2026_current.pkl"
        backfill_path.parent.mkdir(parents=True)
        pd.DataFrame([{
            "tourney_name": "Chengdu",
            "winner_name": "Player One",
            "loser_name": "Player Two",
            "played_at_utc": pd.Timestamp("2026-09-25T12:00:00Z"),
            "round": "R16",
        }]).to_pickle(backfill_path)
        original = {
            "match_id": 123,
            "tournament_name": "Chengdu",
            "p1_name": "Player One",
            "p2_name": "Player Two",
            "start_time_utc": "2026-09-25T12:00:00Z",
            "round": "nan",
            "p1_win_probability": 0.7,
            "p2_win_probability": 0.3,
            "prediction_correct": True,
        }
        service = PredictionService.__new__(PredictionService)
        service.project_root = root
        service.prediction_history = FakeHistory(original)

        displayed = service.previous_match_list()[0]

        assert displayed["round"] == "R16"
        assert displayed["round_at_prediction"] == "nan"
        assert displayed["round_verified_after_match"] is True
        assert displayed["p1_win_probability"] == 0.7
        assert displayed["prediction_correct"] is True
        assert original["round"] == "nan"
