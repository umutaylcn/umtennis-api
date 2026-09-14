from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.prediction_history import PredictionHistoryStore


class FakeState:
    state_as_of = pd.Timestamp("2026-09-10T00:00:00Z")

    def build_feature_row(self, fixture):
        return pd.DataFrame([{"p1_name": fixture.p1_display_name, "p2_name": fixture.p2_display_name}])


class FakePredictor:
    def predict_frame(self, frame):
        return {"p1_win_probability": 0.7, "p2_win_probability": 0.3, "predicted_winner": frame.iloc[0].p1_name, "confidence": 0.7}


class PredictionHistoryTests(unittest.TestCase):
    def test_prediction_is_immutable_and_result_is_finalized(self):
        fixtures = pd.DataFrame([{"match_id": 12, "start_time_utc": pd.Timestamp("2026-09-11T12:00:00Z"), "tournament_name": "US Open", "surface": "hard", "round": "SF", "p1_display_name": "Player One", "p2_display_name": "Player Two", "identities_resolved": True}])
        results = pd.DataFrame([{"provider_match_id": 12, "winner_name": "Player Two", "loser_name": "Player One", "match_status": "completed", "winner_sets": 3, "loser_sets": 1}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            self.assertEqual(store.capture(fixtures, FakeState(), FakePredictor()), 1)
            self.assertEqual(store.capture(fixtures, FakeState(), FakePredictor()), 0)
            self.assertEqual(store.finalize(results), 1)
            row = PredictionHistoryStore(store.path).completed()[0]
            self.assertEqual(row["predicted_winner"], "Player One")
            self.assertEqual(row["actual_winner"], "Player Two")
            self.assertFalse(row["prediction_correct"])


if __name__ == "__main__":
    unittest.main()
