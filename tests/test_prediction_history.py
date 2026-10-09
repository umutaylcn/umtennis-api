from pathlib import Path
from datetime import datetime, timezone
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
    def test_capture_uses_display_side_not_historical_prediction_name(self):
        fixtures = pd.DataFrame([{"match_id": 21, "start_time_utc": pd.Timestamp("2026-09-11T12:00:00Z"), "tournament_name": "Laver Cup", "surface": "hard", "round": "DAY 1", "p1_display_name": "Jakub Menšik", "p2_display_name": "Brandon Nakashima", "identities_resolved": True}])

        class HistoricalNamePredictor:
            def predict_frame(self, frame):
                return {"p1_win_probability": 0.5264, "p2_win_probability": 0.4736, "predicted_winner": "Jakub Mensik", "confidence": 0.5264}

        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            store.capture(fixtures, FakeState(), HistoricalNamePredictor(), now_utc=datetime(2026, 9, 11, 8, tzinfo=timezone.utc))
            record = store._predictions["21"]
            self.assertEqual(record["predicted_side"], "p1")
            self.assertEqual(record["predicted_winner"], "Jakub Menšik")

    def test_legacy_aliases_display_correct_player_and_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            common = {"start_time_utc": "2026-09-11T12:00:00Z", "captured_at_utc": "2026-09-11T08:00:00Z", "state_as_of_utc": "2026-09-10T00:00:00Z", "tournament_name": "Test", "surface": "Hard", "round": "R32", "match_status": "completed"}
            store._predictions["22"] = {**common, "match_id": 22, "p1_name": "Coleman Chak Lam Wong", "p2_name": "Daniel Adolfo Vallejo", "p1_win_probability": 0.7352, "p2_win_probability": 0.2648, "confidence": 0.7352, "predicted_winner": "Coleman Wong", "actual_winner": "Coleman Wong", "actual_loser": "Adolfo Daniel Vallejo", "prediction_correct": True}
            store._predictions["23"] = {**common, "match_id": 23, "p1_name": "Yunchaokete Bu", "p2_name": "Michael Zheng", "p1_win_probability": 0.5939, "p2_win_probability": 0.4061, "confidence": 0.5939, "predicted_winner": "Bu Yunchaokete", "actual_winner": "Bu Yunchaokete", "actual_loser": "Michael Zheng", "prediction_correct": True}
            rows = {row["match_id"]: row for row in store.completed()}
            self.assertEqual(rows[22]["predicted_winner"], "Coleman Chak Lam Wong")
            self.assertEqual(rows[22]["actual_winner"], "Coleman Chak Lam Wong")
            self.assertEqual(rows[22]["predicted_side"], "p1")
            self.assertEqual(rows[22]["actual_side"], "p1")
            self.assertEqual(rows[22]["confidence"], 0.7352)
            self.assertEqual(rows[23]["predicted_winner"], "Yunchaokete Bu")
            self.assertTrue(rows[23]["prediction_correct"])

    def test_prediction_is_immutable_and_result_is_finalized(self):
        fixtures = pd.DataFrame([{"match_id": 12, "start_time_utc": pd.Timestamp("2026-09-11T12:00:00Z"), "tournament_name": "US Open", "surface": "hard", "round": "SF", "p1_display_name": "Player One", "p2_display_name": "Player Two", "identities_resolved": True}])
        results = pd.DataFrame([{"provider_match_id": 12, "winner_name": "Player Two", "loser_name": "Player One", "match_status": "completed", "winner_sets": 3, "loser_sets": 1}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            capture_time = datetime(2026, 9, 11, 8, tzinfo=timezone.utc)
            self.assertEqual(store.capture(fixtures, FakeState(), FakePredictor(), now_utc=capture_time), 1)
            self.assertEqual(store.capture(fixtures, FakeState(), FakePredictor(), now_utc=capture_time), 0)
            self.assertEqual(store.finalize(results), 1)
            row = PredictionHistoryStore(store.path).completed()[0]
            self.assertEqual(row["predicted_winner"], "Player One")
            self.assertEqual(row["actual_winner"], "Player Two")
            self.assertFalse(row["prediction_correct"])

    def test_authoritative_result_corrects_earlier_provider_winner(self):
        class RublevPredictor:
            def predict_frame(self, frame):
                return {"p1_win_probability": 0.21, "p2_win_probability": 0.79, "predicted_winner": "Andrey Rublev", "confidence": 0.79}

        fixtures = pd.DataFrame([{"match_id": 196491, "start_time_utc": pd.Timestamp("2026-09-28T09:30:00Z"), "tournament_name": "Hangzhou", "surface": "hard", "round": "SF", "p1_display_name": "Kyrian Jacquet", "p2_display_name": "Andrey Rublev", "identities_resolved": True}])
        tracked = pd.DataFrame([{"provider_match_id": 196491, "played_at_utc": pd.Timestamp("2026-09-28T09:30:00Z"), "tourney_name": "Hangzhou", "winner_name": "Kyrian Jacquet", "loser_name": "Andrey Rublev", "match_status": "completed", "winner_sets": 2, "loser_sets": 0}])
        official = pd.DataFrame([{"provider_match_id": 95981172, "played_at_utc": pd.Timestamp("2026-09-28T11:00:00Z"), "tourney_name": "Hangzhou", "winner_name": "Andrey Rublev", "loser_name": "Kyrian Jacquet", "match_status": "completed", "winner_sets": 2, "loser_sets": 0}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            capture_time = datetime(2026, 9, 28, 8, tzinfo=timezone.utc)
            self.assertEqual(store.capture(fixtures, FakeState(), RublevPredictor(), now_utc=capture_time), 1)
            self.assertEqual(store.finalize(tracked), 1)
            self.assertFalse(store.completed()[0]["prediction_correct"])
            self.assertEqual(store.finalize(official, reconcile_existing=True), 1)
            row = store.completed()[0]
            self.assertEqual(row["predicted_winner"], "Andrey Rublev")
            self.assertEqual(row["actual_winner"], "Andrey Rublev")
            self.assertTrue(row["prediction_correct"])
            self.assertEqual(row["captured_at_utc"], "2026-09-28T08:00:00Z")
            self.assertEqual(store.finalize(official, reconcile_existing=True), 0)

    def test_started_matches_are_not_captured_or_shown_as_previous_predictions(self):
        fixtures = pd.DataFrame([{"match_id": 13, "start_time_utc": pd.Timestamp("2026-09-11T12:00:00Z"), "tournament_name": "US Open", "surface": "hard", "round": "SF", "p1_display_name": "Player One", "p2_display_name": "Player Two", "identities_resolved": True}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            self.assertEqual(
                store.capture(
                    fixtures, FakeState(), FakePredictor(),
                    now_utc=datetime(2026, 9, 11, 13, tzinfo=timezone.utc),
                ), 0,
            )
            store._predictions["13"] = {
                "match_id": 13,
                "start_time_utc": "2026-09-11T12:00:00Z",
                "captured_at_utc": "2026-09-11T13:00:00Z",
                "state_as_of_utc": "2026-09-10T00:00:00Z",
                "actual_winner": "Player One",
            }
            self.assertEqual(store.completed(), [])

    def test_hyphenated_name_and_postponed_date_keep_original_prediction(self):
        fixtures = pd.DataFrame([{"match_id": 199924, "start_time_utc": pd.Timestamp("2026-10-07T02:00:00Z"), "tournament_name": "Shanghai", "surface": "hard", "round": "R128", "p1_display_name": "Martin Landaluce", "p2_display_name": "Jan-Lennard Struff", "identities_resolved": True}])
        result = pd.DataFrame([{"provider_match_id": 60470111, "played_at_utc": pd.Timestamp("2026-10-08T09:35:00Z"), "tourney_name": "Shanghai", "winner_name": "Struff J-L.", "loser_name": "Martin Landaluce", "match_status": "completed", "winner_sets": 2, "loser_sets": 0}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            store.capture(fixtures, FakeState(), FakePredictor(), now_utc=datetime(2026, 10, 6, 11, tzinfo=timezone.utc))
            original = store._predictions["199924"].copy()
            self.assertEqual(store.finalize(result), 1)
            row = store.completed()[0]
            self.assertEqual(row["actual_winner"], "Jan-Lennard Struff")
            self.assertEqual(row["start_time_utc"], "2026-10-08T09:35:00Z")
            self.assertEqual(row["scheduled_start_time_utc"], "2026-10-07T02:00:00Z")
            self.assertEqual(row["p1_win_probability"], original["p1_win_probability"])
            self.assertEqual(row["captured_at_utc"], original["captured_at_utc"])

    def test_rescheduled_future_match_updates_only_start_time(self):
        fixtures = pd.DataFrame([{"match_id": 42, "start_time_utc": pd.Timestamp("2026-10-10T02:00:00Z"), "tournament_name": "Shanghai", "surface": "hard", "round": "R32", "p1_display_name": "Player One", "p2_display_name": "Player Two", "identities_resolved": True}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            now = datetime(2026, 10, 8, 13, tzinfo=timezone.utc)
            self.assertEqual(store.capture(fixtures, FakeState(), FakePredictor(), now_utc=now), 1)
            original = store._predictions["42"].copy()
            fixtures.loc[0, "start_time_utc"] = pd.Timestamp("2026-10-09T09:00:00Z")
            self.assertEqual(store.capture(fixtures, FakeState(), FakePredictor(), now_utc=now), 0)
            updated = PredictionHistoryStore(store.path)._predictions["42"]
            self.assertEqual(updated["start_time_utc"], "2026-10-09T09:00:00Z")
            self.assertEqual(updated["p1_win_probability"], original["p1_win_probability"])
            self.assertEqual(updated["captured_at_utc"], original["captured_at_utc"])

    def test_completed_without_limit_returns_every_match(self):
        fixtures = pd.DataFrame([{"match_id": 1, "start_time_utc": pd.Timestamp("2026-09-11T12:00:00Z"), "tournament_name": "US Open", "surface": "hard", "round": "SF", "p1_display_name": "Player One", "p2_display_name": "Player Two", "identities_resolved": True}])
        results = pd.DataFrame([{"provider_match_id": 1, "played_at_utc": pd.Timestamp("2026-09-11T12:00:00Z"), "tourney_name": "US Open", "winner_name": "Player One", "loser_name": "Player Two", "match_status": "completed", "winner_sets": 2, "loser_sets": 0}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            store.capture(fixtures, FakeState(), FakePredictor(), now_utc=datetime(2026, 9, 11, 8, tzinfo=timezone.utc))
            store.finalize(results)
            original = store._predictions["1"]
            for match_id in range(2, 128):
                store._predictions[str(match_id)] = {**original, "match_id": match_id}
            self.assertEqual(len(store.completed()), 127)
            self.assertEqual(len(store.completed(100)), 100)


    def test_defaulted_prediction_is_displayed_but_not_scored(self):
        fixtures = pd.DataFrame([{"match_id": 199254, "start_time_utc": pd.Timestamp("2026-10-05T11:25:00Z"), "tournament_name": "Beijing", "surface": "hard", "round": "SF", "p1_display_name": "Novak Djokovic", "p2_display_name": "Daniil Medvedev", "identities_resolved": True}])
        result = pd.DataFrame([{"provider_match_id": 199254, "played_at_utc": pd.Timestamp("2026-10-05T11:25:00Z"), "tourney_name": "Beijing", "winner_name": "Novak Djokovic", "loser_name": "Daniil Medvedev", "match_status": "defaulted", "winner_sets": 0, "loser_sets": 0}])
        with tempfile.TemporaryDirectory() as directory:
            store = PredictionHistoryStore(Path(directory) / "history.json")
            store.capture(fixtures, FakeState(), FakePredictor(), now_utc=datetime(2026, 10, 4, 8, tzinfo=timezone.utc))
            store.finalize(result)
            self.assertIsNone(store._predictions["199254"]["prediction_correct"])
            row = store.completed()[0]
            self.assertEqual(row["match_status"], "defaulted")
            self.assertIsNone(row["prediction_correct"])
            store._predictions["199254"]["prediction_correct"] = True
            self.assertIsNone(store.completed()[0]["prediction_correct"])


if __name__ == "__main__":
    unittest.main()
