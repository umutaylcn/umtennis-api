from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.result_tracker import TrackedFixtureStore, collect_tracked_results
from tennis_ai.results_backfill import match_identity_keys


def fixture_frame(match_id: int = 42) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "start_time_utc": pd.Timestamp("2026-08-23T10:00:00Z"),
                "tournament_name": "Winston-Salem",
                "surface": "hard",
                "round": "R32",
                "p1_historical_name": "Adam Walton",
                "p2_historical_name": "Jesper De Jong",
                "p1_id": 159,
                "p2_id": 229,
                "p1_current_rank": 98,
                "p2_current_rank": 106,
                "p1_current_rank_points": 670,
                "p2_current_rank_points": 618,
                "identities_resolved": True,
            }
        ]
    )


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get_match(self, match_id):
        return {"id": match_id, **self.payload}


class ResultTrackerTests(unittest.TestCase):
    def test_completed_match_survives_restart_and_builds_winner_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracked.json"
            store = TrackedFixtureStore(path)
            self.assertEqual(store.track(fixture_frame()), 1)
            reloaded = TrackedFixtureStore(path)
            results, terminal, pending, mismatch = collect_tracked_results(
                reloaded,
                FakeClient(
                    {
                        "status": "completed",
                        "event_status": None,
                        "winner": 2,
                        "scheduled_time": "2026-08-23T10:00:00Z",
                        "tournament": "Winston-Salem",
                        "surface": "hard",
                        "round_code": "R32",
                        "score": {
                            "sets": [1, 2],
                            "games": [[6, 3, 2], [4, 6, 6]],
                        },
                    }
                ),
                now=pd.Timestamp("2026-08-24T00:00:00Z"),
            )
            self.assertEqual(terminal, [])
            self.assertEqual(pending, [])
            self.assertEqual(mismatch, [])
            self.assertEqual(results.iloc[0].winner_name, "Jesper De Jong")
            self.assertEqual(results.iloc[0].winner_games, 16)
            self.assertEqual(results.iloc[0].loser_games, 11)
            self.assertEqual(len(match_identity_keys(results)), 1)

    def test_retirement_becomes_weighted_state_result(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrackedFixtureStore(Path(directory) / "tracked.json")
            store.track(fixture_frame(99))
            results, terminal, pending, mismatch = collect_tracked_results(
                store,
                FakeClient(
                    {
                        "status": "completed",
                        "event_status": "Retired",
                        "winner": 1,
                    }
                ),
                now=pd.Timestamp("2026-08-24T00:00:00Z"),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results.iloc[0].winner_name, "Adam Walton")
            self.assertEqual(results.iloc[0].match_status, "retirement")
            self.assertEqual(terminal, [])
            self.assertEqual(pending, [])
            self.assertEqual(mismatch, [])

    def test_live_match_stays_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrackedFixtureStore(Path(directory) / "tracked.json")
            store.track(fixture_frame(77))
            results, terminal, pending, mismatch = collect_tracked_results(
                store,
                FakeClient({"status": "live", "winner": None}),
                now=pd.Timestamp("2026-08-24T00:00:00Z"),
            )
            self.assertTrue(results.empty)
            self.assertEqual(terminal, [])
            self.assertEqual(pending, [77])
            self.assertEqual(mismatch, [])

    def test_reused_provider_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TrackedFixtureStore(Path(directory) / "tracked.json")
            store.track(fixture_frame(27084))
            results, terminal, pending, mismatch = collect_tracked_results(
                store,
                FakeClient(
                    {
                        "status": "completed",
                        "winner": 2,
                        "scheduled_time": "2023-01-24T11:55:00Z",
                        "players": {
                            "p1": {"id": 5171, "name": "Sean Hodkin"},
                            "p2": {"id": 14463, "name": "Sahar Simon"},
                        },
                    }
                ),
                now=pd.Timestamp("2026-08-24T00:00:00Z"),
            )
            self.assertTrue(results.empty)
            self.assertEqual(terminal, [])
            self.assertEqual(pending, [])
            self.assertEqual(mismatch, [27084])


if __name__ == "__main__":
    unittest.main()
