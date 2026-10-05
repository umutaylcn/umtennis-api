from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.results_backfill import load_completed_backfill, load_terminal_match_keys


class DefaultBackfillTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame([{
            "match_id": 199254,
            "date_timestamp": int(pd.Timestamp("2026-10-05T11:25:00Z").timestamp()),
            "tour_type": 1,
            "status": "FINISHED",
            "status_extra": "DEFAULTED",
            "winner_code": 1,
            "surface": "Hard",
            "tournament": "Beijing",
            "round": "Semi-finals",
            "home_name": "Novak Djokovic",
            "away_name": "Daniil Medvedev",
            "home_rank": 4,
            "away_rank": 10,
            "home_points": 5000,
            "away_points": 3500,
        }])
        self.names = pd.Series(["Novak Djokovic", "Daniil Medvedev"])

    def test_default_with_official_winner_is_a_result_without_score_stats(self):
        with patch("tennis_ai.results_backfill.pd.read_csv", return_value=self.source):
            result = load_completed_backfill("unused.csv", self.names)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].winner_name, "Novak Djokovic")
        self.assertEqual(result.iloc[0].match_status, "defaulted")
        self.assertEqual(result.iloc[0].winner_sets, 0)
        self.assertEqual(result.iloc[0].winner_games, 0)

    def test_default_without_official_winner_is_terminal_not_inferred(self):
        self.source.loc[0, "winner_code"] = 0
        with patch("tennis_ai.results_backfill.pd.read_csv", return_value=self.source):
            result = load_completed_backfill("unused.csv", self.names)
            terminal = load_terminal_match_keys("unused.csv", self.names)
        self.assertTrue(result.empty)
        self.assertEqual(len(terminal), 1)


if __name__ == "__main__":
    unittest.main()
