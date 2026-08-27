from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.state_engine import CurrentStateEngine, EloConfig


class RetirementPolicyTests(unittest.TestCase):
    def test_retirement_has_half_elo_weight_and_full_result_weight(self):
        state = CurrentStateEngine(EloConfig(provisional_k=32, standard_k=32))
        row = pd.Series(
            {
                "played_at_utc": pd.Timestamp("2026-08-26T12:00:00Z"),
                "tourney_name": "Winston-Salem",
                "surface": "Hard",
                "winner_name": "Juan Manuel Cerundolo",
                "loser_name": "Sebastian Baez",
                "winner_sets": 1,
                "loser_sets": 0,
                "winner_games": 10,
                "loser_games": 4,
                "winner_rank": 51,
                "loser_rank": 53,
                "winner_rank_points": 1006,
                "loser_rank_points": 1005,
                "match_status": "retirement",
            }
        )

        state.apply_completed_match(row)

        self.assertAlmostEqual(state.players["Juan Manuel Cerundolo"].elo, 1508.0)
        self.assertAlmostEqual(state.players["Sebastian Baez"].elo, 1492.0)
        self.assertEqual(state.players["Juan Manuel Cerundolo"].matches, 1)
        self.assertEqual(
            state.head_to_head_snapshot(
                "Juan Manuel Cerundolo", "Sebastian Baez", "Hard"
            )["p1_wins"],
            1,
        )
        tournament = state.tournaments[("Winston-Salem", "Juan Manuel Cerundolo")]
        self.assertEqual(tournament["wins"], 1)
        self.assertEqual(tournament["games_won"], 0)


if __name__ == "__main__":
    unittest.main()
