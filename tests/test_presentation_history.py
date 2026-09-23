from collections import defaultdict
from pathlib import Path
import sys
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.presentation import PlayerPresentationService
from tennis_ai.results_backfill import match_identity_key


class PresentationHistoryTests(unittest.TestCase):
    def test_backfill_only_result_is_added_once_to_both_players(self):
        service = PlayerPresentationService.__new__(PlayerPresentationService)
        service._matches = defaultdict(list)
        backfill = pd.DataFrame(
            [
                {
                    "played_at_utc": "2026-09-08T01:15:00Z",
                    "tourney_name": "US Open",
                    "round": "R16",
                    "surface": "Hard",
                    "winner_name": "Alexander Zverev",
                    "loser_name": "Luciano Darderi",
                },
                {
                    "played_at_utc": "2026-09-10T00:00:00Z",
                    "tourney_name": "US Open",
                    "round": "QF",
                    "surface": "Hard",
                    "winner_name": "Alexander Zverev",
                    "loser_name": "Botic Van De Zandschulp",
                },
            ]
        )
        existing = {match_identity_key(next(backfill.iloc[:1].itertuples(index=False)))}

        service._append_backfill_only_history(
            backfill,
            {"Alexander Zverev", "Luciano Darderi", "Botic Van De Zandschulp"},
            existing,
        )

        self.assertEqual(len(service._matches["Alexander Zverev"]), 1)
        self.assertTrue(service._matches["Alexander Zverev"][0]["won"])
        self.assertEqual(len(service._matches["Botic Van De Zandschulp"]), 1)
        self.assertFalse(service._matches["Botic Van De Zandschulp"][0]["won"])
        self.assertEqual(len(service._matches["Luciano Darderi"]), 0)

    def test_provider_duplicate_and_unsupported_surface_are_not_added(self):
        service = PlayerPresentationService.__new__(PlayerPresentationService)
        service._matches = defaultdict(list)
        backfill = pd.DataFrame(
            [
                {
                    "provider_match_id": 101,
                    "played_at_utc": "2026-09-03T23:40:00Z",
                    "tourney_name": "US Open",
                    "round": "R64",
                    "surface": "Hard",
                    "winner_name": "Michael Zheng",
                    "loser_name": "Bu Y.",
                },
                {
                    "provider_match_id": 102,
                    "played_at_utc": "2026-09-19T12:00:00Z",
                    "tourney_name": "Davis Cup",
                    "round": "RR",
                    "surface": "No Surface",
                    "winner_name": "Michael Zheng",
                    "loser_name": "Player Two",
                },
            ]
        )

        service._append_backfill_only_history(
            backfill,
            {"Michael Zheng", "Bu Y.", "Player Two"},
            set(),
            {101},
        )

        self.assertEqual(len(service._matches["Michael Zheng"]), 0)


if __name__ == "__main__":
    unittest.main()
