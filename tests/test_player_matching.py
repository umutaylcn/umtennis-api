from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.player_matching import HistoricalPlayerMatcher


class HistoricalPlayerMatcherTests(unittest.TestCase):
    def test_short_name_maps_to_cached_full_identity(self):
        matcher = HistoricalPlayerMatcher(
            ["Cruz Hewitt", "Sebastian Gorzny", "Sebastian Baez"]
        )

        self.assertEqual(matcher.match_surname_initial("Hewitt C.")[0], "Cruz Hewitt")
        self.assertEqual(
            matcher.match_surname_initial("Gorzny S.")[0], "Sebastian Gorzny"
        )
        self.assertEqual(
            matcher.match_surname_initial("S. Baez")[0], "Sebastian Baez"
        )

    def test_abbreviated_history_cannot_capture_unrelated_short_name(self):
        matcher = HistoricalPlayerMatcher(["Sesko Z."])

        matched, _, status = matcher.match_surname_initial("Gorzny S.")

        self.assertIsNone(matched)
        self.assertEqual(status, "unresolved_short")


if __name__ == "__main__":
    unittest.main()
