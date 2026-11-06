import unittest

from scripts.check_completed_fixtures import is_terminal, same_fixture, signature


MATCH = {
    "match_id": 42,
    "p1_id": 1,
    "p2_id": 2,
    "p1_name": "Player One",
    "p2_name": "Player Two",
    "tournament_name": "Paris",
    "start_time_utc": "2026-11-06T13:00:00Z",
}


class CompletedFixtureTests(unittest.TestCase):
    def test_reversed_player_order_is_valid(self):
        detail = {"players": {"p1": {"id": 2}, "p2": {"id": 1}},
                  "scheduled_time": "2026-11-06T13:00:00Z"}
        self.assertTrue(same_fixture(MATCH, detail))

    def test_wrong_provider_match_is_rejected(self):
        detail = {"players": {"p1": {"id": 3}, "p2": {"id": 1}}}
        self.assertFalse(same_fixture(MATCH, detail))

    def test_completed_and_cancelled_are_terminal(self):
        self.assertTrue(is_terminal({"status": "completed"}))
        self.assertTrue(is_terminal({"status": "cancelled"}))
        self.assertFalse(is_terminal({"status": "live"}))

    def test_signature_preserves_fixture_identity(self):
        self.assertEqual(signature(MATCH)["start_time_utc"], MATCH["start_time_utc"])


if __name__ == "__main__":
    unittest.main()
