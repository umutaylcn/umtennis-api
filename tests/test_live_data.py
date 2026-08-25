from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.live_data import LiveTennisClient


def fixture(match_id: int, status: str) -> dict[str, object]:
    return {
        "id": match_id,
        "event_date": "2026-08-24",
        "start_time": "2026-08-24T18:00:00Z",
        "player1_id": 1,
        "player1_name": "Player One",
        "player2_id": 2,
        "player2_name": "Player Two",
        "tournament": "Winston-Salem",
        "surface": "Hard",
        "round": "Round of 64",
        "round_code": "R64",
        "status": status,
        "is_qualifying": False,
    }


class LiveTennisClientTests(unittest.TestCase):
    def test_upcoming_feed_keeps_only_scheduled_matches(self):
        client = LiveTennisClient("test-key")
        payload = {
            "data": [
                fixture(1, "scheduled"),
                fixture(2, "cancelled"),
                fixture(3, "finished"),
            ]
        }

        with patch.object(client, "_get", return_value=payload):
            matches = client.get_upcoming_matches()

        self.assertEqual([match.match_id for match in matches], [1])
        self.assertEqual(matches[0].status, "scheduled")

    def test_upcoming_feed_uses_match_detail_id_when_available(self):
        client = LiveTennisClient("test-key")
        row = fixture(27091, "scheduled")
        row["match_id"] = 178288

        with patch.object(client, "_get", return_value={"data": [row]}):
            matches = client.get_upcoming_matches()

        self.assertEqual(matches[0].match_id, 178288)


if __name__ == "__main__":
    unittest.main()
