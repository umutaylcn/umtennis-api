from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.live_data import LiveTennisClient
from tennis_ai.fixture_pipeline import (
    apply_verified_fixture_rounds,
    fixture_round_code,
    verified_fixture_round,
)
import pandas as pd


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
    def test_round_name_fallback_does_not_invent_missing_round(self):
        self.assertEqual(fixture_round_code(None, "Round of 16"), "R16")
        self.assertEqual(fixture_round_code("QF", "Quarterfinal"), "QF")
        self.assertIsNone(fixture_round_code(None, None))

    def test_verified_rounds_match_exact_fixture_identity(self):
        self.assertEqual(
            verified_fixture_round(195725, "Chengdu", "Jenson Brooksby", "Nikoloz Basilashvili"),
            "QF",
        )
        self.assertEqual(
            verified_fixture_round(195478, "ATP Laver Cup", "Casper Ruud", "Francisco Cerundolo"),
            "DAY 1",
        )
        self.assertIsNone(
            verified_fixture_round(195725, "Chengdu", "Jenson Brooksby", "Another Player")
        )

    def test_verified_rounds_repair_only_missing_snapshot_values(self):
        fixtures = pd.DataFrame([
            {"match_id": 195741, "tournament_name": "Hangzhou", "p1_display_name": "Fabian Marozsan", "p2_display_name": "Kyrian Jacquet", "round": float("nan")},
            {"match_id": 195725, "tournament_name": "Chengdu", "p1_display_name": "Jenson Brooksby", "p2_display_name": "Nikoloz Basilashvili", "round": "SF"},
        ])
        updated = apply_verified_fixture_rounds(fixtures)
        self.assertEqual(updated["round"].tolist(), ["QF", "SF"])

    def test_upcoming_feed_keeps_scheduled_and_live_matches(self):
        client = LiveTennisClient("test-key")
        payload = {
            "data": [
                fixture(1, "scheduled"),
                fixture(4, "live"),
                fixture(2, "cancelled"),
                fixture(3, "finished"),
            ]
        }

        with patch.object(client, "_get", return_value=payload):
            matches = client.get_upcoming_matches()

        self.assertEqual([match.match_id for match in matches], [1, 4])
        self.assertEqual(matches[0].status, "scheduled")
        self.assertEqual(matches[1].status, "live")

    def test_upcoming_feed_uses_match_detail_id_when_available(self):
        client = LiveTennisClient("test-key")
        row = fixture(27091, "scheduled")
        row["match_id"] = 178288

        with patch.object(client, "_get", return_value={"data": [row]}):
            matches = client.get_upcoming_matches()

        self.assertEqual(matches[0].match_id, 178288)

    def test_upcoming_feed_excludes_davis_cup(self):
        client = LiveTennisClient("test-key")
        davis_cup = fixture(5, "scheduled")
        davis_cup["tournament"] = "ATP Davis Cup - World Group II"

        with patch.object(client, "_get", return_value={"data": [davis_cup]}):
            matches = client.get_upcoming_matches()

        self.assertEqual(matches, [])

    def test_upcoming_feed_excludes_qualifying_when_provider_flag_is_wrong(self):
        client = LiveTennisClient("test-key")
        qualifying = fixture(6, "scheduled")
        qualifying["round"] = "Qualification"
        qualifying["round_code"] = None

        with patch.object(client, "_get", return_value={"data": [qualifying]}):
            matches = client.get_upcoming_matches()

        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
