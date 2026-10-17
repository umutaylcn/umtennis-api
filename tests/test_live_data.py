from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.live_data import LiveTennisClient, TennisAPIError
from tennis_ai.fixture_pipeline import (
    apply_verified_fixture_rounds,
    exclude_pre_main_draw_fixtures,
    fill_missing_fixture_ranks,
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
    def test_scheduled_csv_fills_only_exact_missing_fixture_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "data" / "external" / "2026-atp-season.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame([{
                "date_timestamp": 1792116000,
                "tournament": "Shanghai ATP",
                "round": "Quarter-finals",
                "status": "SCHEDULED",
                "home_name": "Zverev A.",
                "away_name": "De Minaur A.",
                "home_rank": 2,
                "away_rank": 9,
                "home_points": 9720,
                "away_points": 3300,
            }]).to_csv(csv_path, index=False)
            fixtures = pd.DataFrame([
                {"event_date": "2026-10-16", "tournament_name": "Shanghai", "round": None,
                 "p1_historical_name": "Alexander Zverev", "p2_historical_name": "Alex De Minaur"},
                {"event_date": "2026-10-16", "tournament_name": "Shanghai", "round": "R16",
                 "p1_historical_name": "Alexander Zverev", "p2_historical_name": "Alex De Minaur"},
                {"event_date": "2026-10-17", "tournament_name": "Shanghai", "round": None,
                 "p1_historical_name": "Alexander Zverev", "p2_historical_name": "Alex De Minaur"},
            ])

            enriched = fill_missing_fixture_ranks(fixtures, temporary)

        self.assertEqual(enriched.loc[0, "round"], "QF")
        self.assertEqual(enriched.loc[0, "p1_current_rank"], 2)
        self.assertEqual(enriched.loc[1, "round"], "R16")
        self.assertTrue(pd.isna(enriched.loc[2, "round"]))

    def test_update_budget_counts_usage_and_blocks_extra_calls(self):
        response = Mock(status_code=200)
        response.json.return_value = {"data": {}}
        client = LiveTennisClient("test-key", session=Mock(get=Mock(return_value=response)))
        client.set_request_budget(2)

        client.get_usage()
        client.get_match(1)
        with self.assertRaises(TennisAPIError):
            client.get_match(2)

        self.assertEqual(client.requests_made, 2)
        self.assertEqual(client._session.get.call_count, 2)

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
        self.assertEqual(
            verified_fixture_round(203161, "Shanghai", "Botic Van De Zandschulp", "Rafael Jodar"),
            "R16",
        )
        self.assertEqual(
            verified_fixture_round(203160, "Shanghai", "Tommy Paul", "Zizou Bergs"),
            "R16",
        )
        self.assertIsNone(
            verified_fixture_round(195725, "Chengdu", "Jenson Brooksby", "Another Player")
        )

    def test_verified_rounds_repair_only_missing_snapshot_values(self):
        fixtures = pd.DataFrame([
            {"match_id": 195741, "tournament_name": "Hangzhou", "p1_display_name": "Fabian Marozsan", "p2_display_name": "Kyrian Jacquet", "round": float("nan")},
            {"match_id": 195725, "tournament_name": "Chengdu", "p1_display_name": "Jenson Brooksby", "p2_display_name": "Nikoloz Basilashvili", "round": "SF"},
            {"match_id": 203161, "tournament_name": "Shanghai", "p1_display_name": "Botic Van De Zandschulp", "p2_display_name": "Rafael Jodar", "round": float("nan")},
            {"match_id": 203160, "tournament_name": "Shanghai", "p1_display_name": "Tommy Paul", "p2_display_name": "Zizou Bergs", "round": float("nan")},
        ])
        updated = apply_verified_fixture_rounds(fixtures)
        self.assertEqual(updated["round"].tolist(), ["QF", "SF", "R16", "R16"])

    def test_cached_fixture_filter_removes_only_verified_qualifying_dates(self):
        fixtures = pd.DataFrame([
            {"match_id": 1, "tournament_name": "Tokyo", "event_date": "2026-09-28"},
            {"match_id": 2, "tournament_name": "Beijing", "event_date": "2026-09-29"},
            {"match_id": 3, "tournament_name": "Tokyo", "event_date": "2026-09-30"},
            {"match_id": 4, "tournament_name": "Chengdu", "event_date": "2026-09-28"},
            {"match_id": 5, "tournament_name": "Lyon (France) - Qualification", "event_date": "2026-10-18"},
        ])
        self.assertEqual(exclude_pre_main_draw_fixtures(fixtures)["match_id"].tolist(), [3, 4])

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

    def test_upcoming_feed_excludes_qualification_named_tournament(self):
        client = LiveTennisClient("test-key")
        qualifying = fixture(5, "scheduled")
        qualifying["tournament"] = "Lyon (France) - Qualification"
        qualifying["round"] = None
        qualifying["round_code"] = None

        with patch.object(client, "_get", return_value={"data": [qualifying]}):
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

    def test_upcoming_feed_excludes_unlabeled_2026_tokyo_and_beijing_qualifiers(self):
        client = LiveTennisClient("test-key")
        tokyo_q = fixture(7, "scheduled")
        tokyo_q.update(tournament="Tokyo", event_date="2026-09-28", round=None, round_code=None)
        beijing_q = fixture(8, "scheduled")
        beijing_q.update(tournament="Beijing", event_date="2026-09-29", round=None, round_code=None)
        tokyo_main = fixture(9, "scheduled")
        tokyo_main.update(tournament="Tokyo", event_date="2026-09-30", round="Round of 32", round_code="R32")
        with patch.object(client, "_get", return_value={"data": [tokyo_q, beijing_q, tokyo_main]}):
            matches = client.get_upcoming_matches()
        self.assertEqual([match.match_id for match in matches], [9])


if __name__ == "__main__":
    unittest.main()
