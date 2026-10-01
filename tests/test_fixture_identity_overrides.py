from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.fixture_pipeline import apply_player_id_name_overrides, fill_missing_fixture_ranks


def test_zhizhen_zhang_id_overrides_wrong_provider_identity_on_either_side():
    fixtures = pd.DataFrame(
        [
            {
                "p1_id": 445,
                "p1_display_name": "Ying Zhang",
                "p1_historical_name": "Ying Zhang",
                "p2_id": 10,
                "p2_display_name": "Other Player",
                "p2_historical_name": "Other Player",
            },
            {
                "p1_id": 10,
                "p1_display_name": "Other Player",
                "p1_historical_name": "Other Player",
                "p2_id": 445,
                "p2_display_name": "Ying Zhang",
                "p2_historical_name": "Ying Zhang",
            },
        ]
    )

    corrected = apply_player_id_name_overrides(fixtures)

    assert corrected.loc[0, "p1_display_name"] == "Zhizhen Zhang"
    assert corrected.loc[0, "p1_historical_name"] == "Zhizhen Zhang"
    assert corrected.loc[1, "p2_display_name"] == "Zhizhen Zhang"
    assert corrected.loc[1, "p2_historical_name"] == "Zhizhen Zhang"
    assert corrected.loc[0, "p2_display_name"] == "Other Player"


def test_abbreviated_names_use_full_display_names():
    fixtures = pd.DataFrame([{
        "p1_id": 4, "p1_display_name": "U. Humbert", "p1_historical_name": "Ugo Humbert",
        "p2_id": 625, "p2_display_name": "K. Nishikori", "p2_historical_name": "Kei Nishikori",
    }, {
        "p1_id": 13352, "p1_display_name": "H. Rune", "p1_historical_name": "Holger Rune",
        "p2_id": 1575, "p2_display_name": "Kyrian Jacquet", "p2_historical_name": "Kyrian Jacquet",
    }])

    corrected = apply_player_id_name_overrides(fixtures)

    assert corrected.loc[0, "p1_display_name"] == "Ugo Humbert"
    assert corrected.loc[0, "p2_display_name"] == "Kei Nishikori"
    assert corrected.loc[1, "p1_display_name"] == "Holger Rune"


def test_scheduled_csv_refreshes_only_exact_matching_fixture_ranks(tmp_path):
    csv_path = tmp_path / "data" / "external" / "2026-atp-season.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame([{
        "date_timestamp": int(pd.Timestamp("2026-09-30T00:00:00Z").timestamp()),
        "tournament": "Tokyo ATP", "status": "SCHEDULED",
        "home_name": "Rune H.", "away_name": "Jacquet K.",
        "home_rank": 142, "away_rank": 107,
        "home_points": 400, "away_points": 581,
    }]).to_csv(csv_path, index=False)
    fixtures = pd.DataFrame([{
        "event_date": "2026-09-30", "tournament_name": "Tokyo",
        "p1_historical_name": "Holger Rune", "p2_historical_name": "Kyrian Jacquet",
        "p1_current_rank": None, "p2_current_rank": 108,
        "p1_current_rank_points": None, "p2_current_rank_points": None,
    }, {
        "event_date": "2026-10-01", "tournament_name": "Tokyo",
        "p1_historical_name": "Holger Rune", "p2_historical_name": "Kyrian Jacquet",
        "p1_current_rank": None, "p2_current_rank": None,
        "p1_current_rank_points": None, "p2_current_rank_points": None,
    }])

    enriched = fill_missing_fixture_ranks(fixtures, tmp_path)

    assert enriched.loc[0, "p1_current_rank"] == 142
    assert enriched.loc[0, "p1_current_rank_points"] == 400
    assert enriched.loc[0, "p2_current_rank"] == 107
    assert enriched.loc[0, "p2_current_rank_points"] == 581
    assert pd.isna(enriched.loc[1, "p1_current_rank"])
