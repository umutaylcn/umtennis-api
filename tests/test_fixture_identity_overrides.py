from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.fixture_pipeline import apply_player_id_name_overrides


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
