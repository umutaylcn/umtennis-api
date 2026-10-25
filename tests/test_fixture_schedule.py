from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.fixture_schedule import reconcile_fixture_schedule


def test_official_schedule_overrides_provider_and_csv_times(tmp_path):
    external = tmp_path / "data" / "external"
    external.mkdir(parents=True)
    pd.DataFrame([{
        "date_timestamp": int(pd.Timestamp("2026-10-26T12:30:00Z").timestamp()),
        "tournament": "Basel ATP (indoor)", "status": "SCHEDULED",
        "home_name": "Vacherot V.", "away_name": "Navone M.",
    }]).to_csv(external / "2026-atp-season.csv", index=False)
    manual = tmp_path / "data" / "manual"
    manual.mkdir(parents=True)
    (manual / "verified_schedule.json").write_text(json.dumps({"fixtures": [{
        "tournament": "Basel", "players": ["Valentin Vacherot", "Mariano Navone"],
        "start_time_utc": "2026-10-26T12:30:00Z", "precision": "time",
    }]}), encoding="utf-8")
    fixtures = pd.DataFrame([{
        "tournament_name": "Basel", "p1_display_name": "Valentin Vacherot",
        "p2_display_name": "Mariano Navone", "event_date": "2026-10-27",
        "start_time_utc": "2026-10-27T09:00:00Z",
    }])
    result = reconcile_fixture_schedule(fixtures, tmp_path).iloc[0]
    assert result.start_time_utc == pd.Timestamp("2026-10-26T12:30:00Z")
    assert result.event_date == "2026-10-26"
    assert result.schedule_precision == "time"


def test_mass_0900_csv_and_provider_times_are_not_publishable(tmp_path):
    external = tmp_path / "data" / "external"
    external.mkdir(parents=True)
    names = [(f"Player {index} A", f"Player {index} B") for index in range(6)]
    pd.DataFrame([{
        "date_timestamp": int(pd.Timestamp("2026-10-27T09:00:00Z").timestamp()),
        "tournament": "Vienna ATP", "status": "SCHEDULED",
        "home_name": left, "away_name": right,
    } for left, right in names]).to_csv(external / "2026-atp-season.csv", index=False)
    fixtures = pd.DataFrame([{
        "tournament_name": "Vienna", "p1_display_name": left,
        "p2_display_name": right, "event_date": "2026-10-27",
        "start_time_utc": "2026-10-27T09:00:00Z",
    } for left, right in names])
    result = reconcile_fixture_schedule(fixtures, tmp_path)
    assert set(result.schedule_precision) == {"tba"}


def test_csv_day_replaces_placeholder_but_clock_remains_unverified(tmp_path):
    external = tmp_path / "data" / "external"
    external.mkdir(parents=True)
    pd.DataFrame([{
        "date_timestamp": int(pd.Timestamp("2026-10-26T19:00:00Z").timestamp()),
        "tournament": "Basel ATP", "status": "SCHEDULED",
        "home_name": "Stricker D.", "away_name": "Ruud C.",
    }]).to_csv(external / "2026-atp-season.csv", index=False)
    fixtures = pd.DataFrame([{
        "tournament_name": "Basel", "p1_display_name": "Dominic Stricker",
        "p2_display_name": "Casper Ruud", "event_date": "2026-10-27",
        "start_time_utc": "2026-10-27T09:00:00Z",
    }])
    result = reconcile_fixture_schedule(fixtures, tmp_path).iloc[0]
    assert result.start_time_utc == pd.Timestamp("2026-10-26T19:00:00Z")
    assert result.schedule_precision == "date_only"
