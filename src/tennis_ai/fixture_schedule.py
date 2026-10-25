"""Separate a fixture's internal sort time from a verified public start time."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .results_backfill import clean_tournament_name, player_identity_key


def _identity(tournament: object, first: object, second: object) -> tuple[str, frozenset[str]]:
    return clean_tournament_name(tournament), frozenset(
        (player_identity_key(first), player_identity_key(second))
    )


def _placeholder_keys(rows: pd.DataFrame, tournament_column: str, time_column: str) -> set[tuple[str, str]]:
    """The feeds batch unscheduled matches at 09:00 UTC rather than assigning a slot."""
    if rows.empty:
        return set()
    times = pd.to_datetime(rows[time_column], utc=True, errors="coerce")
    keys = pd.DataFrame({
        "tournament": rows[tournament_column].map(clean_tournament_name),
        "time": times.dt.strftime("%Y-%m-%dT%H:%M"),
    })
    counts = keys.value_counts()
    return {
        (tournament, time)
        for (tournament, time), count in counts.items()
        if count >= 5 and time.endswith("T09:00")
    }


def reconcile_fixture_schedule(table: pd.DataFrame, project_root: str | Path) -> pd.DataFrame:
    """Prefer official order of play, then CSV date, never a mass placeholder as a clock time.

    `start_time_utc` remains usable for chronological processing. The separate
    `schedule_precision` field tells the UI whether a clock time is publishable.
    """
    if table.empty:
        return table.copy()
    root = Path(project_root)
    corrected = table.copy()
    corrected["start_time_utc"] = pd.to_datetime(
        corrected["start_time_utc"], utc=True, errors="coerce"
    )
    corrected["schedule_precision"] = "date_only"

    csv_path = root / "data" / "external" / "2026-atp-season.csv"
    scheduled = pd.DataFrame()
    if csv_path.exists():
        scheduled = pd.read_csv(
            csv_path,
            usecols=["date_timestamp", "tournament", "status", "home_name", "away_name"],
            low_memory=False,
        )
        scheduled = scheduled[scheduled["status"].eq("SCHEDULED")].copy()
        scheduled["start"] = pd.to_datetime(
            scheduled["date_timestamp"], unit="s", utc=True, errors="coerce"
        )
    csv_placeholders = _placeholder_keys(scheduled, "tournament", "start")
    csv_by_pair: dict[tuple[str, frozenset[str]], list[pd.Timestamp]] = {}
    for row in scheduled.itertuples(index=False):
        if pd.notna(row.start):
            csv_by_pair.setdefault(
                _identity(row.tournament, row.home_name, row.away_name), []
            ).append(row.start)

    overrides_path = root / "data" / "manual" / "verified_schedule.json"
    overrides = {}
    if overrides_path.exists():
        for item in json.loads(overrides_path.read_text(encoding="utf-8")).get("fixtures", []):
            overrides[_identity(item["tournament"], *item["players"])] = item

    provider_placeholders = _placeholder_keys(
        corrected, "tournament_name", "start_time_utc"
    )
    for index, fixture in corrected.iterrows():
        identity = _identity(
            fixture["tournament_name"], fixture["p1_display_name"], fixture["p2_display_name"]
        )
        original = pd.to_datetime(fixture["start_time_utc"], utc=True, errors="coerce")
        override = overrides.get(identity)
        if override is not None:
            verified_start = pd.Timestamp(override["start_time_utc"])
            if pd.notna(original) and abs(verified_start - original) <= pd.Timedelta(days=3):
                corrected.at[index, "start_time_utc"] = verified_start
                corrected.at[index, "event_date"] = verified_start.date().isoformat()
                corrected.at[index, "schedule_precision"] = override["precision"]
                continue

        candidates = csv_by_pair.get(identity, [])
        if pd.notna(original):
            candidates = [start for start in candidates if abs(start - original) <= pd.Timedelta(days=2)]
        if len(candidates) == 1:
            csv_start = candidates[0]
            corrected.at[index, "start_time_utc"] = csv_start
            corrected.at[index, "event_date"] = csv_start.date().isoformat()
            if (identity[0], csv_start.strftime("%Y-%m-%dT%H:%M")) in csv_placeholders:
                corrected.at[index, "schedule_precision"] = "tba"
            else:
                corrected.at[index, "schedule_precision"] = "date_only"
        elif pd.isna(original) or (
            identity[0], original.strftime("%Y-%m-%dT%H:%M")
        ) in provider_placeholders:
            corrected.at[index, "schedule_precision"] = "tba"
    corrected["start_time_utc"] = pd.to_datetime(
        corrected["start_time_utc"], utc=True, errors="coerce"
    )
    return corrected
