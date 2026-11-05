"""Block publication when ATP main-draw fixture coverage is inconsistent."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.fixture_pipeline import exclude_csv_qualifying_fixtures
from tennis_ai.live_data import EXCLUDED_TOURNAMENT_MARKERS, is_pre_main_draw
from tennis_ai.results_backfill import ROUND_MAP, clean_tournament_name, player_identity_key


def missing_scheduled_csv_fixtures(
    source: pd.DataFrame, fixtures: pd.DataFrame, now: datetime
) -> list[tuple[str, str, str, str]]:
    """Catch source-backed main-draw matches dropped by identity matching."""
    snapshot_keys = {
        (
            str(row.event_date)[:10], clean_tournament_name(row.tournament_name),
            str(row.round),
            frozenset((player_identity_key(row.p1_display_name), player_identity_key(row.p2_display_name))),
        )
        for row in fixtures.itertuples(index=False)
    } if not fixtures.empty else set()
    missing = []
    for row in source.itertuples(index=False):
        if row.tour_type != 1 or row.status != "SCHEDULED":
            continue
        start = pd.to_datetime(row.date_timestamp, unit="s", utc=True, errors="coerce")
        if pd.isna(start) or not now - timedelta(hours=3) <= start <= now + timedelta(days=8):
            continue
        tournament = clean_tournament_name(row.tournament)
        if any(marker in tournament.casefold() for marker in EXCLUDED_TOURNAMENT_MARKERS):
            continue
        if is_pre_main_draw(tournament, start):
            continue
        round_code = ROUND_MAP.get(str(row.round))
        if round_code is None or str(row.surface).casefold() not in {"hard", "clay", "grass"}:
            continue
        players = frozenset((player_identity_key(row.home_name), player_identity_key(row.away_name)))
        if len(players) != 2 or "" in players:
            continue
        key = (start.date().isoformat(), tournament, round_code, players)
        if key not in snapshot_keys:
            missing.append((tournament, round_code, str(row.home_name), str(row.away_name)))
    return missing


def _placeholder_clock_exposed(group: pd.DataFrame, time: str) -> bool:
    """Date-only rows can use 09:00 for sorting without showing that time."""
    return (
        len(group) >= 5
        and str(time).endswith("T09:00")
        and group["precision"].isin({"time", "not_before"}).any()
    )


def main() -> int:
    source = pd.read_csv(ROOT / "data" / "external" / "2026-atp-season.csv", low_memory=False)
    snapshot = json.loads((ROOT / "data" / "cache" / "upcoming_fixtures.json").read_text(encoding="utf-8"))
    fixtures = pd.DataFrame(snapshot.get("fixtures", []))
    if fixtures.empty:
        print("AUDIT: no upcoming fixtures in snapshot")
    leaked = len(fixtures) - len(exclude_csv_qualifying_fixtures(fixtures, ROOT))
    if leaked:
        print(f"AUDIT FAIL: {leaked} qualifying fixtures entered the main-draw snapshot")
    valid_precision = {"time", "not_before", "date_only", "tba"}
    invalid_schedule = fixtures[
        ~fixtures["schedule_precision"].isin(valid_precision)
    ] if not fixtures.empty and "schedule_precision" in fixtures else fixtures
    if not fixtures.empty and ("schedule_precision" not in fixtures or not invalid_schedule.empty):
        print("AUDIT FAIL: upcoming fixtures lack a valid schedule precision label")
        leaked += len(invalid_schedule) or 1
    if not fixtures.empty and "schedule_precision" in fixtures:
        times = pd.to_datetime(fixtures["start_time_utc"], utc=True, errors="coerce")
        batched = pd.DataFrame({
            "event": fixtures["tournament_name"],
            "time": times.dt.strftime("%Y-%m-%dT%H:%M"),
            "precision": fixtures["schedule_precision"],
        })
        for (event, time), group in batched.groupby(["event", "time"]):
            if _placeholder_clock_exposed(group, time):
                print(f"AUDIT FAIL: {event} {time} placeholder displayed as a schedule")
                leaked += 1
    missing_round = fixtures[
        ~fixtures["round"].fillna("").astype(str).str.upper().isin(
            {"R128", "R64", "R32", "R16", "QF", "SF", "F", "RR"}
        )
    ] if not fixtures.empty else fixtures
    if not missing_round.empty:
        print(f"AUDIT REVIEW: {len(missing_round)} fixtures have no verified main-draw round")
    needs_review = not missing_round.empty
    missing_csv = missing_scheduled_csv_fixtures(source, fixtures, datetime.now(timezone.utc))
    for event, round_code, player1, player2 in missing_csv:
        print(f"AUDIT FAIL: scheduled CSV fixture missing: {event} {round_code} {player1} vs {player2}")
    leaked += len(missing_csv)

    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=3))
    recent = source[
        source["tour_type"].eq(1)
        & source["status"].eq("FINISHED")
        & source["round"].eq("Quarter-finals")
        & pd.to_datetime(source["date_timestamp"], unit="s", utc=True).ge(cutoff)
    ]
    quarterfinals: dict[str, set[frozenset[str]]] = defaultdict(set)
    for row in recent.itertuples(index=False):
        event = clean_tournament_name(row.tournament)
        quarterfinals[event].add(frozenset((
            player_identity_key(row.home_name), player_identity_key(row.away_name)
        )))
    for event, matches in sorted(quarterfinals.items()):
        if len(matches) != 4:
            continue
        csv_sf = source[
            source["tournament"].map(clean_tournament_name).eq(event)
            & source["round"].eq("Semi-finals")
            & source["status"].isin({"SCHEDULED", "FINISHED"})
        ]
        snapshot_sf = fixtures[
            fixtures["tournament_name"].map(clean_tournament_name).eq(event)
            & fixtures["round"].eq("SF")
        ] if not fixtures.empty else fixtures
        known = {
            frozenset((player_identity_key(row.home_name), player_identity_key(row.away_name)))
            for row in csv_sf.itertuples(index=False)
        }
        known.update(
            frozenset((player_identity_key(row.p1_display_name), player_identity_key(row.p2_display_name)))
            for row in snapshot_sf.itertuples(index=False)
        )
        if len(known) < 2:
            print(f"AUDIT REVIEW: {event} has 4 completed QFs but only {len(known)}/2 known semifinals")
            needs_review = True
        else:
            print(f"AUDIT OK: {event} semifinal coverage {len(known)}/2")
        finished_sf = csv_sf[csv_sf["status"].eq("FINISHED")]
        if len(finished_sf) == 2:
            finals = source[
                source["tournament"].map(clean_tournament_name).eq(event)
                & source["round"].eq("Final")
                & source["status"].isin({"SCHEDULED", "FINISHED"})
            ]
            snapshot_final = fixtures[
                fixtures["tournament_name"].map(clean_tournament_name).eq(event)
                & fixtures["round"].eq("F")
            ] if not fixtures.empty else fixtures
            if finals.empty and snapshot_final.empty:
                print(f"AUDIT REVIEW: {event} has two completed SFs but no final fixture or result")
                needs_review = True
            else:
                print(f"AUDIT OK: {event} final is present after both SFs")
                winners = {
                    player_identity_key(row.home_name if str(row.winner_code) == "1" else row.away_name)
                    for row in finished_sf.itertuples(index=False)
                    if str(row.winner_code) in {"1", "2"}
                }
                final_pairs = {
                    frozenset((player_identity_key(row.home_name), player_identity_key(row.away_name)))
                    for row in finals.itertuples(index=False)
                }
                final_pairs.update(
                    frozenset((player_identity_key(row.p1_display_name), player_identity_key(row.p2_display_name)))
                    for row in snapshot_final.itertuples(index=False)
                )
                if len(winners) == 2 and frozenset(winners) not in final_pairs:
                    print(f"AUDIT REVIEW: {event} final entrants do not match the SF winners")
                    needs_review = True
    if not leaked:
        print(f"AUDIT OK: {len(fixtures)} upcoming fixtures; no CSV-confirmed qualifying leaks")
    return 1 if leaked or needs_review else 0


if __name__ == "__main__":
    raise SystemExit(main())
