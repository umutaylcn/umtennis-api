"""Build a model-ready identity table from upcoming ATP fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from .live_data import (
    EXCLUDED_TOURNAMENT_MARKERS, MAIN_DRAW_START_DATES, LiveTennisClient,
    TennisAPIError, UpcomingMatch, is_pre_main_draw,
)
from .player_matching import (
    HistoricalPlayerMatcher,
    PlayerProfileCache,
    normalize_player_name,
)
from .result_tracker import TrackedFixtureStore
from .results_backfill import ROUND_MAP, clean_tournament_name, player_identity_key


FIXTURE_SNAPSHOT_NAME = "upcoming_fixtures.json"

# Provider round fields are missing for these exact fixtures. Source-backed
# fallbacks are intentionally fixture-specific rather than tournament-wide.
# Laver Cup: https://lavercup.com/news/2026/09/24/laver-cup-2026-day-1-lineups-ruud-and-cerundolo-open-at-the-o2
# Chengdu: https://www.protennislive.com/posting/2026/7581/mds.pdf
# Hangzhou: https://www.hangzhouopen.com/en/scores/draw
# Chengdu SF: https://www.protennislive.com/posting/2026/7581/mds.pdf
# Shanghai R4: https://en.rolexshanghaimasters.com/en/media/news/jodar-lehecka-shanghai-2026-monday
# Shanghai R4: https://en.rolexshanghaimasters.com/en/media/news/day-6-preview-alcaraz-vacherot-tsitsipas-v-fritz
# Brussels R32: https://www.protennislive.com/posting/2026/7485/mds.pdf
# Brussels R16: https://www.protennislive.com/posting/2026/7485/mds.pdf
VERIFIED_FIXTURE_ROUNDS = {
    195478: ("ATP Laver Cup", "Casper Ruud", "Francisco Cerundolo", "DAY 1"),
    195479: ("ATP Laver Cup", "Jakub Menšik", "Brandon Nakashima", "DAY 1"),
    195480: ("ATP Laver Cup", "Rafael Jodar", "Alexander Bublik", "DAY 1"),
    195725: ("Chengdu", "Jenson Brooksby", "Nikoloz Basilashvili", "QF"),
    195741: ("Hangzhou", "Fabian Marozsan", "Kyrian Jacquet", "QF"),
    196548: ("Chengdu", "Hubert Hurkacz", "Denis Shapovalov", "SF"),
    203161: ("Shanghai", "Botic Van De Zandschulp", "Rafael Jodar", "R16"),
    203160: ("Shanghai", "Tommy Paul", "Zizou Bergs", "R16"),
    206066: ("Brussels", "Federico Cinà", "Jaume Munar", "R32"),
    206067: ("Brussels", "Yannick Hanfmann", "Martin Damm", "R32"),
    206068: ("Brussels", "Andrea Guerrieri", "Alex Michelsen", "R32"),
    206063: ("Brussels", "Nishesh Basavareddy", "Raphael Collignon", "R32"),
    206064: ("Brussels", "David Goffin", "Remy Bertola", "R32"),
    206065: ("Brussels", "Botic Van De Zandschulp", "Clement Chidekh", "R32"),
    207258: ("Brussels", "Mariano Navone", "Tallon Griekspoor", "R16"),
    207369: ("Brussels", "Clement Chidekh", "Jacob Fearnley", "R16"),
    207300: ("Brussels", "Gauthier Onclin", "Jan-Lennard Struff", "R16"),
}

DISPLAY_NAME_ALIASES = {
    "t skatov": "Timofey Skatov",
    "u humbert": "Ugo Humbert",
    "k nishikori": "Kei Nishikori",
    "h rune": "Holger Rune",
    "a molcan": "Alex Molcan",
    "f cina": "Federico Cina",
    "luca van assche": "Luca Van Assche",
    "pablo carreno busta": "Pablo Carreno Busta",
    "s baez": "Sebastian Baez",
    "tomas vera barrios": "Tomas Barrios Vera",
    "v kopriva": "Vit Kopriva",
}

# The live provider currently labels ATP player 445 as "Ying Zhang", which is
# a different (WTA) player.  IDs are stable, so correct the identity before
# historical matching, presentation, and photo lookup.
PLAYER_ID_NAME_OVERRIDES = {
    183: "Timofey Skatov",
    445: "Zhizhen Zhang",
}


def apply_player_id_name_overrides(table: pd.DataFrame) -> pd.DataFrame:
    """Repair known provider identity collisions in fresh and cached fixtures."""
    corrected = table.copy()
    for side in ("p1", "p2"):
        id_column = f"{side}_id"
        if id_column not in corrected:
            continue
        for player_id, name in PLAYER_ID_NAME_OVERRIDES.items():
            mask = pd.to_numeric(corrected[id_column], errors="coerce").eq(player_id)
            for column in (f"{side}_display_name", f"{side}_historical_name"):
                if column in corrected:
                    corrected.loc[mask, column] = name
        display_column = f"{side}_display_name"
        if display_column in corrected:
            corrected[display_column] = corrected[display_column].map(canonical_display_name)
    return corrected


def canonical_display_name(name: object) -> str:
    value = str(name).strip()
    return DISPLAY_NAME_ALIASES.get(normalize_player_name(value), value)


def fill_missing_fixture_ranks(table: pd.DataFrame, project_root: str | Path) -> pd.DataFrame:
    """Enrich an exact scheduled ATP match with CSV ranks and a missing round."""
    if table.empty:
        return table.copy()
    csv_path = Path(project_root) / "data" / "external" / "2026-atp-season.csv"
    if not csv_path.exists():
        return table.copy()
    columns = [
        "date_timestamp", "tournament", "round", "status", "home_name", "away_name",
        "home_rank", "away_rank", "home_points", "away_points",
    ]
    scheduled = pd.read_csv(csv_path, usecols=lambda column: column in columns)
    scheduled = scheduled[scheduled["status"].eq("SCHEDULED")]
    by_fixture: dict[
        tuple[str, str, tuple[str, str]],
        tuple[dict[str, tuple[object, object]], str | None] | None,
    ] = {}
    for row in scheduled.itertuples(index=False):
        home_key = player_identity_key(row.home_name)
        away_key = player_identity_key(row.away_name)
        if not home_key or not away_key or home_key == away_key:
            continue
        event_date = pd.to_datetime(row.date_timestamp, unit="s", utc=True, errors="coerce")
        if pd.isna(event_date):
            continue
        key = (
            event_date.date().isoformat(),
            clean_tournament_name(row.tournament),
            tuple(sorted((home_key, away_key))),
        )
        values = {
            home_key: (row.home_rank, row.home_points),
            away_key: (row.away_rank, row.away_points),
        }
        by_fixture[key] = None if key in by_fixture else (
            values, ROUND_MAP.get(str(getattr(row, "round", None)))
        )

    enriched = table.copy()
    for index, fixture in enriched.iterrows():
        players = [player_identity_key(fixture[f"{side}_historical_name"]) for side in ("p1", "p2")]
        event_date = str(fixture.get("event_date") or "")[:10]
        if not event_date or event_date == "nan" or not all(players):
            continue
        key = (event_date, clean_tournament_name(fixture["tournament_name"]), tuple(sorted(players)))
        match_data = by_fixture.get(key)
        if match_data is None:
            continue
        values, csv_round = match_data
        for side, player_key in zip(("p1", "p2"), players):
            rank, points = values[player_key]
            if pd.notna(rank):
                enriched.at[index, f"{side}_current_rank"] = rank
            if pd.notna(points):
                enriched.at[index, f"{side}_current_rank_points"] = points
        current_round = str(fixture.get("round") or "").strip().upper()
        if csv_round and current_round in {"", "NAN", "NONE", "NULL", "TBD"}:
            enriched.at[index, "round"] = csv_round
    return enriched


def load_scheduled_csv_fixtures(
    project_root: str | Path,
    matcher: HistoricalPlayerMatcher,
    *,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Supplement API gaps with genuine, still-scheduled ATP CSV fixtures.

    CSV match IDs are kept so the later CSV result can finalize the captured
    prediction. Unresolvable abbreviated names are never guessed.
    """
    csv_path = Path(project_root) / "data" / "external" / "2026-atp-season.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    columns = [
        "match_id", "tour_type", "date_timestamp", "tournament", "round",
        "surface", "status", "home_name", "away_name", "home_rank",
        "away_rank", "home_points", "away_points",
    ]
    source = pd.read_csv(csv_path, usecols=columns, low_memory=False)
    reference = pd.Timestamp(now_utc or datetime.now(timezone.utc))
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    else:
        reference = reference.tz_convert("UTC")
    source = source[
        source["tour_type"].eq(1)
        & source["status"].eq("SCHEDULED")
        & source["surface"].astype(str).str.casefold().isin({"hard", "clay", "grass"})
    ]

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in source.itertuples(index=False):
        start = pd.to_datetime(row.date_timestamp, unit="s", utc=True, errors="coerce")
        if pd.isna(start) or not reference - timedelta(hours=3) <= start <= reference + timedelta(days=8):
            continue
        tournament = clean_tournament_name(row.tournament)
        if any(marker in tournament.casefold() for marker in EXCLUDED_TOURNAMENT_MARKERS):
            continue
        if is_pre_main_draw(tournament, start):
            continue
        round_code = ROUND_MAP.get(str(row.round))
        if round_code is None:
            continue
        p1_name, p1_score, p1_status = matcher.match_surname_initial(str(row.home_name))
        p2_name, p2_score, p2_status = matcher.match_surname_initial(str(row.away_name))
        if not p1_name or not p2_name or p1_name == p2_name:
            continue
        identity = (
            tournament, round_code,
            min(player_identity_key(p1_name), player_identity_key(p2_name)),
            max(player_identity_key(p1_name), player_identity_key(p2_name)),
        )
        if identity in seen:
            continue
        seen.add(identity)
        fixture = UpcomingMatch(
            match_id=int(row.match_id), event_date=start.date().isoformat(),
            start_time=start.isoformat(), p1_id=None, p1_name=p1_name,
            p2_id=None, p2_name=p2_name, tournament_name=tournament,
            surface=str(row.surface).title(), round_name=None,
            round_code=round_code, status="scheduled", is_qualifying=False,
        )
        profiles = []
        for name, score, status, rank, points in (
            (p1_name, p1_score, p1_status, row.home_rank, row.home_points),
            (p2_name, p2_score, p2_status, row.away_rank, row.away_points),
        ):
            profiles.append({
                "provider_full_name": name, "historical_name": name,
                "match_score": round(score, 4), "match_status": status,
                "current_rank": rank if pd.notna(rank) else None,
                "current_rank_points": points if pd.notna(points) else None,
                "hand": None, "birthday": None, "country": None,
            })
        rows.append(_fixture_row(fixture, profiles[0], profiles[1]))
    return pd.DataFrame(rows)


def _same_scheduled_match(left: pd.Series, right: pd.Series) -> bool:
    if clean_tournament_name(left["tournament_name"]) != clean_tournament_name(right["tournament_name"]):
        return False
    left_players = {player_identity_key(left[f"{side}_historical_name"]) for side in ("p1", "p2")}
    right_players = {player_identity_key(right[f"{side}_historical_name"]) for side in ("p1", "p2")}
    if "" in left_players or left_players != right_players:
        return False
    left_round, right_round = left.get("round"), right.get("round")
    if pd.notna(left_round) and pd.notna(right_round) and left_round != right_round:
        return False
    return abs(pd.Timestamp(left["start_time_utc"]) - pd.Timestamp(right["start_time_utc"])) <= pd.Timedelta(days=2)


def fixture_round_code(code: object, name: object) -> str | None:
    """Use provider round names when its compact code is absent; never guess a stage."""
    known_codes = {"R128", "R64", "R32", "R16", "QF", "SF", "F", "RR"}
    compact = str(code or "").strip().upper()
    if compact in known_codes:
        return compact
    normalized = str(name or "").strip().casefold()
    names = {
        "round of 128": "R128", "round of 64": "R64",
        "round of 32": "R32", "round of 16": "R16",
        "quarterfinal": "QF", "quarterfinals": "QF",
        "semi-final": "SF", "semi-finals": "SF",
        "semifinal": "SF", "semifinals": "SF",
        "final": "F", "round robin": "RR",
    }
    return names.get(normalized)


def verified_fixture_round(
    match_id: object, tournament: object, p1_name: object, p2_name: object
) -> str | None:
    """Return a verified fallback only when the match identity also agrees."""
    try:
        expected = VERIFIED_FIXTURE_ROUNDS.get(int(match_id))
    except (TypeError, ValueError):
        return None
    if expected is None:
        return None
    expected_tournament, expected_p1, expected_p2, round_code = expected
    if normalize_player_name(tournament) != normalize_player_name(expected_tournament):
        return None
    actual_players = {normalize_player_name(p1_name), normalize_player_name(p2_name)}
    expected_players = {normalize_player_name(expected_p1), normalize_player_name(expected_p2)}
    return round_code if actual_players == expected_players else None


def apply_verified_fixture_rounds(table: pd.DataFrame) -> pd.DataFrame:
    """Repair missing round labels in previously saved fixture snapshots."""
    corrected = table.copy()
    required = {"match_id", "tournament_name", "p1_display_name", "p2_display_name", "round"}
    if not required.issubset(corrected.columns):
        return corrected
    for index, row in corrected.iterrows():
        current = row["round"]
        if pd.notna(current) and str(current).strip().upper() not in {"", "NAN", "NONE", "NULL", "TBD"}:
            continue
        verified = verified_fixture_round(
            row["match_id"], row["tournament_name"],
            row["p1_display_name"], row["p2_display_name"],
        )
        if verified is not None:
            corrected.at[index, "round"] = verified
    return corrected


def exclude_pre_main_draw_fixtures(table: pd.DataFrame) -> pd.DataFrame:
    """Apply verified qualifying windows to cached fixtures as well as live ones."""
    if table.empty or not {"event_date", "tournament_name"}.issubset(table.columns):
        return table.copy()
    event_days = pd.to_datetime(table["event_date"], errors="coerce").dt.date
    excluded = table["tournament_name"].str.contains(
        "qualification|qualifying", case=False, na=False
    )
    for (tournament, year), main_draw_start in MAIN_DRAW_START_DATES.items():
        excluded |= (
            table["tournament_name"].eq(tournament)
            & event_days.map(lambda day: day is not None and pd.notna(day) and day.year == year and day < main_draw_start)
        )
    return table.loc[~excluded].copy()


def fixture_snapshot_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / "cache" / FIXTURE_SNAPSHOT_NAME


def fixture_snapshot_is_fresh(
    project_root: str | Path,
    max_age_seconds: int,
    *,
    now_utc: datetime | None = None,
) -> bool:
    path = fixture_snapshot_path(project_root)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_at = pd.Timestamp(payload["saved_at_utc"])
        if saved_at.tzinfo is None:
            saved_at = saved_at.tz_localize("UTC")
        else:
            saved_at = saved_at.tz_convert("UTC")
    except (OSError, KeyError, ValueError, TypeError):
        return False
    reference = pd.Timestamp(now_utc or datetime.now(timezone.utc))
    age_seconds = (reference - saved_at).total_seconds()
    return 0 <= age_seconds < max_age_seconds


def save_fixture_snapshot(project_root: str | Path, table: pd.DataFrame) -> None:
    """Persist the last successful model-ready fixture response across restarts."""
    if table.empty:
        return
    path = fixture_snapshot_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = exclude_pre_main_draw_fixtures(
        apply_verified_fixture_rounds(apply_player_id_name_overrides(table))
    )
    serializable["start_time_utc"] = pd.to_datetime(
        serializable["start_time_utc"], utc=True, errors="coerce"
    ).map(lambda value: value.isoformat() if pd.notna(value) else None)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixtures": serializable.where(pd.notna(serializable), None).to_dict("records"),
    }
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def load_fixture_snapshot(
    project_root: str | Path,
    *,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Load still-relevant fixtures when the provider is unavailable or rate-limited."""
    path = fixture_snapshot_path(project_root)
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = pd.DataFrame(payload.get("fixtures", []))
    except (OSError, ValueError, TypeError):
        return pd.DataFrame()
    if table.empty or "start_time_utc" not in table:
        return pd.DataFrame()

    optional_columns = (
        "event_date",
        "p1_match_status",
        "p2_match_status",
        "p1_match_score",
        "p2_match_score",
        "p1_current_rank_points",
        "p2_current_rank_points",
        "p1_hand",
        "p2_hand",
        "p1_birthday",
        "p2_birthday",
        "p1_country",
        "p2_country",
    )
    for column in optional_columns:
        if column not in table:
            table[column] = None

    table["start_time_utc"] = pd.to_datetime(
        table["start_time_utc"], utc=True, errors="coerce"
    )
    reference = pd.Timestamp(now_utc or datetime.now(timezone.utc))
    # Keep a short grace period for matches whose scheduled start moved or was delayed.
    lower_bound = reference - timedelta(hours=3)
    upper_bound = reference + timedelta(days=8)
    table = table[
        table["start_time_utc"].between(lower_bound, upper_bound, inclusive="both")
    ]
    table = exclude_pre_main_draw_fixtures(
        apply_verified_fixture_rounds(apply_player_id_name_overrides(table))
    )
    table = fill_missing_fixture_ranks(table, project_root)
    return table.sort_values(
        ["start_time_utc", "tournament_name"], na_position="last"
    ).reset_index(drop=True)


def _resolve_player(
    player_id: int | None,
    fallback_name: str,
    client: LiveTennisClient,
    cache: PlayerProfileCache,
    matcher: HistoricalPlayerMatcher,
) -> dict[str, object]:
    profile = cache.get(player_id) if player_id is not None else None
    if player_id is not None and profile is None:
        # Reuse known identity and bio fields. The season CSV refreshes rankings
        # for exact scheduled matches without spending a call per player.
        try:
            profile = client.get_player(player_id)
            cache.set(player_id, profile)
        except TennisAPIError:
            # Fixture data still contains a usable provider identity. A
            # missing optional profile must not fail the full daily snapshot.
            profile = {"name": fallback_name}

    full_name = PLAYER_ID_NAME_OVERRIDES.get(
        player_id,
        str((profile or {}).get("name") or fallback_name).strip(),
    )
    historical_name, score, status = matcher.match_surname_initial(full_name)
    # A genuinely new tour player has no row in the historical model data yet.
    # Keep authoritative full provider names as cold-start identities, while
    # continuing to reject close/ambiguous matches that could merge two players.
    if historical_name is None and status == "unresolved":
        historical_name = full_name
        status = "new_player"
    return {
        "provider_full_name": full_name,
        "historical_name": historical_name,
        "match_score": round(score, 4),
        "match_status": status,
        "current_rank": (profile or {}).get("ranking"),
        "current_rank_points": (profile or {}).get("ranking_points"),
        "hand": (profile or {}).get("hand"),
        "birthday": (profile or {}).get("birthday"),
        "country": (profile or {}).get("country"),
    }


def build_upcoming_fixture_table(
    project_root: str | Path,
    client: LiveTennisClient,
    historical_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = Path(project_root)
    if historical_names is None:
        model_data = pd.read_pickle(
            root / "data" / "processed" / "atp_model_data_1990_2026.pkl"
        )
        historical_names = pd.concat(
            [model_data["p1_name"], model_data["p2_name"]],
            ignore_index=True,
        ).dropna()

    matcher = HistoricalPlayerMatcher(historical_names)
    cache = PlayerProfileCache(root / "data" / "cache" / "live_players.json")
    provider_error: TennisAPIError | None = None
    try:
        fixtures = client.get_upcoming_matches()
    except TennisAPIError as exc:
        provider_error = exc
        fixtures = []

    rows: list[dict[str, object]] = []
    for fixture in fixtures:
        p1 = _resolve_player(
            fixture.p1_id, fixture.p1_name, client, cache, matcher
        )
        p2 = _resolve_player(
            fixture.p2_id, fixture.p2_name, client, cache, matcher
        )
        rows.append(_fixture_row(fixture, p1, p2))

    provider_table = pd.DataFrame(rows)
    csv_table = load_scheduled_csv_fixtures(root, matcher)
    csv_only = [
        csv_row for _, csv_row in csv_table.iterrows()
        if provider_table.empty or not any(
            _same_scheduled_match(provider_row, csv_row)
            for _, provider_row in provider_table.iterrows()
        )
    ]
    print(f"Scheduled CSV fixtures not in live feed: {len(csv_only)}")
    if provider_error is not None:
        if provider_table.empty and not csv_only:
            raise provider_error
        print(f"Live fixture feed unavailable; using scheduled CSV: {provider_error}")
    table = pd.concat([provider_table, pd.DataFrame(csv_only)], ignore_index=True)
    if not table.empty:
        table["start_time_utc"] = pd.to_datetime(
            table["start_time_utc"], utc=True, errors="coerce"
        )
        table = fill_missing_fixture_ranks(table, root)
        table = table.sort_values(
            ["start_time_utc", "tournament_name"], na_position="last"
        ).reset_index(drop=True)
        if not provider_table.empty:
            TrackedFixtureStore(
                root / "data" / "cache" / "tracked_fixtures.json"
            ).track(provider_table)
        save_fixture_snapshot(root, table)
    return table


def _fixture_row(
    fixture: UpcomingMatch,
    p1: dict[str, object],
    p2: dict[str, object],
) -> dict[str, object]:
    return {
        "match_id": fixture.match_id,
        "event_date": fixture.event_date,
        "start_time_utc": fixture.start_time,
        "tournament_name": fixture.tournament_name,
        "surface": fixture.surface,
        "round": fixture_round_code(fixture.round_code, fixture.round_name) or verified_fixture_round(
            fixture.match_id, fixture.tournament_name,
            p1["provider_full_name"], p2["provider_full_name"],
        ),
        # Model lookup uses the historical identity, but the UI should retain
        # the current provider's canonical full name (for example John Jeffrey
        # Wolf rather than the archive abbreviation J J Wolf).
        "p1_display_name": canonical_display_name(p1["provider_full_name"]),
        "p2_display_name": canonical_display_name(p2["provider_full_name"]),
        "p1_id": fixture.p1_id,
        "p2_id": fixture.p2_id,
        "p1_historical_name": p1["historical_name"],
        "p2_historical_name": p2["historical_name"],
        "p1_match_status": p1["match_status"],
        "p2_match_status": p2["match_status"],
        "p1_match_score": p1["match_score"],
        "p2_match_score": p2["match_score"],
        "p1_current_rank": p1["current_rank"],
        "p2_current_rank": p2["current_rank"],
        "p1_current_rank_points": p1["current_rank_points"],
        "p2_current_rank_points": p2["current_rank_points"],
        "p1_hand": p1["hand"],
        "p2_hand": p2["hand"],
        "p1_birthday": p1["birthday"],
        "p2_birthday": p2["birthday"],
        "p1_country": p1["country"],
        "p2_country": p2["country"],
        "identities_resolved": (
            p1["historical_name"] is not None and p2["historical_name"] is not None
        ),
    }
