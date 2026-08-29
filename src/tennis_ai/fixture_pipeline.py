"""Build a model-ready identity table from upcoming ATP fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from .live_data import LiveTennisClient, UpcomingMatch
from .player_matching import (
    HistoricalPlayerMatcher,
    PlayerProfileCache,
    normalize_player_name,
)
from .result_tracker import TrackedFixtureStore


FIXTURE_SNAPSHOT_NAME = "upcoming_fixtures.json"

DISPLAY_NAME_ALIASES = {
    "a molcan": "Alex Molcan",
    "f cina": "Federico Cina",
    "luca van assche": "Luca Van Assche",
    "pablo carreno busta": "Pablo Carreno Busta",
    "s baez": "Sebastian Baez",
    "tomas vera barrios": "Tomas Barrios Vera",
    "v kopriva": "Vit Kopriva",
}


def canonical_display_name(name: object) -> str:
    value = str(name).strip()
    return DISPLAY_NAME_ALIASES.get(normalize_player_name(value), value)


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
    serializable = table.copy()
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
    if profile is None and player_id is not None:
        profile = client.get_player(player_id)
        cache.set(player_id, profile)

    full_name = str((profile or {}).get("name") or fallback_name).strip()
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
    fixtures = client.get_upcoming_matches()

    rows: list[dict[str, object]] = []
    for fixture in fixtures:
        p1 = _resolve_player(
            fixture.p1_id, fixture.p1_name, client, cache, matcher
        )
        p2 = _resolve_player(
            fixture.p2_id, fixture.p2_name, client, cache, matcher
        )
        rows.append(_fixture_row(fixture, p1, p2))

    table = pd.DataFrame(rows)
    if not table.empty:
        table["start_time_utc"] = pd.to_datetime(
            table["start_time_utc"], utc=True, errors="coerce"
        )
        table = table.sort_values(
            ["start_time_utc", "tournament_name"], na_position="last"
        ).reset_index(drop=True)
        TrackedFixtureStore(
            root / "data" / "cache" / "tracked_fixtures.json"
        ).track(table)
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
        "round": fixture.round_code,
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
