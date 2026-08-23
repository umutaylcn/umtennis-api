"""Build a model-ready identity table from upcoming ATP fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .live_data import LiveTennisClient, UpcomingMatch
from .player_matching import HistoricalPlayerMatcher, PlayerProfileCache
from .result_tracker import TrackedFixtureStore


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
    historical_name, score, status = matcher.match(full_name)
    # A genuinely new tour player has no row in the historical model data yet.
    # Keep authoritative full provider names as cold-start identities, while
    # continuing to reject close/ambiguous matches that could merge two players.
    if historical_name is None and status == "unresolved" and score < 0.70:
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
) -> pd.DataFrame:
    root = Path(project_root)
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
        "p1_display_name": p1["historical_name"] or p1["provider_full_name"],
        "p2_display_name": p2["historical_name"] or p2["provider_full_name"],
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
