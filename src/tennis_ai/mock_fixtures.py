"""Deterministic, model-ready mock ATP fixtures for the UMTennis UI."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .player_matching import HistoricalPlayerMatcher


MOCK_TOURNAMENTS = (
    ("Cincinnati", "Hard"),
    ("Monte Carlo", "Clay"),
    ("Halle", "Grass"),
    ("Montreal", "Hard"),
    ("Wimbledon", "Grass"),
)
MOCK_ROUNDS = ("QF", "SF", "F", "R32", "R16")


def _known_profiles(root: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    cache_path = root / "data" / "cache" / "live_players.json"
    if not cache_path.exists():
        return {}
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    matcher = HistoricalPlayerMatcher(names)
    profiles: dict[str, dict[str, Any]] = {}
    for raw_id, profile in cache.items():
        matched, _, _ = matcher.match(str(profile.get("name") or ""))
        if matched is not None:
            profiles[matched] = {**profile, "id": int(raw_id)}
    return profiles


def _prefer(primary: Any, fallback: Any) -> Any:
    return primary if primary is not None and not pd.isna(primary) else fallback


def build_mock_fixture_table(
    project_root: str | Path,
    state: Any,
    elo_leaderboard: list[str],
    *,
    offset: int = 0,
    limit: int = 20,
) -> pd.DataFrame:
    """Pair adjacent Elo ranks in a stable daily mock schedule."""
    root = Path(project_root)
    names = elo_leaderboard[offset : offset + limit]
    if len(names) % 2:
        names = names[:-1]
    profiles = _known_profiles(root, names)

    today = pd.Timestamp.now(tz="UTC").normalize()
    rng = random.Random(int(today.strftime("%Y%m%d")) + offset * 101)
    first_start = today + pd.Timedelta(days=1, hours=10)
    rows: list[dict[str, Any]] = []

    for pair_index in range(0, len(names), 2):
        match_index = pair_index // 2
        pair = [names[pair_index], names[pair_index + 1]]
        if rng.random() < 0.5:
            pair.reverse()
        p1_name, p2_name = pair
        p1_profile = profiles.get(p1_name, {})
        p2_profile = profiles.get(p2_name, {})
        p1_state, p2_state = state.players[p1_name], state.players[p2_name]
        tournament_name, surface = rng.choice(MOCK_TOURNAMENTS)
        start = first_start + pd.Timedelta(minutes=95 * match_index)

        rows.append(
            {
                "match_id": 9_000_000 + offset * 100 + match_index + 1,
                "event_date": start.date().isoformat(),
                "start_time_utc": start,
                "tournament_name": tournament_name,
                "surface": surface,
                # Keep every visual tier visible together during UI QA.
                "round": MOCK_ROUNDS[match_index % len(MOCK_ROUNDS)],
                "p1_display_name": p1_name,
                "p2_display_name": p2_name,
                "p1_id": p1_profile.get("id"),
                "p2_id": p2_profile.get("id"),
                "p1_historical_name": p1_name,
                "p2_historical_name": p2_name,
                "p1_match_status": "mock_elo_rank",
                "p2_match_status": "mock_elo_rank",
                "p1_match_score": 1.0,
                "p2_match_score": 1.0,
                "p1_current_rank": _prefer(p1_profile.get("ranking"), p1_state.rank),
                "p2_current_rank": _prefer(p2_profile.get("ranking"), p2_state.rank),
                "p1_current_rank_points": _prefer(
                    p1_profile.get("ranking_points"), p1_state.rank_points
                ),
                "p2_current_rank_points": _prefer(
                    p2_profile.get("ranking_points"), p2_state.rank_points
                ),
                "p1_hand": _prefer(p1_profile.get("hand"), p1_state.hand),
                "p2_hand": _prefer(p2_profile.get("hand"), p2_state.hand),
                "p1_birthday": p1_profile.get("birthday"),
                "p2_birthday": p2_profile.get("birthday"),
                "p1_country": p1_profile.get("country"),
                "p2_country": p2_profile.get("country"),
                "identities_resolved": True,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("start_time_utc").reset_index(drop=True)
    return table
