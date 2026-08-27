"""Convert the free current-season CSV into state-applicable main-tour updates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .player_matching import HistoricalPlayerMatcher


BACKFILL_START_UTC = pd.Timestamp("2026-06-08", tz="UTC")

ROUND_MAP = {
    "1/64-finals": "R128",
    "1/32-finals": "R64",
    "1/16-finals": "R32",
    "1/8-finals": "R16",
    "Quarter-finals": "QF",
    "Semi-finals": "SF",
    "Final": "F",
}

MASTERS_TOURNAMENTS = {"Montreal", "Cincinnati"}
GRAND_SLAMS = {"Wimbledon"}

DRAW_SIZE_MAP = {
    "Wimbledon": 128,
    "Montreal": 96,
    "Cincinnati": 96,
    "Washington": 48,
}


def match_identity_key(row: object) -> tuple[str, str, frozenset[str], str]:
    """Provider-independent identity used to prevent cross-source duplicates."""
    played_at = pd.to_datetime(getattr(row, "played_at_utc"), utc=True)
    tournament = clean_tournament_name(
        getattr(row, "tourney_name", getattr(row, "tournament_name", ""))
    )
    return (
        played_at.date().isoformat(),
        tournament,
        frozenset((str(getattr(row, "winner_name")), str(getattr(row, "loser_name")))),
        str(getattr(row, "round")),
    )


def match_identity_keys(frame: pd.DataFrame) -> set[tuple[str, str, frozenset[str], str]]:
    if frame.empty:
        return set()
    return {match_identity_key(row) for row in frame.itertuples(index=False)}


def clean_tournament_name(provider_name: str) -> str:
    name = str(provider_name).strip()
    if name.endswith(" ATP"):
        name = name[:-4]
    aliases = {
        "London": "Queens Club",
        "Hertogenbosch": "s-Hertogenbosch",
    }
    return aliases.get(name, name)


def tournament_level(tournament_name: str) -> str:
    if tournament_name in GRAND_SLAMS:
        return "G"
    if tournament_name in MASTERS_TOURNAMENTS:
        return "M"
    return "A"


def tournament_draw_size(tournament_name: str) -> int:
    return DRAW_SIZE_MAP.get(tournament_name, 32)


def _match_name(short_name: str, matcher: HistoricalPlayerMatcher) -> tuple[str, str, float]:
    matched, score, status = matcher.match_surname_initial(short_name)
    if matched is not None:
        return matched, status, score
    # A new player can legitimately be absent from the historical snapshot.
    return str(short_name).strip(), status, score


def load_completed_backfill(
    csv_path: str | Path,
    historical_names: pd.Series,
) -> pd.DataFrame:
    source = pd.read_csv(csv_path, low_memory=False)
    source["played_at_utc"] = pd.to_datetime(
        source["date_timestamp"], unit="s", utc=True, errors="coerce"
    )

    mask = (
        source["tour_type"].eq(1)
        & source["status"].eq("FINISHED")
        & source["status_extra"].isin(["FINISHED", "RETIRED"])
        & source["winner_code"].isin([1, 2])
        & source["surface"].str.casefold().isin(["hard", "clay", "grass"])
        & ~source["tournament"].str.contains("Qualification", case=False, na=False)
        & source["played_at_utc"].ge(BACKFILL_START_UTC)
        & ~source["tournament"].str.contains("French Open", case=False, na=False)
    )
    filtered = source.loc[mask].copy()
    matcher = HistoricalPlayerMatcher(historical_names)

    records: list[dict[str, object]] = []
    for row in filtered.itertuples(index=False):
        tournament = clean_tournament_name(row.tournament)
        home_name, home_status, home_score = _match_name(row.home_name, matcher)
        away_name, away_status, away_score = _match_name(row.away_name, matcher)

        if int(row.winner_code) == 1:
            winner_name, loser_name = home_name, away_name
            winner_prefix, loser_prefix = "home", "away"
        else:
            winner_name, loser_name = away_name, home_name
            winner_prefix, loser_prefix = "away", "home"

        winner_sets, loser_sets, winner_games, loser_games = _score_totals(
            row, winner_prefix, loser_prefix
        )
        records.append(
            {
                "provider_match_id": int(row.match_id),
                "played_at_utc": row.played_at_utc,
                "tourney_id": f"2026-{tournament}",
                "tourney_name": tournament,
                "tourney_level": tournament_level(tournament),
                "draw_size": tournament_draw_size(tournament),
                "surface": str(row.surface).title(),
                "round": ROUND_MAP.get(row.round),
                "best_of": 5 if tournament in GRAND_SLAMS else 3,
                "winner_name": winner_name,
                "loser_name": loser_name,
                "winner_sets": winner_sets,
                "loser_sets": loser_sets,
                "winner_games": winner_games,
                "loser_games": loser_games,
                "winner_rank": getattr(row, f"{winner_prefix}_rank"),
                "loser_rank": getattr(row, f"{loser_prefix}_rank"),
                "winner_rank_points": getattr(row, f"{winner_prefix}_points"),
                "loser_rank_points": getattr(row, f"{loser_prefix}_points"),
                "match_status": (
                    "retirement" if str(row.status_extra).upper() == "RETIRED" else "completed"
                ),
                "home_match_status": home_status,
                "away_match_status": away_status,
                "home_match_score": round(home_score, 4),
                "away_match_score": round(away_score, 4),
            }
        )

    result = pd.DataFrame(records)
    result = result[result["round"].notna()].copy()
    return result.sort_values(
        ["played_at_utc", "tourney_id", "round", "provider_match_id"]
    ).reset_index(drop=True)


def load_terminal_match_keys(
    csv_path: str | Path,
    historical_names: pd.Series,
) -> set[tuple[str, frozenset[str]]]:
    """Matches that are over but have no official result usable by the model."""
    source = pd.read_csv(csv_path, low_memory=False)
    source["played_at_utc"] = pd.to_datetime(
        source["date_timestamp"], unit="s", utc=True, errors="coerce"
    )
    mask = (
        source["tour_type"].eq(1)
        & source["status"].eq("FINISHED")
        & source["status_extra"].isin(["WALKOVER", "CANCELLED"])
        & ~source["tournament"].str.contains("Qualification", case=False, na=False)
        & source["played_at_utc"].ge(BACKFILL_START_UTC)
    )
    matcher = HistoricalPlayerMatcher(historical_names)
    keys: set[tuple[str, frozenset[str]]] = set()
    for row in source.loc[mask].itertuples(index=False):
        home_name, _, _ = _match_name(row.home_name, matcher)
        away_name, _, _ = _match_name(row.away_name, matcher)
        keys.add(
            (
                clean_tournament_name(row.tournament),
                frozenset((home_name, away_name)),
            )
        )
    return keys


def _score_totals(row: object, winner_prefix: str, loser_prefix: str) -> tuple[int, int, int, int]:
    winner_sets = int(getattr(row, f"{winner_prefix}_set_score"))
    loser_sets = int(getattr(row, f"{loser_prefix}_set_score"))
    winner_games = 0
    loser_games = 0
    for set_number in range(1, 6):
        winner_value = getattr(row, f"{winner_prefix}_set_{set_number}_score")
        loser_value = getattr(row, f"{loser_prefix}_set_{set_number}_score")
        if pd.notna(winner_value) and pd.notna(loser_value):
            winner_games += int(winner_value)
            loser_games += int(loser_value)
    return winner_sets, loser_sets, winner_games, loser_games
