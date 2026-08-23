"""Leakage-free Elo feature regeneration for parameter tuning and production."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from .state_engine import EloConfig, PlayerState, SUPPORTED_SURFACES, elo_changes


ROLLING_WINDOWS = (5, 10)


def _rolling(history: deque[float], window: int) -> dict[str, float]:
    changes = list(history)[-window:]
    gained = sum(value for value in changes if value > 0)
    lost = sum(-value for value in changes if value < 0)
    return {"gained": gained, "lost": lost, "net": gained - lost}


def _regress(player: PlayerState, season: int, config: EloConfig) -> None:
    if player.last_season is None:
        player.last_season = season
        return
    elapsed = max(0, season - player.last_season)
    if elapsed and config.annual_regression > 0:
        retention = (1.0 - config.annual_regression) ** elapsed
        player.elo = config.initial_elo + (player.elo - config.initial_elo) * retention
        for surface in SUPPORTED_SURFACES:
            player.surface_elo[surface] = config.initial_elo + (
                player.surface_elo[surface] - config.initial_elo
            ) * retention
    player.last_season = max(player.last_season, season)


def regenerate_elo_features(
    model_data: pd.DataFrame,
    config: EloConfig,
) -> pd.DataFrame:
    """Return a copy with all Elo-derived pre-match features rebuilt chronologically.

    The first observed rating for each player/surface is used as the pre-1990 seed,
    preserving the historical work that predates the modeling table.
    """
    if not model_data["tourney_date"].is_monotonic_increasing:
        raise ValueError("model_data must be sorted chronologically")

    states: dict[str, PlayerState] = defaultdict(PlayerState)
    initialized_players: set[str] = set()
    initialized_surfaces: set[tuple[str, str]] = set()
    tournament_elo: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"gained": 0.0, "lost": 0.0}
    )
    output: dict[str, list[float | int]] = defaultdict(list)

    for row in model_data.itertuples(index=False):
        p1_name, p2_name = str(row.p1_name), str(row.p2_name)
        surface = str(row.surface)
        season = int(row.source_year)
        p1, p2 = states[p1_name], states[p2_name]

        for prefix, name, player in (
            ("p1", p1_name, p1),
            ("p2", p2_name, p2),
        ):
            if name not in initialized_players:
                player.elo = float(getattr(row, f"{prefix}_elo"))
                player.matches = int(getattr(row, f"{prefix}_matches_played"))
                player.last_season = season
                initialized_players.add(name)
            surface_key = (name, surface)
            if surface_key not in initialized_surfaces:
                player.surface_elo[surface] = float(
                    getattr(row, f"{prefix}_surface_elo")
                )
                player.surface_matches[surface] = int(
                    getattr(row, f"{prefix}_surface_matches_played")
                )
                initialized_surfaces.add(surface_key)
            _regress(player, season, config)

        output["p1_elo"].append(p1.elo)
        output["p2_elo"].append(p2.elo)
        output["elo_difference"].append(p1.elo - p2.elo)
        output["elo_total"].append(p1.elo + p2.elo)
        output["p1_matches_played"].append(p1.matches)
        output["p2_matches_played"].append(p2.matches)
        output["experience_difference"].append(p1.matches - p2.matches)

        p1_surface_elo = p1.surface_elo[surface]
        p2_surface_elo = p2.surface_elo[surface]
        output["p1_surface_elo"].append(p1_surface_elo)
        output["p2_surface_elo"].append(p2_surface_elo)
        output["surface_elo_difference"].append(p1_surface_elo - p2_surface_elo)
        output["surface_elo_total"].append(p1_surface_elo + p2_surface_elo)
        output["p1_surface_matches_played"].append(p1.surface_matches[surface])
        output["p2_surface_matches_played"].append(p2.surface_matches[surface])
        output["surface_experience_difference"].append(
            p1.surface_matches[surface] - p2.surface_matches[surface]
        )

        for window in ROLLING_WINDOWS:
            for prefix, player in (("p1", p1), ("p2", p2)):
                for metric, value in _rolling(player.changes, window).items():
                    output[f"{prefix}_elo_{metric}_last_{window}"].append(value)
                for metric, value in _rolling(
                    player.surface_changes[surface], window
                ).items():
                    output[f"{prefix}_surface_elo_{metric}_last_{window}"].append(
                        value
                    )

        tourney_id = str(row.tourney_id)
        p1_tourney = tournament_elo[(tourney_id, p1_name)]
        p2_tourney = tournament_elo[(tourney_id, p2_name)]
        for prefix, history in (("p1", p1_tourney), ("p2", p2_tourney)):
            output[f"{prefix}_tourney_elo_gained_before"].append(history["gained"])
            output[f"{prefix}_tourney_elo_lost_before"].append(history["lost"])
            output[f"{prefix}_tourney_elo_net_before"].append(
                history["gained"] - history["lost"]
            )
        output["tourney_elo_gained_difference"].append(
            p1_tourney["gained"] - p2_tourney["gained"]
        )
        output["tourney_elo_lost_difference"].append(
            p1_tourney["lost"] - p2_tourney["lost"]
        )
        output["tourney_elo_net_difference"].append(
            (p1_tourney["gained"] - p1_tourney["lost"])
            - (p2_tourney["gained"] - p2_tourney["lost"])
        )

        p1_won = int(row.target) == 1
        p1_change, p2_change = elo_changes(
            p1.elo,
            p2.elo,
            p1.matches,
            p2.matches,
            p1_won,
            config,
        )
        p1.elo += p1_change
        p2.elo += p2_change
        p1.matches += 1
        p2.matches += 1
        p1.changes.append(p1_change)
        p2.changes.append(p2_change)

        p1_surface_change, p2_surface_change = elo_changes(
            p1.surface_elo[surface],
            p2.surface_elo[surface],
            p1.surface_matches[surface],
            p2.surface_matches[surface],
            p1_won,
            config,
        )
        p1.surface_elo[surface] += p1_surface_change
        p2.surface_elo[surface] += p2_surface_change
        p1.surface_matches[surface] += 1
        p2.surface_matches[surface] += 1
        p1.surface_changes[surface].append(p1_surface_change)
        p2.surface_changes[surface].append(p2_surface_change)

        if p1_won:
            p1_tourney["gained"] += p1_change
            p2_tourney["lost"] += -p2_change
        else:
            p2_tourney["gained"] += p2_change
            p1_tourney["lost"] += -p1_change

    result = model_data.copy()
    rebuilt = pd.DataFrame(output, index=result.index)
    result[rebuilt.columns] = rebuilt
    if result[list(rebuilt.columns)].isna().any().any():
        raise ValueError("Elo regeneration created missing values")
    return result


def elo_probability(data: pd.DataFrame, divisor: float) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-data["elo_difference"].to_numpy() / divisor))


def blend_elo_feature_sets(
    stable: pd.DataFrame,
    provisional: pd.DataFrame,
    transition_start: int,
    stable_from: int,
) -> pd.DataFrame:
    """Blend two leakage-free Elo tracks using only pre-match experience.

    ``transition_start == stable_from`` creates a hard switch. Otherwise the
    provisional track fades linearly into the stable track over that interval.
    """
    if transition_start < 0 or stable_from <= 0 or transition_start > stable_from:
        raise ValueError("Invalid Elo transition interval")
    if len(stable) != len(provisional) or not stable.index.equals(provisional.index):
        raise ValueError("Elo feature sets must have identical rows")

    result = stable.copy()
    weights: dict[str, np.ndarray] = {}
    for prefix in ("p1", "p2"):
        matches = stable[f"{prefix}_matches_played"].to_numpy(dtype=float)
        if transition_start == stable_from:
            weight = (matches < stable_from).astype(float)
        else:
            weight = np.clip(
                (stable_from - matches) / (stable_from - transition_start),
                0.0,
                1.0,
            )
        weights[prefix] = weight

        player_columns = [
            column
            for column in stable.columns
            if column.startswith(f"{prefix}_elo_")
            or column == f"{prefix}_elo"
            or column.startswith(f"{prefix}_surface_elo")
            or column.startswith(f"{prefix}_tourney_elo_")
        ]
        for column in player_columns:
            result[column] = (
                weight * provisional[column].to_numpy(dtype=float)
                + (1.0 - weight) * stable[column].to_numpy(dtype=float)
            )

    result["elo_difference"] = result["p1_elo"] - result["p2_elo"]
    result["elo_total"] = result["p1_elo"] + result["p2_elo"]
    result["surface_elo_difference"] = (
        result["p1_surface_elo"] - result["p2_surface_elo"]
    )
    result["surface_elo_total"] = result["p1_surface_elo"] + result["p2_surface_elo"]
    result["tourney_elo_gained_difference"] = (
        result["p1_tourney_elo_gained_before"]
        - result["p2_tourney_elo_gained_before"]
    )
    result["tourney_elo_lost_difference"] = (
        result["p1_tourney_elo_lost_before"]
        - result["p2_tourney_elo_lost_before"]
    )
    result["tourney_elo_net_difference"] = (
        result["p1_tourney_elo_net_before"] - result["p2_tourney_elo_net_before"]
    )
    return result


def load_core_match_history(project_root: Path) -> pd.DataFrame:
    """Load and order every core-tour match used to advance the Elo state."""
    archive = (
        project_root
        / "tennis-sackmann-archive-main-aneeshers"
        / "tennis-sackmann-archive-main"
        / "atp"
    )
    columns = [
        "tourney_id", "tourney_date", "match_num", "tourney_level", "round",
        "surface", "score", "winner_id", "loser_id", "winner_name", "loser_name",
    ]
    frames = []
    for year in range(1981, 2027):
        frame = pd.read_csv(archive / f"atp_matches_{year}.csv", usecols=columns)
        frame["source_year"] = year
        frames.append(frame)
    matches = pd.concat(frames, ignore_index=True)
    matches = matches[matches["tourney_level"].isin(["A", "M", "G", "F"])].copy()
    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype(str), format="%Y%m%d"
    )
    score = matches["score"].fillna("").str.upper()
    matches["completed"] = ~(
        score.eq("")
        | score.str.contains(
            r"W/O|WALKOVER|\bRET\b|\bDEF\b|ABD|ABN|UNFINISHED|IN PROGRESS",
            regex=True,
        )
    )
    round_order = {
        "ER": 0, "RR": 0, "R128": 1, "R64": 2, "R32": 3,
        "R16": 4, "QF": 5, "SF": 6, "BR": 7, "F": 8,
    }
    matches["round_order"] = matches["round"].map(round_order)
    matches["source_row_order"] = np.arange(len(matches))
    return matches.sort_values(
        ["tourney_date", "tourney_id", "round_order", "match_num", "source_row_order"],
        kind="stable",
    ).reset_index(drop=True)


def regenerate_from_match_history(
    model_data: pd.DataFrame,
    match_history: pd.DataFrame,
    config: EloConfig,
) -> pd.DataFrame:
    """Rebuild Elo features from the complete match stream and snapshot model rows."""
    lookup = {
        (str(row.tourney_id), int(row.match_num)): (
            int(index), str(row.p1_name), str(row.p2_name)
        )
        for index, row in model_data[[
            "tourney_id", "match_num", "p1_name", "p2_name"
        ]].iterrows()
    }
    if len(lookup) != len(model_data):
        raise ValueError("Model match keys are not unique")

    states: dict[str, PlayerState] = defaultdict(PlayerState)
    tournament_elo: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"gained": 0.0, "lost": 0.0}
    )
    feature_names = [
        "p1_elo", "p2_elo", "elo_difference", "elo_total",
        "p1_matches_played", "p2_matches_played", "experience_difference",
        "p1_surface_elo", "p2_surface_elo", "surface_elo_difference",
        "surface_elo_total", "p1_surface_matches_played",
        "p2_surface_matches_played", "surface_experience_difference",
    ]
    for window in ROLLING_WINDOWS:
        for prefix in ("p1", "p2"):
            for metric in ("gained", "lost", "net"):
                feature_names.append(f"{prefix}_elo_{metric}_last_{window}")
                feature_names.append(f"{prefix}_surface_elo_{metric}_last_{window}")
    feature_names += [
        "p1_tourney_elo_gained_before", "p1_tourney_elo_lost_before",
        "p1_tourney_elo_net_before", "p2_tourney_elo_gained_before",
        "p2_tourney_elo_lost_before", "p2_tourney_elo_net_before",
        "tourney_elo_gained_difference", "tourney_elo_lost_difference",
        "tourney_elo_net_difference",
    ]
    snapshots = {
        name: np.full(len(model_data), np.nan, dtype=np.float64)
        for name in feature_names
    }
    captured = np.zeros(len(model_data), dtype=bool)

    for row in match_history.itertuples(index=False):
        winner_id, loser_id = int(row.winner_id), int(row.loser_id)
        winner, loser = states[winner_id], states[loser_id]
        season = int(row.source_year)
        _regress(winner, season, config)
        _regress(loser, season, config)
        surface = str(row.surface)
        key = (str(row.tourney_id), int(row.match_num))
        model_match = lookup.get(key)

        if model_match is not None:
            index, p1_name, p2_name = model_match
            if p1_name == str(row.winner_name) and p2_name == str(row.loser_name):
                p1_id, p2_id = winner_id, loser_id
            elif p1_name == str(row.loser_name) and p2_name == str(row.winner_name):
                p1_id, p2_id = loser_id, winner_id
            else:
                raise ValueError(f"Player mismatch for model key {key}")
            p1, p2 = states[p1_id], states[p2_id]
            if {p1_id, p2_id} != {winner_id, loser_id}:
                raise ValueError(f"Player mismatch for model key {key}")
            values = {
                "p1_elo": p1.elo,
                "p2_elo": p2.elo,
                "elo_difference": p1.elo - p2.elo,
                "elo_total": p1.elo + p2.elo,
                "p1_matches_played": p1.matches,
                "p2_matches_played": p2.matches,
                "experience_difference": p1.matches - p2.matches,
                "p1_surface_elo": p1.surface_elo[surface],
                "p2_surface_elo": p2.surface_elo[surface],
                "surface_elo_difference": p1.surface_elo[surface] - p2.surface_elo[surface],
                "surface_elo_total": p1.surface_elo[surface] + p2.surface_elo[surface],
                "p1_surface_matches_played": p1.surface_matches[surface],
                "p2_surface_matches_played": p2.surface_matches[surface],
                "surface_experience_difference": (
                    p1.surface_matches[surface] - p2.surface_matches[surface]
                ),
            }
            for window in ROLLING_WINDOWS:
                for prefix, player in (("p1", p1), ("p2", p2)):
                    for metric, value in _rolling(player.changes, window).items():
                        values[f"{prefix}_elo_{metric}_last_{window}"] = value
                    for metric, value in _rolling(
                        player.surface_changes[surface], window
                    ).items():
                        values[f"{prefix}_surface_elo_{metric}_last_{window}"] = value
            p1_tourney = tournament_elo[(str(row.tourney_id), p1_id)]
            p2_tourney = tournament_elo[(str(row.tourney_id), p2_id)]
            for prefix, history in (("p1", p1_tourney), ("p2", p2_tourney)):
                values[f"{prefix}_tourney_elo_gained_before"] = history["gained"]
                values[f"{prefix}_tourney_elo_lost_before"] = history["lost"]
                values[f"{prefix}_tourney_elo_net_before"] = history["gained"] - history["lost"]
            values["tourney_elo_gained_difference"] = p1_tourney["gained"] - p2_tourney["gained"]
            values["tourney_elo_lost_difference"] = p1_tourney["lost"] - p2_tourney["lost"]
            values["tourney_elo_net_difference"] = (
                values["p1_tourney_elo_net_before"] - values["p2_tourney_elo_net_before"]
            )
            for name, value in values.items():
                snapshots[name][index] = value
            captured[index] = True

        if not bool(row.completed):
            continue
        winner_change, loser_change = elo_changes(
            winner.elo, loser.elo, winner.matches, loser.matches, True, config
        )
        winner.elo += winner_change
        loser.elo += loser_change
        winner.matches += 1
        loser.matches += 1
        winner.changes.append(winner_change)
        loser.changes.append(loser_change)

        if surface in SUPPORTED_SURFACES:
            winner_surface_change, loser_surface_change = elo_changes(
                winner.surface_elo[surface], loser.surface_elo[surface],
                winner.surface_matches[surface], loser.surface_matches[surface],
                True, config,
            )
            winner.surface_elo[surface] += winner_surface_change
            loser.surface_elo[surface] += loser_surface_change
            winner.surface_matches[surface] += 1
            loser.surface_matches[surface] += 1
            winner.surface_changes[surface].append(winner_surface_change)
            loser.surface_changes[surface].append(loser_surface_change)

        winner_tourney = tournament_elo[(str(row.tourney_id), winner_id)]
        loser_tourney = tournament_elo[(str(row.tourney_id), loser_id)]
        winner_tourney["gained"] += winner_change
        loser_tourney["lost"] += -loser_change

    if not captured.all():
        missing = int((~captured).sum())
        raise ValueError(f"{missing} model rows were not found in complete match history")
    result = model_data.copy()
    for name, values in snapshots.items():
        result[name] = values
    return result
