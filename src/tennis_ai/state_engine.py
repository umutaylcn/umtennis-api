"""Leakage-free current state rebuilt from model rows and completed backfill."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .results_backfill import (
    clean_tournament_name,
    tournament_draw_size,
    tournament_level,
)


INITIAL_ELO = 1500.0
ELO_DIVISOR = 400.0
PROVISIONAL_MATCH_LIMIT = 10
PROVISIONAL_K = 80.0
STANDARD_K = 40.0
SUPPORTED_SURFACES = ("Hard", "Clay", "Grass")

ROUND_ORDER = {
    "ER": 0,
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}


@dataclass(frozen=True)
class EloConfig:
    initial_elo: float = INITIAL_ELO
    divisor: float = ELO_DIVISOR
    provisional_match_limit: int = PROVISIONAL_MATCH_LIMIT
    provisional_k: float = PROVISIONAL_K
    standard_k: float = STANDARD_K
    annual_regression: float = 0.0


DEFAULT_ELO_CONFIG = EloConfig()


def expected_score(
    rating_a: float,
    rating_b: float,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / config.divisor))


def k_factor(
    matches_played: int,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> float:
    return (
        config.provisional_k
        if matches_played < config.provisional_match_limit
        else config.standard_k
    )


def elo_changes(
    p1_rating: float,
    p2_rating: float,
    p1_matches: int,
    p2_matches: int,
    p1_won: bool,
    config: EloConfig = DEFAULT_ELO_CONFIG,
) -> tuple[float, float]:
    p1_expected = expected_score(p1_rating, p2_rating, config)
    p2_expected = 1.0 - p1_expected
    p1_actual = 1.0 if p1_won else 0.0
    p2_actual = 1.0 - p1_actual
    return (
        k_factor(p1_matches, config) * (p1_actual - p1_expected),
        k_factor(p2_matches, config) * (p2_actual - p2_expected),
    )


@dataclass
class PlayerState:
    elo: float = INITIAL_ELO
    matches: int = 0
    surface_elo: dict[str, float] = field(
        default_factory=lambda: {surface: INITIAL_ELO for surface in SUPPORTED_SURFACES}
    )
    surface_matches: dict[str, int] = field(
        default_factory=lambda: {surface: 0 for surface in SUPPORTED_SURFACES}
    )
    changes: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    surface_changes: dict[str, deque[float]] = field(
        default_factory=lambda: {
            surface: deque(maxlen=10) for surface in SUPPORTED_SURFACES
        }
    )
    height: float = np.nan
    hand: str | None = None
    rank: float = np.nan
    rank_points: float = np.nan
    last_season: int | None = None


def _tournament_record() -> dict[str, float]:
    return {
        "matches": 0,
        "wins": 0,
        "losses": 0,
        "elo_gained": 0.0,
        "elo_lost": 0.0,
        "sets_won": 0,
        "sets_lost": 0,
        "games_won": 0,
        "games_lost": 0,
    }


class CurrentStateEngine:
    def __init__(self, elo_config: EloConfig | None = None) -> None:
        self.elo_config = elo_config or DEFAULT_ELO_CONFIG
        self.players: dict[str, PlayerState] = {}
        self.h2h: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.surface_h2h: dict[str, dict[tuple[str, str], Counter[str]]] = {
            surface: defaultdict(Counter) for surface in SUPPORTED_SURFACES
        }
        self.tournaments: dict[tuple[str, str], dict[str, float]] = defaultdict(
            _tournament_record
        )
        self.completed_keys: set[tuple[str, frozenset[str]]] = set()
        self.terminal_keys: set[tuple[str, frozenset[str]]] = set()
        self.state_as_of: pd.Timestamp | None = None

    def _player(self, name: str) -> PlayerState:
        return self.players.setdefault(str(name), PlayerState())

    def _apply_annual_regression(self, player: PlayerState, season: int) -> None:
        if player.last_season is None:
            player.last_season = season
            return
        elapsed_seasons = max(0, season - player.last_season)
        config = getattr(self, "elo_config", DEFAULT_ELO_CONFIG)
        if elapsed_seasons and config.annual_regression > 0:
            retention = (1.0 - config.annual_regression) ** elapsed_seasons
            player.elo = config.initial_elo + (
                player.elo - config.initial_elo
            ) * retention
            for surface in SUPPORTED_SURFACES:
                player.surface_elo[surface] = config.initial_elo + (
                    player.surface_elo[surface] - config.initial_elo
                ) * retention
        player.last_season = max(player.last_season, season)

    def bootstrap(self, model_data: pd.DataFrame) -> "CurrentStateEngine":
        for row in model_data.itertuples(index=False):
            p1_name, p2_name = str(row.p1_name), str(row.p2_name)
            p1, p2 = self._player(p1_name), self._player(p2_name)
            surface = str(row.surface)
            p1_won = int(row.target) == 1

            p1.elo, p2.elo = float(row.p1_elo), float(row.p2_elo)
            p1.matches, p2.matches = int(row.p1_matches_played), int(row.p2_matches_played)
            change1, change2 = elo_changes(
                p1.elo, p2.elo, p1.matches, p2.matches, p1_won, self.elo_config
            )
            p1.elo += change1
            p2.elo += change2
            p1.matches += 1
            p2.matches += 1
            p1.changes.append(change1)
            p2.changes.append(change2)

            p1.surface_elo[surface] = float(row.p1_surface_elo)
            p2.surface_elo[surface] = float(row.p2_surface_elo)
            p1.surface_matches[surface] = int(row.p1_surface_matches_played)
            p2.surface_matches[surface] = int(row.p2_surface_matches_played)
            surface_change1, surface_change2 = elo_changes(
                p1.surface_elo[surface],
                p2.surface_elo[surface],
                p1.surface_matches[surface],
                p2.surface_matches[surface],
                p1_won,
                self.elo_config,
            )
            p1.surface_elo[surface] += surface_change1
            p2.surface_elo[surface] += surface_change2
            p1.surface_matches[surface] += 1
            p2.surface_matches[surface] += 1
            p1.surface_changes[surface].append(surface_change1)
            p2.surface_changes[surface].append(surface_change2)

            self._set_pair_from_features(row, p1_name, p2_name, surface, p1_won)
            self._update_static_from_model_row(row, p1, p2)
            season = int(pd.Timestamp(row.tourney_date).year)
            p1.last_season = season
            p2.last_season = season

        return self

    def _set_pair_from_features(
        self, row: Any, p1_name: str, p2_name: str, surface: str, p1_won: bool
    ) -> None:
        pair = tuple(sorted((p1_name, p2_name)))
        self.h2h[pair] = Counter(
            {
                p1_name: int(row.p1_h2h_wins_before),
                p2_name: int(row.p2_h2h_wins_before),
            }
        )
        self.h2h[pair][p1_name if p1_won else p2_name] += 1
        self.surface_h2h[surface][pair] = Counter(
            {
                p1_name: int(row.p1_surface_h2h_wins_before),
                p2_name: int(row.p2_surface_h2h_wins_before),
            }
        )
        self.surface_h2h[surface][pair][p1_name if p1_won else p2_name] += 1

    @staticmethod
    def _update_static_from_model_row(row: Any, p1: PlayerState, p2: PlayerState) -> None:
        for state, prefix in ((p1, "p1"), (p2, "p2")):
            height = getattr(row, f"{prefix}_ht")
            if pd.notna(height):
                state.height = float(height)
            hand = getattr(row, f"{prefix}_hand", None)
            if pd.notna(hand):
                state.hand = str(hand)
            rank = getattr(row, f"{prefix}_rank")
            points = getattr(row, f"{prefix}_rank_points")
            if pd.notna(rank):
                state.rank = float(rank)
            if pd.notna(points):
                state.rank_points = float(points)

    def apply_backfill(self, backfill: pd.DataFrame) -> "CurrentStateEngine":
        for row in backfill.itertuples(index=False):
            self.apply_completed_match(row)
            self.state_as_of = pd.Timestamp(row.played_at_utc)
        return self

    def apply_completed_match(self, row: Any) -> None:
        winner_name, loser_name = str(row.winner_name), str(row.loser_name)
        winner, loser = self._player(winner_name), self._player(loser_name)
        played_at = getattr(row, "played_at_utc", getattr(row, "tourney_date", None))
        season = int(pd.Timestamp(played_at).year)
        self._apply_annual_regression(winner, season)
        self._apply_annual_regression(loser, season)
        surface = str(row.surface)
        config = getattr(self, "elo_config", DEFAULT_ELO_CONFIG)
        winner_change, loser_change = elo_changes(
            winner.elo, loser.elo, winner.matches, loser.matches, True, config
        )
        winner.elo += winner_change
        loser.elo += loser_change
        winner.matches += 1
        loser.matches += 1
        winner.changes.append(winner_change)
        loser.changes.append(loser_change)

        winner_surface_change, loser_surface_change = elo_changes(
            winner.surface_elo[surface],
            loser.surface_elo[surface],
            winner.surface_matches[surface],
            loser.surface_matches[surface],
            True,
            config,
        )
        winner.surface_elo[surface] += winner_surface_change
        loser.surface_elo[surface] += loser_surface_change
        winner.surface_matches[surface] += 1
        loser.surface_matches[surface] += 1
        winner.surface_changes[surface].append(winner_surface_change)
        loser.surface_changes[surface].append(loser_surface_change)

        pair = tuple(sorted((winner_name, loser_name)))
        self.h2h[pair][winner_name] += 1
        self.surface_h2h[surface][pair][winner_name] += 1

        tournament = clean_tournament_name(row.tourney_name)
        winner_tourney = self.tournaments[(tournament, winner_name)]
        loser_tourney = self.tournaments[(tournament, loser_name)]
        winner_tourney["matches"] += 1
        winner_tourney["wins"] += 1
        winner_tourney["elo_gained"] += winner_change
        loser_tourney["matches"] += 1
        loser_tourney["losses"] += 1
        loser_tourney["elo_lost"] += -loser_change

        winner_tourney["sets_won"] += int(row.winner_sets)
        winner_tourney["sets_lost"] += int(row.loser_sets)
        winner_tourney["games_won"] += int(row.winner_games)
        winner_tourney["games_lost"] += int(row.loser_games)
        loser_tourney["sets_won"] += int(row.loser_sets)
        loser_tourney["sets_lost"] += int(row.winner_sets)
        loser_tourney["games_won"] += int(row.loser_games)
        loser_tourney["games_lost"] += int(row.winner_games)

        if pd.notna(row.winner_rank):
            winner.rank = float(row.winner_rank)
        if pd.notna(row.loser_rank):
            loser.rank = float(row.loser_rank)
        if pd.notna(row.winner_rank_points):
            winner.rank_points = float(row.winner_rank_points)
        if pd.notna(row.loser_rank_points):
            loser.rank_points = float(row.loser_rank_points)

        self.completed_keys.add((tournament, frozenset((winner_name, loser_name))))

    @staticmethod
    def _rolling(history: deque[float], window: int) -> dict[str, float]:
        changes = list(history)[-window:]
        gained = sum(value for value in changes if value > 0)
        lost = sum(-value for value in changes if value < 0)
        return {"gained": gained, "lost": lost, "net": gained - lost}

    def is_completed_fixture(self, tournament: str, p1_name: str, p2_name: str) -> bool:
        key = (clean_tournament_name(tournament), frozenset((p1_name, p2_name)))
        return key in self.completed_keys or key in self.terminal_keys

    def head_to_head_snapshot(
        self, p1_name: str, p2_name: str, surface: str
    ) -> dict[str, int]:
        """Return leakage-free H2H scores from matches completed before a fixture."""
        pair = tuple(sorted((str(p1_name), str(p2_name))))
        general = self.h2h.get(pair, Counter())
        surface_record = self.surface_h2h[str(surface).title()].get(pair, Counter())
        p1_wins = int(general[str(p1_name)])
        p2_wins = int(general[str(p2_name)])
        p1_surface_wins = int(surface_record[str(p1_name)])
        p2_surface_wins = int(surface_record[str(p2_name)])
        return {
            "matches": p1_wins + p2_wins,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "surface_matches": p1_surface_wins + p2_surface_wins,
            "p1_surface_wins": p1_surface_wins,
            "p2_surface_wins": p2_surface_wins,
        }

    def build_feature_row(self, fixture: pd.Series) -> pd.DataFrame:
        p1_name = str(fixture.p1_historical_name)
        p2_name = str(fixture.p2_historical_name)
        p1, p2 = self._player(p1_name), self._player(p2_name)
        fixture_season = int(pd.Timestamp(fixture.start_time_utc).year)
        self._apply_annual_regression(p1, fixture_season)
        self._apply_annual_regression(p2, fixture_season)
        surface = str(fixture.surface).title()
        tournament = clean_tournament_name(fixture.tournament_name)
        round_code = str(fixture["round"])
        draw_size = tournament_draw_size(tournament)
        level = tournament_level(tournament)

        values: dict[str, Any] = {
            "p1_name": p1_name,
            "p2_name": p2_name,
            "draw_size": draw_size,
            "best_of": 5 if level == "G" else 3,
            "round_order": ROUND_ORDER.get(round_code),
            "p1_seed": draw_size + 1,
            "p2_seed": draw_size + 1,
            "p1_is_seeded": 0,
            "p2_is_seeded": 0,
            "seed_advantage": 0,
            "p1_age": self._age(fixture.p1_birthday, fixture.start_time_utc),
            "p2_age": self._age(fixture.p2_birthday, fixture.start_time_utc),
            "p1_ht": p1.height,
            "p2_ht": p2.height,
            "p1_rank": self._prefer(fixture.p1_current_rank, p1.rank),
            "p2_rank": self._prefer(fixture.p2_current_rank, p2.rank),
            "p1_rank_points": self._prefer(
                fixture.p1_current_rank_points, p1.rank_points
            ),
            "p2_rank_points": self._prefer(
                fixture.p2_current_rank_points, p2.rank_points
            ),
            "p1_elo": p1.elo,
            "p2_elo": p2.elo,
            "p1_matches_played": p1.matches,
            "p2_matches_played": p2.matches,
            "p1_surface_elo": p1.surface_elo[surface],
            "p2_surface_elo": p2.surface_elo[surface],
            "p1_surface_matches_played": p1.surface_matches[surface],
            "p2_surface_matches_played": p2.surface_matches[surface],
            "surface": surface,
            "tourney_level": level,
            "round": round_code,
            "hand_matchup": f"{fixture.p1_hand or p1.hand or 'U'}_{fixture.p2_hand or p2.hand or 'U'}",
            "p1_entry": "DA",
            "p2_entry": "DA",
        }
        values["age_difference"] = values["p1_age"] - values["p2_age"]
        values["height_difference"] = values["p1_ht"] - values["p2_ht"]
        values["rank_difference"] = values["p2_rank"] - values["p1_rank"]
        values["rank_points_difference"] = (
            values["p1_rank_points"] - values["p2_rank_points"]
        )
        values["elo_difference"] = p1.elo - p2.elo
        values["elo_total"] = p1.elo + p2.elo
        values["experience_difference"] = p1.matches - p2.matches
        values["surface_elo_difference"] = (
            p1.surface_elo[surface] - p2.surface_elo[surface]
        )
        values["surface_elo_total"] = (
            p1.surface_elo[surface] + p2.surface_elo[surface]
        )
        values["surface_experience_difference"] = (
            p1.surface_matches[surface] - p2.surface_matches[surface]
        )

        self._add_rolling(values, p1, p2, surface)
        self._add_h2h(values, p1_name, p2_name, surface)
        self._add_tournament(values, tournament, p1_name, p2_name)
        return pd.DataFrame([values])

    def _add_rolling(
        self, values: dict[str, Any], p1: PlayerState, p2: PlayerState, surface: str
    ) -> None:
        for window in (5, 10):
            for prefix, player in (("p1", p1), ("p2", p2)):
                for metric, value in self._rolling(player.changes, window).items():
                    values[f"{prefix}_elo_{metric}_last_{window}"] = value
                for metric, value in self._rolling(
                    player.surface_changes[surface], window
                ).items():
                    values[f"{prefix}_surface_elo_{metric}_last_{window}"] = value

    def _add_h2h(
        self, values: dict[str, Any], p1_name: str, p2_name: str, surface: str
    ) -> None:
        pair = tuple(sorted((p1_name, p2_name)))
        general = self.h2h[pair]
        surface_record = self.surface_h2h[surface][pair]
        p1_wins, p2_wins = general[p1_name], general[p2_name]
        p1_surface_wins = surface_record[p1_name]
        p2_surface_wins = surface_record[p2_name]
        total, surface_total = p1_wins + p2_wins, p1_surface_wins + p2_surface_wins
        values.update(
            {
                "h2h_matches_before": total,
                "p1_h2h_wins_before": p1_wins,
                "p2_h2h_wins_before": p2_wins,
                "p1_h2h_win_rate_before": p1_wins / total if total else 0.5,
                "h2h_win_difference": p1_wins - p2_wins,
                "surface_h2h_matches_before": surface_total,
                "p1_surface_h2h_wins_before": p1_surface_wins,
                "p2_surface_h2h_wins_before": p2_surface_wins,
                "p1_surface_h2h_win_rate_before": (
                    p1_surface_wins / surface_total if surface_total else 0.5
                ),
                "surface_h2h_win_difference": p1_surface_wins - p2_surface_wins,
            }
        )

    def _add_tournament(
        self, values: dict[str, Any], tournament: str, p1_name: str, p2_name: str
    ) -> None:
        metrics = (
            "matches", "wins", "losses", "elo_gained", "elo_lost",
            "sets_won", "sets_lost", "games_won", "games_lost",
        )
        records = {
            "p1": self.tournaments[(tournament, p1_name)],
            "p2": self.tournaments[(tournament, p2_name)],
        }
        for prefix, record in records.items():
            for metric in metrics:
                values[f"{prefix}_tourney_{metric}_before"] = record[metric]
            values[f"{prefix}_tourney_win_rate_before"] = (
                record["wins"] / record["matches"] if record["matches"] else 0.5
            )
            values[f"{prefix}_tourney_elo_net_before"] = (
                record["elo_gained"] - record["elo_lost"]
            )
            sets = record["sets_won"] + record["sets_lost"]
            games = record["games_won"] + record["games_lost"]
            values[f"{prefix}_tourney_sets_net_before"] = (
                record["sets_won"] - record["sets_lost"]
            )
            values[f"{prefix}_tourney_set_win_rate_before"] = (
                record["sets_won"] / sets if sets else 0.5
            )
            values[f"{prefix}_tourney_games_net_before"] = (
                record["games_won"] - record["games_lost"]
            )
            values[f"{prefix}_tourney_game_win_rate_before"] = (
                record["games_won"] / games if games else 0.5
            )

        comparison_metrics = (
            "matches", "wins", "losses", "win_rate", "elo_gained", "elo_lost",
            "elo_net", "sets_won", "sets_lost", "sets_net", "set_win_rate",
            "games_won", "games_lost", "games_net", "game_win_rate",
        )
        for metric in comparison_metrics:
            values[f"tourney_{metric}_difference"] = (
                values[f"p1_tourney_{metric}_before"]
                - values[f"p2_tourney_{metric}_before"]
            )

    @staticmethod
    def _prefer(primary: Any, fallback: float) -> float:
        return float(primary) if pd.notna(primary) else float(fallback)

    @staticmethod
    def _age(birthday: Any, match_time: Any) -> float:
        if pd.isna(birthday) or pd.isna(match_time):
            return np.nan
        born = pd.Timestamp(birthday)
        played = pd.Timestamp(match_time).tz_localize(None)
        return (played - born).days / 365.25
