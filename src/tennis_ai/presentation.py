"""Leakage-free player snapshots used by the match-detail presentation layer."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .player_matching import HistoricalPlayerMatcher
from .state_engine import CurrentStateEngine, SUPPORTED_SURFACES


STAT_COLUMNS = {
    "aces": "aces",
    "double_faults": "double_faults",
    "service_points_won_perc": "service_points_won_pct",
    "return_points_won_perc": "return_points_won_pct",
    "break_points_saved_perc": "break_points_saved_pct",
}

# The 2026 provider feed contains this shortened duplicate of Ben Shelton.
# Keep it available for historical matching, but never rank it as a second player.
LEADERBOARD_EXCLUDED_ALIASES = {"Shelton B."}


class PlayerPresentationService:
    def __init__(self, project_root: Path, state: CurrentStateEngine) -> None:
        self.root = project_root
        self.state = state
        self._matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._elo_ranks: dict[str, int] = {}
        self._surface_ranks: dict[str, dict[str, int]] = {}
        self._active_names: set[str] = set()
        self._load()

    def _rebuild_rankings(self) -> None:
        ranked_names = self._active_names - LEADERBOARD_EXCLUDED_ALIASES
        self._elo_ranks = self._rank(
            {name: self.state.players[name].elo for name in ranked_names}
        )
        self._surface_ranks = {
            surface: self._rank(
                {
                    name: self.state.players[name].surface_elo[surface]
                    for name in ranked_names
                }
            )
            for surface in SUPPORTED_SURFACES
        }

    def update_state(self, state: CurrentStateEngine) -> None:
        """Swap ratings without re-reading decades of immutable match history."""
        self.state = state
        backfill_path = self.root / "data" / "processed" / "atp_backfill_2026_current.pkl"
        if backfill_path.exists():
            backfill = pd.read_pickle(backfill_path)
            current_names = set(backfill["winner_name"].dropna().astype(str))
            current_names |= set(backfill["loser_name"].dropna().astype(str))
            self._active_names |= current_names
        self._active_names = {
            name for name in self._active_names if name in self.state.players
        }
        self._rebuild_rankings()

    @staticmethod
    def _percentage(numerator: Any, denominator: Any) -> float | None:
        numerator = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
        denominator = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
        if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= 0:
            return None
        return float(numerator) / float(denominator) * 100.0

    def _load_archive_history(self, active_names: set[str]) -> None:
        archive = (
            self.root
            / "tennis-sackmann-archive-main-aneeshers"
            / "tennis-sackmann-archive-main"
            / "atp"
        )
        columns = [
            "tourney_date", "match_num", "surface", "score",
            "winner_name", "loser_name", "w_ace", "l_ace", "w_df", "l_df",
            "w_svpt", "l_svpt", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",
            "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
        ]
        for path in sorted(archive.glob("atp_matches_[0-9][0-9][0-9][0-9].csv")):
            year = int(path.stem.rsplit("_", 1)[-1])
            if year < 1990 or year > 2025:
                continue
            source = pd.read_csv(path, usecols=columns, low_memory=False)
            source = source[
                source["surface"].isin(SUPPORTED_SURFACES)
                & (source["winner_name"].isin(active_names) | source["loser_name"].isin(active_names))
                & source["score"].notna()
                & ~source["score"].str.contains(r"RET|W/O|DEF", case=False, na=False)
            ].copy()
            source["played_at_utc"] = pd.to_datetime(
                source["tourney_date"].astype("Int64").astype(str),
                format="%Y%m%d",
                utc=True,
                errors="coerce",
            )
            for row in source.itertuples(index=False):
                sides = (
                    (
                        str(row.winner_name), str(row.loser_name), True,
                        row.w_ace, row.w_df, row.w_svpt, row.w_1stWon, row.w_2ndWon,
                        row.w_bpSaved, row.w_bpFaced,
                        row.l_svpt, row.l_1stWon, row.l_2ndWon,
                    ),
                    (
                        str(row.loser_name), str(row.winner_name), False,
                        row.l_ace, row.l_df, row.l_svpt, row.l_1stWon, row.l_2ndWon,
                        row.l_bpSaved, row.l_bpFaced,
                        row.w_svpt, row.w_1stWon, row.w_2ndWon,
                    ),
                )
                for (
                    name, opponent, won, aces, double_faults, service_points,
                    first_won, second_won, bp_saved, bp_faced,
                    opponent_service_points, opponent_first_won, opponent_second_won,
                ) in sides:
                    if name not in active_names:
                        continue
                    service_won = self._percentage(
                        pd.to_numeric(first_won, errors="coerce")
                        + pd.to_numeric(second_won, errors="coerce"),
                        service_points,
                    )
                    opponent_won = self._percentage(
                        pd.to_numeric(opponent_first_won, errors="coerce")
                        + pd.to_numeric(opponent_second_won, errors="coerce"),
                        opponent_service_points,
                    )
                    self._matches[name].append(
                        {
                            "played_at_utc": row.played_at_utc,
                            "sort_order": int(row.match_num) if pd.notna(row.match_num) else 0,
                            "surface": str(row.surface).title(),
                            "won": won,
                            "opponent": opponent,
                            "aces": aces,
                            "double_faults": double_faults,
                            "service_points_won_pct": service_won,
                            "return_points_won_pct": 100.0 - opponent_won if opponent_won is not None else None,
                            "break_points_saved_pct": self._percentage(bp_saved, bp_faced),
                        }
                    )

    def _load(self) -> None:
        active_index_path = (
            self.root / "data" / "processed" / "active_players_2026.json"
        )
        if active_index_path.exists():
            active_names = set(
                json.loads(active_index_path.read_text(encoding="utf-8"))
            )
        else:
            model_data = pd.read_pickle(
                self.root / "data" / "processed" / "atp_model_data_1990_2026.pkl"
            )
            current = model_data[model_data["source_year"].eq(2026)]
            active_names = set(current["p1_name"].dropna().astype(str)) | set(
                current["p2_name"].dropna().astype(str)
            )

        backfill_path = self.root / "data" / "processed" / "atp_backfill_2026_current.pkl"
        if backfill_path.exists():
            backfill = pd.read_pickle(backfill_path)
            active_names |= set(backfill["winner_name"].dropna().astype(str))
            active_names |= set(backfill["loser_name"].dropna().astype(str))

        csv_path = self.root / "data" / "external" / "2026-atp-season.csv"
        if csv_path.exists():
            provider_players = pd.read_csv(
                csv_path,
                usecols=["tour_type", "home_name", "away_name"],
                low_memory=False,
            )
            provider_players = provider_players[provider_players["tour_type"].eq(1)]
            short_names = set(provider_players["home_name"].dropna().astype(str))
            short_names |= set(provider_players["away_name"].dropna().astype(str))
            state_matcher = HistoricalPlayerMatcher(set(self.state.players))
            active_names |= {
                matched
                for short_name in short_names
                if (matched := state_matcher.match_surname_initial(short_name)[0])
            }

        active_names = {name for name in active_names if name in self.state.players}
        self._active_names = active_names
        self._rebuild_rankings()

        self._load_archive_history(active_names)

        if csv_path.exists():
            source = pd.read_csv(csv_path, low_memory=False)
            source["played_at_utc"] = pd.to_datetime(
                source["date_timestamp"], unit="s", utc=True, errors="coerce"
            )
            source = source[
                source["tour_type"].eq(1)
                & source["status"].eq("FINISHED")
                & source["status_extra"].eq("FINISHED")
                & source["winner_code"].isin([1, 2])
                & ~source["tournament"].str.contains("Qualification", case=False, na=False)
            ].copy()
            matcher = HistoricalPlayerMatcher(active_names)
            unique_names = set(source["home_name"].dropna().astype(str)) | set(
                source["away_name"].dropna().astype(str)
            )
            name_map = {
                short: matcher.match_surname_initial(short)[0] for short in unique_names
            }

            for row in source.sort_values("played_at_utc").itertuples(index=False):
                for side, opponent_side, won in (
                    ("home", "away", int(row.winner_code) == 1),
                    ("away", "home", int(row.winner_code) == 2),
                ):
                    short_name = str(getattr(row, f"{side}_name"))
                    name = name_map.get(short_name)
                    if not name:
                        continue
                    record: dict[str, Any] = {
                        "played_at_utc": row.played_at_utc,
                        "sort_order": 0,
                        "surface": str(row.surface).title(),
                        "won": bool(won),
                        "opponent": name_map.get(str(getattr(row, f"{opponent_side}_name"))),
                    }
                    for source_name, target_name in STAT_COLUMNS.items():
                        record[target_name] = getattr(row, f"{side}_{source_name}")
                    self._matches[name].append(record)

        for history in self._matches.values():
            history.sort(
                key=lambda row: (row["played_at_utc"], row.get("sort_order", 0))
            )

    @staticmethod
    def _rank(ratings: dict[str, float]) -> dict[str, int]:
        ordered = sorted(ratings, key=ratings.get, reverse=True)
        return {name: index + 1 for index, name in enumerate(ordered)}

    def elo_leaderboard(self) -> list[str]:
        """Return active players ordered by the current leakage-free Elo state."""
        return sorted(self._elo_ranks, key=self._elo_ranks.get)

    def elo_rank(self, name: str) -> int | None:
        """Return a player's current UMTennis Elo rank."""
        return self._elo_ranks.get(name)

    def player_snapshot(
        self,
        name: str,
        surface: str,
        match_time: Any,
        current_rank: Any,
    ) -> dict[str, Any]:
        player = self.state.players[name]
        cutoff = pd.Timestamp(match_time)
        history = [row for row in self._matches.get(name, []) if row["played_at_utc"] < cutoff]
        return {
            "atp_rank": int(current_rank) if pd.notna(current_rank) else None,
            "elo": round(player.elo),
            "elo_rank": self._elo_ranks.get(name),
            "surface_elo": round(player.surface_elo[surface]),
            "surface_elo_rank": self._surface_ranks.get(surface, {}).get(name),
            "last_5": self._summary(history[-5:]),
            "last_10": self._summary(history[-10:]),
            "surface_last_10": self._summary(
                [row for row in history if row["surface"] == surface][-10:]
            ),
            # Career-to-date only: the fixture cutoff keeps this leakage-free.
            "career": self._summary(history),
        }

    @staticmethod
    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        matches = len(rows)
        wins = sum(int(row["won"]) for row in rows)
        result: dict[str, Any] = {
            "matches": matches,
            "wins": wins,
            "losses": matches - wins,
            "form": "".join("W" if row["won"] else "L" for row in rows),
            "win_rate": round(wins / matches, 3) if matches else None,
        }
        for metric in STAT_COLUMNS.values():
            values = pd.to_numeric(
                pd.Series([row.get(metric) for row in rows]), errors="coerce"
            ).dropna()
            result[metric] = round(float(values.mean()), 1) if len(values) else None
        return result
