"""FastAPI application serving upcoming ATP matches and model predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from threading import RLock
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

from .fixture_pipeline import (
    load_fixture_snapshot,
    verified_fixture_round,
)
from .inference import EnsemblePredictor
from .mock_fixtures import build_mock_fixture_table
from .presentation import PlayerPresentationService
from .prediction_history import PredictionHistoryStore
from .results_backfill import clean_tournament_name, player_identity_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CACHE_SECONDS = int(os.getenv("FIXTURE_CACHE_SECONDS", "86400"))
USE_MOCK_FIXTURES = os.getenv("USE_MOCK_FIXTURES", "0").strip().lower() not in {
    "0", "false", "no"
}
MOCK_ELO_OFFSET = int(os.getenv("MOCK_ELO_OFFSET", "0"))
MOCK_ELO_LIMIT = int(os.getenv("MOCK_ELO_LIMIT", "20"))


def _iso_utc(value: Any) -> str | None:
    if pd.isna(value):
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


class PredictionService:
    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root
        self._state_path = project_root / "models" / "current_state.joblib"
        self.state = joblib.load(self._state_path)
        self.predictor = EnsemblePredictor(project_root)
        self.presentation = PlayerPresentationService(project_root, self.state)
        self.prediction_history = PredictionHistoryStore(
            project_root / "data" / "cache" / "prediction_history.json"
        )
        self._fixtures = pd.DataFrame()
        self._fixtures_loaded_at = 0.0
        self._prediction_cache: dict[int, dict[str, Any]] = {}
        self._lock = RLock()
        self._state_signature = self._state_path.stat().st_mtime_ns
        self._model_signature = self._current_model_signature()

    def _current_model_signature(self) -> tuple[int, int]:
        model_dir = self.project_root / "models"
        return tuple(
            path.stat().st_mtime_ns
            for path in (
                model_dir / "xgboost_pipeline.joblib",
                model_dir / "logistic_pipeline.joblib",
            )
        )

    def ensure_current_artifacts(self) -> None:
        """Hot-reload atomically replaced state/models before serving a request."""
        state_signature = self._state_path.stat().st_mtime_ns
        model_signature = self._current_model_signature()
        if state_signature == self._state_signature and model_signature == self._model_signature:
            return
        with self._lock:
            state_signature = self._state_path.stat().st_mtime_ns
            model_signature = self._current_model_signature()
            if state_signature == self._state_signature and model_signature == self._model_signature:
                return
            if state_signature != self._state_signature:
                state = joblib.load(self._state_path)
                self.state = state
                self.presentation.update_state(state)
            if model_signature != self._model_signature:
                self.predictor = EnsemblePredictor(self.project_root)
            self._fixtures = pd.DataFrame()
            self._fixtures_loaded_at = 0.0
            self._prediction_cache.clear()
            self._state_signature = state_signature
            self._model_signature = model_signature

    def fixtures(self, force_refresh: bool = False) -> pd.DataFrame:
        self.ensure_current_artifacts()
        with self._lock:
            cache_fresh = (
                self._fixtures_loaded_at > 0
                and time.monotonic() - self._fixtures_loaded_at < FIXTURE_CACHE_SECONDS
            )
            if cache_fresh and not force_refresh:
                return self._fixtures.copy()

            if USE_MOCK_FIXTURES:
                fixtures = build_mock_fixture_table(
                    self.project_root,
                    self.state,
                    self.presentation.elo_leaderboard(),
                    offset=MOCK_ELO_OFFSET,
                    limit=MOCK_ELO_LIMIT,
                )
            else:
                # Live requests are made only by the explicit update pipeline.
                # A page visit must never consume the shared provider quota.
                fixtures = load_fixture_snapshot(self.project_root)
            if not USE_MOCK_FIXTURES and not fixtures.empty:
                fixtures = fixtures[
                    fixtures["identities_resolved"]
                    & ~fixtures.apply(
                        lambda row: self.state.is_completed_fixture(
                            row.tournament_name,
                            row.p1_historical_name,
                            row.p2_historical_name,
                        ),
                        axis=1,
                    )
                ].reset_index(drop=True)
            self._fixtures = fixtures
            self._fixtures_loaded_at = time.monotonic()
            active_ids = set(fixtures["match_id"].astype(int)) if not fixtures.empty else set()
            self._prediction_cache = {
                match_id: prediction
                for match_id, prediction in self._prediction_cache.items()
                if match_id in active_ids
            }
            return fixtures.copy()

    def match_list(self) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for _, row in self.fixtures().iterrows():
            payload = self._match_payload(row)
            prediction = self.predictor.predict_frame(self.state.build_feature_row(row))
            payload.update(
                {
                    "p1_win_probability": prediction["p1_win_probability"],
                    "p2_win_probability": prediction["p2_win_probability"],
                    "predicted_winner": prediction["predicted_winner"],
                    "confidence": prediction["confidence"],
                    "confidence_label": prediction["confidence_label"],
                }
            )
            matches.append(payload)
        return matches

    def previous_match_list(self, limit: int | None = None) -> list[dict[str, Any]]:
        records = self.prediction_history.completed(limit)
        if not records:
            return records
        backfill_path = self.project_root / "data" / "processed" / "atp_backfill_2026_current.pkl"
        backfill = pd.read_pickle(backfill_path) if backfill_path.exists() else pd.DataFrame()
        for record in records:
            original_round = record.get("round")
            record["round_at_prediction"] = original_round
            record["round_verified_after_match"] = False
            if pd.notna(original_round) and str(original_round).strip().upper() not in {"", "NAN", "NONE", "NULL", "TBD"}:
                continue
            verified = verified_fixture_round(
                record["match_id"], record["tournament_name"],
                record["p1_name"], record["p2_name"],
            )
            if verified is None and not backfill.empty:
                players = {player_identity_key(record["p1_name"]), player_identity_key(record["p2_name"])}
                expected_time = pd.to_datetime(record.get("start_time_utc"), utc=True, errors="coerce")
                matches = backfill[
                    backfill["tourney_name"].map(clean_tournament_name).eq(
                        clean_tournament_name(record["tournament_name"])
                    )
                ]
                matches = matches[
                    matches.apply(
                        lambda row: {player_identity_key(row["winner_name"]), player_identity_key(row["loser_name"])} == players,
                        axis=1,
                    )
                ]
                if pd.notna(expected_time):
                    matches = matches[
                        (pd.to_datetime(matches["played_at_utc"], utc=True) - expected_time).abs()
                        <= pd.Timedelta(days=2)
                    ]
                valid_rounds = matches["round"].dropna().astype(str)
                if len(valid_rounds) == 1 and valid_rounds.iloc[0].upper() not in {"", "NAN", "NONE", "NULL", "TBD"}:
                    verified = valid_rounds.iloc[0]
            if verified is not None:
                record["round"] = verified
                record["round_verified_after_match"] = True
        return records

    def predict(self, match_id: int) -> dict[str, Any]:
        self.ensure_current_artifacts()
        if match_id in self._prediction_cache:
            return self._prediction_cache[match_id]

        fixtures = self.fixtures()
        selected = fixtures[fixtures["match_id"].astype(int).eq(match_id)]
        if selected.empty:
            raise KeyError(match_id)
        fixture = selected.iloc[0]
        feature_frame = self.state.build_feature_row(fixture)
        prediction = self.predictor.predict_frame(feature_frame)
        surface = str(fixture.surface).title()
        h2h = self.state.head_to_head_snapshot(
            str(fixture.p1_historical_name),
            str(fixture.p2_historical_name),
            surface,
        )
        payload = {
            **self._match_payload(fixture),
            "p1_win_probability": prediction["p1_win_probability"],
            "p2_win_probability": prediction["p2_win_probability"],
            "predicted_winner": prediction["predicted_winner"],
            "confidence": prediction["confidence"],
            "confidence_label": prediction["confidence_label"],
            "h2h": h2h,
            "state_as_of_utc": _iso_utc(self.state.state_as_of),
            "p1_profile": self.presentation.player_snapshot(
                str(fixture.p1_historical_name),
                surface,
                fixture.start_time_utc,
                fixture.p1_current_rank,
            ),
            "p2_profile": self.presentation.player_snapshot(
                str(fixture.p2_historical_name),
                surface,
                fixture.start_time_utc,
                fixture.p2_current_rank,
            ),
        }
        self._prediction_cache[match_id] = payload
        return payload

    @staticmethod
    def _match_strength(
        p1_atp_rank: int | None,
        p2_atp_rank: int | None,
        p1_elo_rank: int | None,
        p2_elo_rank: int | None,
    ) -> float:
        def composite(atp_rank: int | None, elo_rank: int | None) -> float:
            if atp_rank is None:
                return float(elo_rank or 300)
            if elo_rank is None:
                return float(atp_rank)
            return 0.4 * atp_rank + 0.6 * elo_rank

        average_rank = (
            composite(p1_atp_rank, p1_elo_rank)
            + composite(p2_atp_rank, p2_elo_rank)
        ) / 2
        strength_bands = (
            (10, 5.0),
            (20, 4.5),
            (35, 4.0),
            (50, 3.5),
            (70, 3.0),
            (85, 2.5),
            (100, 2.0),
            (125, 1.5),
            (150, 1.0),
            (175, 0.5),
            (200, 0.0),
        )
        for ceiling, strength in strength_bands:
            if average_rank < ceiling:
                return strength
        return 0.0

    def _match_payload(self, row: pd.Series) -> dict[str, Any]:
        p1_rank = int(row.p1_current_rank) if pd.notna(row.p1_current_rank) else None
        p2_rank = int(row.p2_current_rank) if pd.notna(row.p2_current_rank) else None
        p1_elo_rank = self.presentation.elo_rank(str(row.p1_historical_name))
        p2_elo_rank = self.presentation.elo_rank(str(row.p2_historical_name))
        return {
            "match_id": int(row.match_id),
            "start_time_utc": _iso_utc(row.start_time_utc),
            "tournament_name": str(row.tournament_name),
            "surface": str(row.surface).title(),
            "round": str(row["round"]) if pd.notna(row["round"]) and str(row["round"]).strip().lower() not in {"", "nan", "none"} else "TBD",
            "p1_name": str(row.p1_display_name),
            "p2_name": str(row.p2_display_name),
            "p1_id": int(row.p1_id) if pd.notna(row.p1_id) else None,
            "p2_id": int(row.p2_id) if pd.notna(row.p2_id) else None,
            "p1_rank": p1_rank,
            "p2_rank": p2_rank,
            "p1_elo_rank": p1_elo_rank,
            "p2_elo_rank": p2_elo_rank,
            "match_strength": self._match_strength(
                p1_rank, p2_rank, p1_elo_rank, p2_elo_rank
            ),
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService()
    yield


app = FastAPI(
    title="TennisAI API",
    version="0.1.0",
    description="Leakage-free ATP pre-match win probabilities.",
    lifespan=lifespan,
)

origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    service = _service(request)
    service.ensure_current_artifacts()
    return {
        "status": "ok",
        "model": "xgboost_logistic_ensemble",
        "state_as_of_utc": _iso_utc(service.state.state_as_of),
        "presentation_cache_ready": service.presentation.cache_path.exists(),
        "fixture_cache_seconds": FIXTURE_CACHE_SECONDS,
    }


@app.get("/api/matches")
def upcoming_matches(request: Request) -> dict[str, Any]:
    matches = _service(request).match_list()
    return {"count": len(matches), "matches": matches}


@app.get("/api/previous-matches")
def previous_matches(request: Request, limit: int | None = None) -> dict[str, Any]:
    matches = _service(request).previous_match_list(None if limit is None else min(max(limit, 1), 250))
    return {"count": len(matches), "matches": matches}


@app.get("/api/matches/{match_id}/prediction")
def match_prediction(match_id: int, request: Request) -> dict[str, Any]:
    try:
        return _service(request).predict(match_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Match bulunamadı, bitmiş olabilir veya fixture cache yenilenmiş olabilir.",
        ) from exc
    except TennisAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
