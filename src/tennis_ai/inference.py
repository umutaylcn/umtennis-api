"""Production ensemble inference for model-ready match frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .state_engine import CurrentStateEngine


def confidence_label(confidence: float) -> str:
    if confidence < 0.60:
        return "Toss-up"
    if confidence < 0.70:
        return "Lean"
    if confidence < 0.80:
        return "Strong pick"
    return "Heavy favorite"


class EnsemblePredictor:
    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root)
        self.xgboost = joblib.load(root / "models" / "xgboost_pipeline.joblib")
        self.logistic = joblib.load(root / "models" / "logistic_pipeline.joblib")
        self.config = json.loads(
            (root / "models" / "ensemble_config.json").read_text(encoding="utf-8")
        )
        registry = json.loads(
            (root / "data" / "processed" / "feature_registry.json").read_text(
                encoding="utf-8"
            )
        )
        self.feature_columns: list[str] = registry["model_feature_columns"]

    def predict_frame(self, match_frame: pd.DataFrame) -> dict[str, Any]:
        if len(match_frame) != 1:
            raise ValueError("match_frame tam olarak bir maç içermeli")
        missing = set(self.feature_columns) - set(match_frame.columns)
        if missing:
            raise ValueError(f"Eksik feature'lar: {sorted(missing)}")

        features = match_frame[self.feature_columns]
        xgb_probability = float(self.xgboost.predict_proba(features)[0, 1])
        logistic_probability = float(self.logistic.predict_proba(features)[0, 1])
        p1_probability = (
            self.config["xgboost_weight"] * xgb_probability
            + self.config["logistic_weight"] * logistic_probability
        )
        p2_probability = 1.0 - p1_probability
        row = match_frame.iloc[0]
        p1_wins = p1_probability >= p2_probability
        confidence = max(p1_probability, p2_probability)
        return {
            "p1_name": row["p1_name"],
            "p2_name": row["p2_name"],
            "p1_win_probability": round(p1_probability, 4),
            "p2_win_probability": round(p2_probability, 4),
            "predicted_winner": row["p1_name"] if p1_wins else row["p2_name"],
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label(confidence),
        }


def predict_upcoming_fixtures(
    fixtures: pd.DataFrame,
    state: CurrentStateEngine,
    predictor: EnsemblePredictor,
) -> pd.DataFrame:
    predictions: list[dict[str, Any]] = []
    for _, fixture in fixtures.iterrows():
        if not bool(fixture.identities_resolved):
            continue
        if state.is_completed_fixture(
            fixture.tournament_name,
            fixture.p1_historical_name,
            fixture.p2_historical_name,
        ):
            continue

        feature_frame = state.build_feature_row(fixture)
        prediction = predictor.predict_frame(feature_frame)
        predictions.append(
            {
                "match_id": int(fixture.match_id),
                "start_time_utc": fixture.start_time_utc,
                "tournament_name": fixture.tournament_name,
                "surface": str(fixture.surface).title(),
                "round": fixture["round"],
                "p1_name": fixture.p1_display_name,
                "p2_name": fixture.p2_display_name,
                **{
                    key: value
                    for key, value in prediction.items()
                    if key not in {"p1_name", "p2_name"}
                },
                "state_as_of_utc": state.state_as_of,
            }
        )
    return pd.DataFrame(predictions)

