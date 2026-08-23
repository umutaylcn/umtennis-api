"""Shared model construction and chronological evaluation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


CV_FOLDS = (
    (2002, 2003, 2006),
    (2006, 2007, 2010),
    (2010, 2011, 2014),
    (2014, 2015, 2018),
    (2018, 2019, 2022),
)


def load_feature_registry(project_root: Path) -> dict:
    path = project_root / "data" / "processed" / "feature_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_preprocessor(registry: dict) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, registry["numeric_features"]),
            ("categorical", categorical, registry["categorical_features"]),
        ]
    )


def build_logistic(registry: dict) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(registry)),
            (
                "model",
                LogisticRegression(C=0.01, max_iter=2_000, random_state=42),
            ),
        ]
    )


def build_xgboost(registry: dict) -> Pipeline:
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocessor", build_preprocessor(registry)), ("model", model)])


def probability_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "roc_auc": roc_auc_score(y_true, probability),
        "log_loss": log_loss(y_true, probability),
        "brier_score": brier_score_loss(y_true, probability),
        "calibration_error": expected_calibration_error(y_true, probability),
    }


def expected_calibration_error(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    frame = pd.DataFrame({"actual": np.asarray(y_true), "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).agg(
        matches=("actual", "size"),
        actual=("actual", "mean"),
        predicted=("probability", "mean"),
    )
    return float(
        ((grouped["actual"] - grouped["predicted"]).abs() * grouped["matches"]).sum()
        / len(frame)
    )

