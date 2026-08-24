# UMTennis Machine Learning API

Production FastAPI service and research pipeline for [UMTennis](https://prediction.umtennis.workers.dev), an ATP pre-match probability platform.

[Live App](https://prediction.umtennis.workers.dev) · [Frontend Repository](https://github.com/umutaylcn/UMTennis) · [Health Check](https://umtennis-api.onrender.com/api/health)

## Overview

This repository contains the complete ML side of the project:

- chronological data audit and match-status cleaning
- leakage-free Player 1 / Player 2 transformation
- stateful general Elo and surface Elo ratings
- rolling form, opponent-adjusted Elo movement, H2H, and tournament features
- expanding-window cross-validation
- Logistic Regression, Decision Tree, Random Forest, XGBoost, and neural-network experiments
- calibrated XGBoost–Logistic Regression probability ensemble
- live-fixture matching, state reconstruction, caching, and FastAPI inference

## Evaluation

The primary evaluation period was kept untouched until model selection was complete.

| Evaluation period | Accuracy | ROC-AUC | Log loss | Brier score |
| --- | ---: | ---: | ---: | ---: |
| 2023-2025 final test | 66.2% | 0.726 | 0.608 | 0.211 |
| 2026 partial backtest | 66.0% | 0.733 | 0.602 | 0.208 |

Production ensemble: **0.60 XGBoost + 0.40 Logistic Regression** across **119 pre-match features**.

## Leakage prevention

Tennis data is naturally stored as winner/loser rows. Using that orientation directly would reveal the target. The pipeline therefore:

1. sorts every match chronologically;
2. snapshots all stateful features before applying the current result;
3. randomly maps winners and losers into Player 1 and Player 2 with an approximately balanced target;
4. performs only time-ordered validation;
5. reserves 2023-2025 as the final untouched test period.

No rolling feature, Elo value, H2H statistic, or tournament statistic uses information from the match being predicted or a later match.

## Feature groups

- general Elo and surface-specific Elo
- ATP rank and ranking points
- last 5 / last 10 form
- Elo gained and lost over rolling windows
- surface experience and surface form
- all-time and surface H2H
- tournament-level set and game performance
- age, height, handedness, draw, round, surface, and tournament level
- missing-value indicators for historically unavailable fields

## Research notebooks

1. `01_data_audit.ipynb` - schema, date coverage, match-status handling, missingness
2. `02_feature_engineering.ipynb` - leakage-free state reconstruction and feature generation
3. `03_modeling.ipynb` - temporal CV, model comparison, calibration, final test
4. `04_inference_pipeline.ipynb` - end-to-end match prediction example

## API

Main endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | service and cache readiness |
| `GET /api/matches` | cached upcoming ATP fixtures |
| `GET /api/matches/{match_id}` | model-ready match details |
| `GET /api/matches/{match_id}/prediction` | ensemble probabilities and pre-match statistics |

Upcoming fixtures and presentation state are cached so ordinary page visits do not repeatedly call the live provider.

## Local setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
copy .env.example .env
uvicorn tennis_ai.api:app --app-dir src --reload
```

Set `LIVE_TENNIS_API_KEY` in `.env` for live upcoming fixtures. The secret is never committed.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Repository structure

```text
src/tennis_ai/    feature state, live-data integration, inference, API
notebooks/        audit, feature engineering, model selection, inference
models/           fitted production pipelines and Elo state
reports/          evaluation metrics
scripts/          cache and deployment utilities
tests/            provider and result-tracking tests
```

## Data attribution and licensing

Historical tennis data was compiled by [Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp) and obtained through the [Aneeshers archive](https://github.com/Aneeshers/tennis-sackmann-archive). Those datasets remain licensed under **CC BY-NC-SA 4.0** and are used here for a non-commercial educational project.

Project source code is released under the MIT License. Dataset files and dataset-derived artifacts are excluded from the MIT grant and retain their original licensing conditions. See [LICENSE](LICENSE).

## Disclaimer

UMTennis predictions are probabilistic research outputs, not guaranteed outcomes or financial advice.

## Author

Developed by [Umut Ali Yalçın](https://github.com/umutaylcn).
