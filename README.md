# UMTennis API

Production FastAPI service for UMTennis upcoming ATP fixtures, player state,
Elo rankings, pre-match statistics, and ensemble win probabilities.

## Runtime

- Build: `pip install -r requirements.txt`
- Start: `uvicorn tennis_ai.api:app --app-dir src --host 0.0.0.0 --port $PORT`
- Health: `/api/health`

`LIVE_TENNIS_API_KEY` must be configured as a deployment secret.
