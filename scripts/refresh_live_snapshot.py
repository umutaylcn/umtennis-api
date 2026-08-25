"""Rebuild presentation and upcoming caches from the latest serialized state."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_ai.fixture_pipeline import build_upcoming_fixture_table
from tennis_ai.live_data import LiveTennisClient
from tennis_ai.presentation import PlayerPresentationService


def main() -> None:
    state = joblib.load(PROJECT_ROOT / "models" / "current_state.joblib")
    presentation = PlayerPresentationService(PROJECT_ROOT, state, use_cache=False)
    client = LiveTennisClient.from_env(WORKSPACE_ROOT / ".env")
    fixtures = build_upcoming_fixture_table(
        PROJECT_ROOT,
        client,
        historical_names=state.players.keys(),
    )
    print(
        f"Presentation cache: {len(presentation._active_names)} players | "
        f"Upcoming fixtures: {len(fixtures)} | State as of: {state.state_as_of}"
    )


if __name__ == "__main__":
    main()
