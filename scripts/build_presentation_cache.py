"""Precompute immutable player histories so API cold starts stay lightweight."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_ai.presentation import PlayerPresentationService


def main() -> None:
    state = joblib.load(PROJECT_ROOT / "models" / "current_state.joblib")
    service = PlayerPresentationService(PROJECT_ROOT, state, use_cache=False)
    print(
        f"Presentation cache ready: {len(service._active_names)} players, "
        f"{sum(len(history) for history in service._matches.values())} match records"
    )


if __name__ == "__main__":
    main()
