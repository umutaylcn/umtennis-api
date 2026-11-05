"""Export public match data so site visits never wait for the Python API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tennis_ai.api import PredictionService


def write_json(output: Path, payload: dict) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-output", type=Path, required=True)
    args = parser.parse_args()

    service = PredictionService(PROJECT_ROOT)
    matches = service.match_list()
    generated_at = datetime.now(timezone.utc).isoformat()
    predictions = {str(match["match_id"]): service.predict(match["match_id"]) for match in matches}
    payload = {
        "generated_at_utc": generated_at,
        "state_as_of_utc": service.state.state_as_of.isoformat()
        if service.state.state_as_of is not None else None,
        "count": len(matches),
        "matches": matches,
        "predictions": predictions,
    }
    previous = service.previous_match_list()
    # Build and validate every payload before replacing any public asset.
    for data in (payload, previous):
        json.dumps(data, ensure_ascii=False, allow_nan=False)
    write_json(args.output, payload)
    write_json(args.previous_output, {
        "generated_at_utc": generated_at,
        "count": len(previous),
        "matches": previous,
    })
    print(f"Cloudflare snapshot: {len(matches)} upcoming/details, {len(previous)} previous")


if __name__ == "__main__":
    main()
