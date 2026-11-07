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

VALID_PUBLIC_ROUNDS = {"R128", "R64", "R32", "R16", "QF", "SF", "F", "RR"}


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
    parser.add_argument("--bundle-output", type=Path)
    args = parser.parse_args()

    service = PredictionService(PROJECT_ROOT)
    # Refresh the durable pick with the latest valid pre-match state before
    # exporting the same probability to Upcoming and, later, Previous.
    fixtures = service.fixtures()
    service.prediction_history.capture(fixtures, service.state, service.predictor)
    round_mismatches = service.prediction_history.pending_round_mismatches(fixtures)
    if round_mismatches:
        raise ValueError(f"Pre-match rounds disagree with prediction history: {round_mismatches}")
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
    if args.bundle_output is not None:
        # A provider fixture with an unknown round can be a qualifier. It is
        # safer to omit it than to publish a confident main-draw prediction.
        safe_matches = [
            match for match in matches
            if str(match.get("round") or "").upper() in VALID_PUBLIC_ROUNDS
        ]
        safe_predictions = {
            str(match["match_id"]): predictions[str(match["match_id"])]
            for match in safe_matches
        }
        write_json(args.bundle_output, {
            "generated_at_utc": generated_at,
            "state_as_of_utc": payload["state_as_of_utc"],
            "upcoming": {"count": len(safe_matches), "matches": safe_matches,
                         "predictions": safe_predictions},
            "previous": {"count": len(previous), "matches": previous},
        })
        print(f"Public bundle: {len(safe_matches)} verified-round upcoming; "
              f"{len(matches) - len(safe_matches)} unknown-round omitted")
    print(f"Cloudflare snapshot: {len(matches)} upcoming/details, {len(previous)} previous")


if __name__ == "__main__":
    main()
