"""Fail closed before a scheduled update publishes its public snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(bundle: dict, previous_bundle: dict | None = None) -> None:
    upcoming = bundle["upcoming"]
    previous = bundle["previous"]
    matches = upcoming["matches"]
    results = previous["matches"]
    ids = [str(match["match_id"]) for match in matches]
    result_ids = [str(match["match_id"]) for match in results]
    if len(ids) != len(set(ids)) or len(result_ids) != len(set(result_ids)):
        raise ValueError("Duplicate match IDs in public bundle")
    if upcoming["count"] != len(ids) or previous["count"] != len(result_ids):
        raise ValueError("Incorrect public bundle counts")
    if set(upcoming["predictions"]) != set(ids):
        raise ValueError("Upcoming predictions and fixtures differ")
    if set(ids) & set(result_ids):
        raise ValueError("Completed match still listed as upcoming")
    if previous_bundle is not None:
        old_results = {str(item["match_id"]) for item in previous_bundle["previous"]["matches"]}
        if not old_results.issubset(result_ids):
            raise ValueError("Scheduled update would remove existing Previous matches")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--previous-bundle", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    previous = None
    if args.previous_bundle is not None and args.previous_bundle.exists():
        previous = json.loads(args.previous_bundle.read_text(encoding="utf-8"))
    validate(bundle, previous)
    print(f"AUDIT OK: {bundle['upcoming']['count']} upcoming; "
          f"{bundle['previous']['count']} previous")


if __name__ == "__main__":
    main()
