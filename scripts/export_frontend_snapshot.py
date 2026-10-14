"""Export the precomputed upcoming list as a Cloudflare-served static asset."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    service = PredictionService(PROJECT_ROOT)
    matches = service.match_list()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_as_of_utc": service.state.state_as_of.isoformat()
        if service.state.state_as_of is not None else None,
        "count": len(matches),
        "matches": matches,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(f"Cloudflare upcoming snapshot: {len(matches)} matches -> {output}")


if __name__ == "__main__":
    main()
