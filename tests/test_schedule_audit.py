import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_live_snapshot import _placeholder_clock_exposed


def test_date_only_in_batch_does_not_expose_placeholder_clock():
    group = pd.DataFrame({"precision": ["tba"] * 4 + ["date_only"]})
    assert not _placeholder_clock_exposed(group, "2026-10-27T09:00")


def test_precise_time_in_batch_exposes_placeholder_clock():
    group = pd.DataFrame({"precision": ["tba"] * 4 + ["time"]})
    assert _placeholder_clock_exposed(group, "2026-10-27T09:00")
