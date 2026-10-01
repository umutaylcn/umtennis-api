from pathlib import Path
from threading import RLock
import sys
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.api import PredictionService


class SnapshotOnlyTests(unittest.TestCase):
    def test_page_refresh_uses_snapshot_without_provider_request(self):
        service = object.__new__(PredictionService)
        service.project_root = ROOT
        service.ensure_current_artifacts = lambda: None
        service._lock = RLock()
        service._fixtures = pd.DataFrame()
        service._fixtures_loaded_at = 0.0
        service._prediction_cache = {}

        with patch("tennis_ai.api.USE_MOCK_FIXTURES", False), patch(
            "tennis_ai.api.load_fixture_snapshot", return_value=pd.DataFrame()
        ) as load, patch("requests.sessions.Session.get", side_effect=AssertionError("provider called")):
            self.assertTrue(service.fixtures(force_refresh=True).empty)

        load.assert_called_once_with(ROOT)
        self.assertGreater(service._fixtures_loaded_at, 0.0)


if __name__ == "__main__":
    unittest.main()
