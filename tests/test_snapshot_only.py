from pathlib import Path
from threading import RLock
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from starlette.requests import Request


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis_ai.api import PredictionService, app, require_api_token


class SnapshotOnlyTests(unittest.TestCase):
    @staticmethod
    def auth_request(value: str = "") -> Request:
        headers = [(b"authorization", value.encode())] if value else []
        return Request({"type": "http", "method": "GET", "path": "/api/matches", "headers": headers})

    def test_data_routes_fail_closed_without_a_token(self):
        with patch.dict(os.environ, {"UMTENNIS_API_TOKEN": ""}):
            with self.assertRaises(HTTPException) as error:
                require_api_token(self.auth_request())
        self.assertEqual(error.exception.status_code, 404)
        protected = {"/api/matches", "/api/previous-matches", "/api/matches/{match_id}/prediction"}
        for route in app.routes:
            if route.path in protected:
                self.assertTrue(any(dependency.dependency is require_api_token for dependency in route.dependencies))

    def test_data_routes_require_the_correct_bearer_token(self):
        with patch.dict(os.environ, {"UMTENNIS_API_TOKEN": "correct-long-secret"}):
            require_api_token(self.auth_request("Bearer correct-long-secret"))
            for supplied in ("Bearer incorrect", "Basic correct-long-secret", ""):
                with self.assertRaises(HTTPException) as error:
                    require_api_token(self.auth_request(supplied))
                self.assertEqual(error.exception.status_code, 401)

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
