"""Free upcoming ATP fixture access through Live Tennis API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import requests


API_BASE_URL = "https://api.livetennisapi.com/api/public/v1"

# The provider occasionally labels national wildcard playoffs as the related
# tour event.  Reject anything dated before the official main-draw start.
MAIN_DRAW_START_DATES: dict[tuple[str, int], date] = {
    ("US Open", 2026): date(2026, 8, 30),
}


class TennisAPIError(RuntimeError):
    """Raised when the tennis provider returns an unsuccessful response."""


@dataclass(frozen=True)
class UpcomingMatch:
    match_id: int
    event_date: str | None
    start_time: str | None
    p1_id: int | None
    p1_name: str
    p2_id: int | None
    p2_name: str
    tournament_name: str
    surface: str | None
    round_name: str | None
    round_code: str | None
    status: str | None
    is_qualifying: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_api_key(env_path: str | Path | None = None) -> str:
    """Load the secret API key without exposing it in logs or errors."""
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("LIVE_TENNIS_API_KEY", "").strip()
    if not api_key:
        raise TennisAPIError(
            "LIVE_TENNIS_API_KEY bulunamadı. Proje kökündeki .env dosyasına ekle."
        )
    return api_key


class LiveTennisClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key boş olamaz")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "LiveTennisClient":
        return cls(load_api_key(env_path))

    def _get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = self._session.get(
                f"{API_BASE_URL}/{endpoint.lstrip('/')}",
                headers=headers,
                params=params,
                timeout=self._timeout_seconds,
            )
            if response.status_code == 429:
                raise TennisAPIError("API günlük veya dakikalık request limitine ulaştı")
            response.raise_for_status()
            payload = response.json()
        except TennisAPIError:
            raise
        except (requests.RequestException, ValueError) as exc:
            raise TennisAPIError(f"Live Tennis API bağlantısı başarısız: {exc}") from exc

        if not isinstance(payload, dict):
            raise TennisAPIError("API beklenmeyen bir response döndürdü")
        if payload.get("error"):
            detail = payload.get("detail") or payload["error"]
            raise TennisAPIError(f"API isteği reddedildi: {detail}")
        return payload

    def get_upcoming_matches(self) -> list[UpcomingMatch]:
        """Return main-tour ATP singles fixtures, earliest first."""
        payload = self._get(
            "fixtures",
            tour="atp",
            draw="singles",
            limit=200,
            offset=0,
        )

        fixtures = payload.get("data", [])
        if not isinstance(fixtures, list):
            raise TennisAPIError("Fixtures response içindeki data bir liste değil")

        upcoming: list[UpcomingMatch] = []
        for fixture in fixtures:
            if bool(fixture.get("is_qualifying", False)):
                continue

            tournament_name = str(fixture.get("tournament") or "").strip()
            event_date_text = str(fixture.get("event_date") or "").strip()
            try:
                event_day = date.fromisoformat(event_date_text)
            except ValueError:
                event_day = None
            official_start = (
                MAIN_DRAW_START_DATES.get((tournament_name, event_day.year))
                if event_day is not None
                else None
            )
            if official_start is not None and event_day < official_start:
                continue

            p1_name = str(fixture.get("player1_name") or "").strip()
            p2_name = str(fixture.get("player2_name") or "").strip()
            if not p1_name or not p2_name:
                continue

            upcoming.append(
                UpcomingMatch(
                    match_id=int(fixture["id"]),
                    event_date=fixture.get("event_date"),
                    start_time=fixture.get("start_time"),
                    p1_id=fixture.get("player1_id"),
                    p1_name=p1_name,
                    p2_id=fixture.get("player2_id"),
                    p2_name=p2_name,
                    tournament_name=tournament_name,
                    surface=fixture.get("surface"),
                    round_name=fixture.get("round"),
                    round_code=fixture.get("round_code"),
                    status=fixture.get("status"),
                    is_qualifying=False,
                )
            )

        return upcoming

    def get_usage(self) -> dict[str, Any]:
        """Check quota usage; the provider documents this call as quota-exempt."""
        return self._get("usage")

    def get_player(self, player_id: int) -> dict[str, Any]:
        """Return one resolved player profile (available on the free plan)."""
        return self._get(f"players/{player_id}")

    def get_match(self, match_id: int) -> dict[str, Any]:
        """Return one match throughout its lifecycle (available on the free plan)."""
        return self._get(f"matches/{int(match_id)}")
