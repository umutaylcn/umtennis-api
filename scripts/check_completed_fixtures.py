"""Hide finished published fixtures without changing predictions, Elo or CSV data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "public" / "completed_fixtures.json"
SITE_URL = "https://prediction.umtennis.workers.dev/data/upcoming.json"
API_BASE = "https://api.livetennisapi.com/api/public/v1"
TERMINAL_EVENTS = {"walk over", "cancelled"}


def get_json(url: str, *, token: str | None = None, allow_not_found: bool = False) -> dict:
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            value = json.load(response)
    except HTTPError as exc:
        if allow_not_found and exc.code == 404:
            return {}
        raise RuntimeError(f"Fixture check failed for {url.split('?')[0]}: HTTP {exc.code}") from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Fixture check failed for {url.split('?')[0]}: {exc}") from exc
    if not isinstance(value, dict) or value.get("error"):
        raise RuntimeError(f"Invalid fixture response from {url.split('?')[0]}")
    return value


def normalized_name(name: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(name or ""))
    ascii_name = "".join(char for char in ascii_name if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


def same_fixture(fixture: dict, detail: dict) -> bool:
    players = detail.get("players") or {}
    profiles = [players.get("p1") or {}, players.get("p2") or {}]
    fixture_ids = [fixture.get("p1_id"), fixture.get("p2_id")]
    detail_ids = [profile.get("id") for profile in profiles]
    if all(value is not None for value in (*fixture_ids, *detail_ids)):
        same_players = sorted(map(str, fixture_ids)) == sorted(map(str, detail_ids))
    else:
        fixture_names = [normalized_name(fixture.get(f"p{side}_name")) for side in (1, 2)]
        detail_names = [normalized_name(profile.get("name")) for profile in profiles]
        same_players = all(fixture_names) and sorted(fixture_names) == sorted(detail_names)
    if not same_players:
        return False
    try:
        fixture_time = datetime.fromisoformat(str(fixture["start_time_utc"]).replace("Z", "+00:00"))
        detail_time = datetime.fromisoformat(str(detail["scheduled_time"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return True
    return abs(fixture_time - detail_time) <= timedelta(days=2)


def signature(fixture: dict) -> dict:
    return {
        "match_id": int(fixture["match_id"]),
        "p1_name": str(fixture["p1_name"]),
        "p2_name": str(fixture["p2_name"]),
        "tournament_name": str(fixture["tournament_name"]),
        "start_time_utc": str(fixture["start_time_utc"]),
    }


def is_terminal(detail: dict) -> bool:
    return (str(detail.get("status") or "").casefold() in {"completed", "cancelled"}
            or str(detail.get("event_status") or "").casefold() in TERMINAL_EVENTS)


def write_atomic(payload: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    try:
        os.replace(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    published = get_json(SITE_URL)
    fixtures = published.get("matches")
    if not isinstance(fixtures, list):
        raise RuntimeError("Published upcoming response has no matches list")
    old = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.exists() else {"completed": []}
    completed = {int(item["match_id"]): item for item in old.get("completed", [])}
    now = datetime.now(timezone.utc)
    due = []
    for fixture in fixtures:
        match_id = int(fixture["match_id"])
        if match_id in completed:
            continue
        try:
            start = datetime.fromisoformat(str(fixture["start_time_utc"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if now - timedelta(days=3) <= start <= now + timedelta(hours=2):
            due.append((start, fixture))
    due.sort(key=lambda item: item[0])
    if not due:
        print("No published matches need a result check")
        return
    token = os.environ.get("LIVE_TENNIS_API_KEY", "").strip()
    if not token:
        raise RuntimeError("LIVE_TENNIS_API_KEY is missing")
    usage = get_json(f"{API_BASE}/usage", token=token)
    remaining = int(usage["today"]["remaining_day"])
    budget = min(14, max(0, remaining - 10))
    if budget == 0:
        raise RuntimeError("Provider quota is too low to check published matches")
    checked = 0
    newly_completed = 0
    for _, fixture in due[:budget]:
        match_id = int(fixture["match_id"])
        detail = get_json(f"{API_BASE}/matches/{match_id}", token=token, allow_not_found=True)
        checked += 1
        if same_fixture(fixture, detail) and is_terminal(detail):
            completed[match_id] = signature(fixture)
            newly_completed += 1
    if newly_completed:
        write_atomic({"completed": list(completed.values())})
    print(f"Published fixtures checked: {checked}; newly terminal: {newly_completed}")


if __name__ == "__main__":
    main()
