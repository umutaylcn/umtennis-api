"""Resolve live-provider player identities against the historical model data."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any, Iterable
import unicodedata


def normalize_player_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_name = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


class PlayerProfileCache:
    """Small persistent cache so free API quota is not spent repeatedly."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            self._profiles = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._profiles: dict[str, dict[str, Any]] = {}

    def get(self, player_id: int) -> dict[str, Any] | None:
        return self._profiles.get(str(player_id))

    def set(self, player_id: int, profile: dict[str, Any]) -> None:
        self._profiles[str(player_id)] = profile
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(self._profiles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


class HistoricalPlayerMatcher:
    def __init__(self, historical_names: Iterable[str]) -> None:
        names = sorted({str(name).strip() for name in historical_names if str(name).strip()})
        self._names_by_normalized: dict[str, list[str]] = {}
        self._names_by_compact: dict[str, list[str]] = {}
        self._names_by_token_set: dict[str, list[str]] = {}
        for name in names:
            normalized = normalize_player_name(name)
            self._names_by_normalized.setdefault(normalized, []).append(name)
            self._names_by_compact.setdefault(normalized.replace(" ", ""), []).append(name)
            token_set = " ".join(sorted(normalized.split()))
            self._names_by_token_set.setdefault(token_set, []).append(name)

    def match(self, live_name: str) -> tuple[str | None, float, str]:
        normalized_live = normalize_player_name(live_name)
        exact = self._names_by_normalized.get(normalized_live, [])
        if len(exact) == 1:
            return exact[0], 1.0, "exact"
        if len(exact) > 1:
            return None, 1.0, "ambiguous_exact"

        compact = self._names_by_compact.get(normalized_live.replace(" ", ""), [])
        if len(compact) == 1:
            return compact[0], 1.0, "exact_compact"
        if len(compact) > 1:
            return None, 1.0, "ambiguous_compact"

        token_set = " ".join(sorted(normalized_live.split()))
        reordered = self._names_by_token_set.get(token_set, [])
        if len(reordered) == 1:
            return reordered[0], 1.0, "exact_token_order"
        if len(reordered) > 1:
            return None, 1.0, "ambiguous_token_order"

        live_tokens = normalized_live.split()
        live_surname = live_tokens[-1] if live_tokens else ""
        candidates: list[tuple[float, str]] = []

        for normalized_historical, original_names in self._names_by_normalized.items():
            historical_tokens = normalized_historical.split()
            if live_surname and historical_tokens and historical_tokens[-1] != live_surname:
                continue
            score = SequenceMatcher(None, normalized_live, normalized_historical).ratio()
            for original_name in original_names:
                candidates.append((score, original_name))

        best_by_name: dict[str, float] = {}
        for score, name in candidates:
            best_by_name[name] = max(score, best_by_name.get(name, 0.0))
        candidates = sorted(
            ((score, name) for name, score in best_by_name.items()),
            reverse=True,
        )
        if not candidates or candidates[0][0] < 0.86:
            return None, candidates[0][0] if candidates else 0.0, "unresolved"

        best_score, best_name = candidates[0]
        if len(candidates) > 1 and best_score - candidates[1][0] < 0.03:
            return None, best_score, "ambiguous_fuzzy"
        return best_name, best_score, "fuzzy"

    def match_surname_initial(self, short_name: str) -> tuple[str | None, float, str]:
        """Match provider names such as ``Zverev A.`` to full historical names."""
        normalized = normalize_player_name(short_name)
        tokens = normalized.split()
        if len(tokens) < 2 or len(tokens[-1]) != 1:
            return self.match(short_name)

        initial_tokens: list[str] = []
        while tokens and len(tokens[-1]) == 1:
            initial_tokens.insert(0, tokens.pop())
        if not tokens or not initial_tokens:
            return self.match(short_name)

        provider_family = "".join(tokens)
        candidates: list[tuple[float, str]] = []

        for historical_names in self._names_by_normalized.values():
            for historical_name in historical_names:
                historical_tokens = normalize_player_name(historical_name).split()
                # Do not treat another provider-style abbreviation (for example
                # ``Sesko Z.``) as a full historical identity. A one-letter family
                # token can otherwise fuzzy-match unrelated surnames containing
                # that letter, such as ``Gorzny S.``.
                if any(len(token) == 1 for token in historical_tokens):
                    continue
                for family_start in range(1, len(historical_tokens)):
                    given_initials = [
                        token[0] for token in historical_tokens[:family_start]
                    ]
                    initials_match = all(
                        initial in given_initials for initial in initial_tokens
                    )
                    if not initials_match:
                        continue
                    candidate_family = "".join(historical_tokens[family_start:])
                    score = SequenceMatcher(None, provider_family, candidate_family).ratio()
                    if provider_family in candidate_family or candidate_family in provider_family:
                        score = max(score, 0.95)
                    candidates.append((score, historical_name))

        best_by_name: dict[str, float] = {}
        for score, name in candidates:
            best_by_name[name] = max(score, best_by_name.get(name, 0.0))
        candidates = sorted(
            ((score, name) for name, score in best_by_name.items()),
            reverse=True,
        )
        if not candidates or candidates[0][0] < 0.82:
            return None, candidates[0][0] if candidates else 0.0, "unresolved_short"
        best_score, best_name = candidates[0]
        if len(candidates) > 1 and best_score - candidates[1][0] < 0.025:
            return None, best_score, "ambiguous_short"
        return best_name, best_score, "surname_initial"
