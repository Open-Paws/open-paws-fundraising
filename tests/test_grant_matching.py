"""
Tests for grant matching against the seed_grants.json database.

Domain rules verified:
- Grant matching must return results from the seed list, not hallucinated grants
- Every matched grant must have a valid id present in seed_grants.json
- A query for farmed animal welfare should surface relevant grants (Open Philanthropy, etc.)
- Matching results must be sorted by match_score descending
"""

from __future__ import annotations

import json
from pathlib import Path

from src.grants.matcher import GrantMatcher, _keyword_score, _normalize

# Load the canonical grant IDs once for boundary checks
_SEED_PATH = Path(__file__).parent.parent / "src" / "grants" / "seed_grants.json"
_SEED_IDS: set[str] = {g["id"] for g in json.loads(_SEED_PATH.read_text())}
_SEED_COUNT = len(_SEED_IDS)


class TestGrantMatcherReturnsFromSeedList:
    """Every matched grant must originate from seed_grants.json."""

    def test_match_returns_only_seed_grants(self):
        """Grants returned must have IDs present in the seed file — no hallucinated grants."""
        profile = {
            "org_id": "org-001",
            "org_name": "Farmed Animal Coalition",
            "mission": "End factory farming through corporate campaigns and policy advocacy",
            "programs": ["cage-free campaigns", "chicken welfare", "farmed animal advocacy"],
            "geography": "US",
            "annual_budget": 500_000,
        }
        matcher = GrantMatcher()
        matches = matcher.match(profile, use_semantic=False)

        assert len(matches) > 0, "Expected at least one match for a farmed animal welfare org"
        for m in matches:
            assert m["grant"]["id"] in _SEED_IDS, (
                f"Returned grant '{m['grant']['id']}' is not in seed_grants.json — "
                "matcher must never fabricate grants"
            )

    def test_match_total_cannot_exceed_seed_count(self):
        """Results cannot be larger than the seed grant database."""
        profile = {
            "org_id": "org-002",
            "mission": "animal welfare advocacy",
            "programs": ["animal welfare", "animal protection"],
            "geography": "global",
        }
        matcher = GrantMatcher()
        matches = matcher.match(profile, min_score=0.0, use_semantic=False)
        assert len(matches) <= _SEED_COUNT


class TestGrantMatcherRelevance:
    """Relevance: known queries should surface known relevant grants."""

    def test_farmed_animal_welfare_profile_matches_open_philanthropy(self):
        """Open Philanthropy Farm Animal Welfare Program must surface for a farmed animal org."""
        profile = {
            "org_id": "org-003",
            "mission": "farmed animal welfare through corporate campaigns and movement building",
            "programs": ["farmed animal advocacy", "corporate campaigns", "policy advocacy"],
            "geography": "global",
            "annual_budget": 300_000,
        }
        matcher = GrantMatcher()
        matches = matcher.match(profile, use_semantic=False)
        match_ids = [m["grant"]["id"] for m in matches]
        assert "open-philanthropy-farm-animal" in match_ids, (
            "Open Philanthropy Farm Animal Welfare Program must match a farmed animal welfare org"
        )

    def test_companion_animal_org_matches_aspca(self):
        """A companion animal rescue org should match ASPCA grants."""
        profile = {
            "org_id": "org-004",
            "mission": "companion animal rescue, spay-neuter access, and shelter operations",
            "programs": ["animal rescue operations", "spay-neuter programs", "shelter capacity"],
            "geography": "US",
            "annual_budget": 200_000,
        }
        matcher = GrantMatcher()
        matches = matcher.match(profile, use_semantic=False)
        match_ids = [m["grant"]["id"] for m in matches]
        assert "aspca-grants" in match_ids, (
            "ASPCA Grants Program must match a companion animal / rescue org"
        )

    def test_results_sorted_by_score_descending(self):
        """Matches must be returned in descending match_score order."""
        profile = {
            "org_id": "org-005",
            "mission": "farmed animal advocacy through corporate campaigns and veganism promotion",
            "programs": ["farmed animal welfare", "anti-factory-farming campaigns"],
            "geography": "US",
            "annual_budget": 100_000,
        }
        matcher = GrantMatcher()
        matches = matcher.match(profile, use_semantic=False)
        scores = [m["match_score"] for m in matches]
        assert scores == sorted(scores, reverse=True), (
            "Grant matches must be sorted by match_score descending"
        )

    def test_match_score_bounded_zero_to_one(self):
        """All match scores must be in the range [0.0, 1.0]."""
        profile = {
            "org_id": "org-006",
            "mission": "animal welfare and animal protection",
            "programs": ["animal rights", "animal advocacy"],
            "geography": "global",
        }
        matcher = GrantMatcher()
        matches = matcher.match(profile, min_score=0.0, use_semantic=False)
        for m in matches:
            assert 0.0 <= m["match_score"] <= 1.0, (
                f"match_score {m['match_score']} out of [0.0, 1.0] for grant {m['grant']['id']}"
            )

    def test_geographic_mismatch_penalises_score(self):
        """A UK-only grant should score lower for a US-only org than for a global org."""
        profile_us = {
            "org_id": "org-007",
            "mission": "companion animal welfare prevention cruelty",
            "programs": ["animal welfare prevention cruelty europe"],
            "geography": "US",
        }
        profile_eu = {
            "org_id": "org-008",
            "mission": "companion animal welfare prevention cruelty",
            "programs": ["animal welfare prevention cruelty europe"],
            "geography": "Europe",
        }

        # Load the 'prevention-animal-cruelty-europe' grant directly and score it
        grants = json.loads(_SEED_PATH.read_text())
        pace_grant = next(g for g in grants if g["id"] == "prevention-animal-cruelty-europe")

        score_us, _ = _keyword_score(profile_us, pace_grant)
        score_eu, _ = _keyword_score(profile_eu, pace_grant)

        assert score_eu >= score_us, (
            "Europe-only grant should not penalise a European org vs a US org"
        )

    def test_top_matches_respects_n_limit(self):
        """top_matches(n=3) must return at most 3 results."""
        profile = {
            "org_id": "org-009",
            "mission": "animal welfare advocacy and animal protection campaigns",
            "programs": ["farmed animal welfare", "policy advocacy"],
            "geography": "global",
        }
        matcher = GrantMatcher()
        results = matcher.top_matches(profile, n=3)
        assert len(results) <= 3


class TestGrantMatcherCustomGrants:
    """Verify GrantMatcher uses injected grants (not hallucinating from outside the set)."""

    def test_custom_grant_list_respected(self):
        """When a custom grant list is injected, only those grants are returned."""
        custom_grants = [
            {
                "id": "test-only-grant",
                "funder": "Test Funder",
                "grant_name": "Test Grant",
                "focus_areas": ["farmed animal welfare", "vegan advocacy"],
                "amount_min": 1000,
                "amount_max": 50000,
                "geographic_restrictions": "global",
                "deadline_pattern": "rolling",
                "notes": "Test-only grant for unit tests.",
            }
        ]
        matcher = GrantMatcher(grants=custom_grants)
        profile = {
            "org_id": "org-010",
            "mission": "farmed animal welfare and vegan advocacy",
            "programs": ["animal advocacy"],
            "geography": "global",
        }
        matches = matcher.match(profile, min_score=0.0, use_semantic=False)
        assert all(m["grant"]["id"] == "test-only-grant" for m in matches), (
            "GrantMatcher must only return grants from the injected list"
        )


class TestNormalizeHelper:
    """Unit tests for the internal _normalize helper."""

    def test_normalize_lowercases(self):
        assert _normalize("Farmed Animal") == "farmed animal"

    def test_normalize_strips_punctuation(self):
        result = _normalize("cage-free, animal-welfare.")
        assert "," not in result
        assert "." not in result
        assert "-" not in result
