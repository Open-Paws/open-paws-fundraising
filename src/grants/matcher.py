"""
Semantic grant matching against the seed_grants.json database.

Approach:
1. Keyword matching (always available, no API needed)
2. Optional: sentence-transformers semantic similarity (install with [embeddings] extra)

Takes an org profile and returns ranked grants with match_score and rationale.

Usage:
    matcher = GrantMatcher()
    matches = matcher.match(org_profile)
    for m in matches[:5]:
        print(m["grant"]["grant_name"], m["match_score"])
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent / "seed_grants.json"


def _load_grants() -> list[dict]:
    with open(_SEED_PATH, encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for keyword matching."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def _keyword_score(org_profile: dict, grant: dict) -> tuple[float, list[str]]:
    """
    Score a grant against an org profile using keyword overlap.

    Returns (score 0.0–1.0, list of matched keywords).

    Weighted factors:
    - Focus area keyword overlap (primary signal)
    - Geographic compatibility
    - Budget range compatibility
    """
    mission = _normalize(org_profile.get("mission", ""))
    programs = _normalize(" ".join(org_profile.get("programs", [])))
    org_text = mission + " " + programs

    grant_focus = " ".join(grant.get("focus_areas", []))
    grant_text = _normalize(grant_focus)

    # Extract meaningful keywords (filter stopwords)
    stopwords = {
        "and", "or", "the", "a", "an", "in", "of", "for", "to", "with",
        "is", "are", "that", "this", "on", "at", "by", "from",
    }

    grant_keywords = {
        w for w in grant_text.split() if len(w) > 3 and w not in stopwords
    }
    org_keywords = {
        w for w in org_text.split() if len(w) > 3 and w not in stopwords
    }

    if not grant_keywords:
        return 0.0, []

    matched = grant_keywords & org_keywords
    keyword_score = len(matched) / len(grant_keywords)

    # Geographic compatibility check
    geo = grant.get("geographic_restrictions", "global").lower()
    org_geo = _normalize(org_profile.get("geography", "US"))
    geo_penalty = 0.0
    if "uk only" in geo and "uk" not in org_geo and "united kingdom" not in org_geo:
        geo_penalty = 0.4
    elif "europe only" in geo and not any(
        kw in org_geo for kw in ["europe", "eu", "uk", "germany", "france"]
    ):
        geo_penalty = 0.3
    elif "us only" in geo and not any(
        kw in org_geo for kw in ["us", "united states", "america"]
    ):
        geo_penalty = 0.3

    # Budget range compatibility
    org_budget = org_profile.get("annual_budget", 0)
    grant_min = grant.get("amount_min", 0)
    grant_max = grant.get("amount_max", float("inf"))
    budget_penalty = 0.0
    if org_budget > 0:
        # Org too large for grant (asking for grant smaller than 1% of budget — poor ROI)
        if grant_max < org_budget * 0.01:
            budget_penalty = 0.2
        # Grant too large for org capacity (grant > 5x annual budget — capacity concern)
        if grant_min > org_budget * 5:
            budget_penalty = 0.15

    final_score = max(0.0, keyword_score - geo_penalty - budget_penalty)
    return round(final_score, 4), sorted(matched)


def _semantic_score(org_text: str, grant_text: str) -> Optional[float]:
    """
    Optional semantic similarity score using sentence-transformers.

    Returns None if sentence-transformers is not installed.
    Install with: pip install open-paws-fundraising[embeddings]
    """
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore[import]
    except ImportError:
        return None

    model = _get_embedding_model()
    if model is None:
        return None

    emb_org = model.encode(org_text, convert_to_tensor=True)
    emb_grant = model.encode(grant_text, convert_to_tensor=True)
    similarity = float(util.cos_sim(emb_org, emb_grant))
    return round(max(0.0, min(1.0, similarity)), 4)


_embedding_model = None


def _get_embedding_model():
    """Lazy-load embedding model (cached after first call)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformers all-MiniLM-L6-v2 for semantic matching")
        return _embedding_model
    except ImportError:
        return None


class GrantMatcher:
    """
    Match coalition partner org profiles against the grant database.

    Org profile schema:
    {
        "org_id": "str",
        "org_name": "str",
        "mission": "str",
        "programs": ["list", "of", "program", "descriptions"],
        "geography": "str (e.g. 'US', 'Europe', 'global')",
        "annual_budget": int,           # Optional — for budget fit scoring
        "previous_funders": ["str"],    # Optional — skip already-awarded grants
    }
    """

    def __init__(self, grants: Optional[list[dict]] = None) -> None:
        self._grants = grants if grants is not None else _load_grants()

    def match(
        self,
        org_profile: dict,
        min_score: float = 0.1,
        use_semantic: bool = True,
    ) -> list[dict]:
        """
        Match org profile against all grants and return ranked results.

        Args:
            org_profile: Organization profile dict (see class docstring for schema).
            min_score: Minimum match score to include in results (0.0–1.0).
            use_semantic: Whether to attempt semantic scoring with sentence-transformers.
                          Falls back to keyword-only if library not installed.

        Returns:
            List of match dicts, sorted by match_score descending:
            {
                "grant": {grant dict},
                "match_score": float,
                "matched_keywords": [str],
                "rationale": str,
                "scoring_method": "semantic" | "keyword",
            }
        """
        mission = org_profile.get("mission", "")
        programs = " ".join(org_profile.get("programs", []))
        org_text = f"{mission} {programs}".strip()

        results = []
        for grant in self._grants:
            kw_score, matched_keywords = _keyword_score(org_profile, grant)

            if use_semantic:
                grant_text = " ".join(grant.get("focus_areas", [])) + " " + grant.get(
                    "grant_name", ""
                )
                sem_score = _semantic_score(org_text, grant_text)
            else:
                sem_score = None

            if sem_score is not None:
                # Blend: 60% semantic + 40% keyword (semantic is more nuanced)
                final_score = round(0.60 * sem_score + 0.40 * kw_score, 4)
                method = "semantic"
            else:
                final_score = kw_score
                method = "keyword"

            if final_score < min_score:
                continue

            rationale = _build_rationale(grant, matched_keywords, final_score)

            results.append({
                "grant": grant,
                "match_score": final_score,
                "matched_keywords": matched_keywords,
                "rationale": rationale,
                "scoring_method": method,
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results

    def top_matches(
        self,
        org_profile: dict,
        n: int = 10,
        min_score: float = 0.1,
    ) -> list[dict]:
        """Return top N matches for an org profile."""
        return self.match(org_profile, min_score=min_score)[:n]


def _build_rationale(grant: dict, matched_keywords: list[str], score: float) -> str:
    """Build a human-readable rationale string for a match."""
    funder = grant.get("funder", "Funder")
    grant_name = grant.get("grant_name", "Grant")
    geo = grant.get("geographic_restrictions", "global")
    deadline = grant.get("deadline_pattern", "varies")
    amount_min = grant.get("amount_min", 0)
    amount_max = grant.get("amount_max", 0)

    amount_str = ""
    if amount_min and amount_max:
        amount_str = f"${amount_min:,}–${amount_max:,}"
    elif amount_max:
        amount_str = f"up to ${amount_max:,}"

    keyword_str = ""
    if matched_keywords:
        keyword_str = f" Matching areas: {', '.join(matched_keywords[:5])}."

    score_label = "Strong" if score >= 0.5 else "Moderate" if score >= 0.25 else "Partial"

    return (
        f"{score_label} alignment with {funder}'s {grant_name}."
        f"{keyword_str}"
        f" Award range: {amount_str}. Deadline: {deadline}. Geography: {geo}."
        f" Notes: {grant.get('notes', '')}"
    )
