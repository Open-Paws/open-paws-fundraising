"""
Grant application drafter — adapted from grantmatch-ai's writer.py.

Generates professional grant application components using Claude:
- Letter of Inquiry (LOI)
- Executive summary
- Program narrative outline
- Budget narrative stub

Static system prompt is placed first for cache optimization.
Uses claude-haiku-4-5 for cost efficiency on drafting tasks.
Requires ANTHROPIC_API_KEY in environment.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Static system prompt — placed first for maximum cache hit rate.
# This block is identical across all drafting calls.
_SYSTEM_PROMPT = """You are an expert grant writer with 20+ years of experience securing
funding for animal advocacy organizations. You have a proven track record with
Open Philanthropy, Wilks Foundation, Humane Society Foundation, and similar funders.

You write flowing, professional prose — never bracketed placeholders like [Add here].
If any value is missing, write around it naturally without drawing attention to gaps.

Movement language requirements:
- "farmed animals" not "livestock" or "farm animals"
- "factory farm" not "farm" or "production facility"
- "slaughterhouse" not "processing facility"
- "supporters" or "activists" not "users"
- "campaign" not "project"
- Never use speciesist idioms

Tone: Confident, specific, urgent. Written by a seasoned development director, not AI.

Format all output as structured markdown with clear section headers."""


def draft_application(
    grant: dict,
    org_profile: dict,
    impact_data: Optional[dict] = None,
    match_rationale: Optional[str] = None,
) -> dict:
    """
    Generate a full grant application draft for a matched grant opportunity.

    Args:
        grant: Grant dict from seed_grants.json (or matcher output).
        org_profile: Organization profile with name, mission, programs, geography, budget.
        impact_data: Optional impact metrics dict (animals helped, campaigns won, etc.).
        match_rationale: Optional match rationale from GrantMatcher (used to strengthen narrative).

    Returns:
        Dict with sections: loi, executive_summary, program_narrative_outline,
        budget_narrative_stub, metadata.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; returning template draft")
        return _template_draft(grant, org_profile)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; returning template draft")
        return _template_draft(grant, org_profile)

    client = anthropic.Anthropic(api_key=api_key)

    # Build a single comprehensive user prompt
    user_prompt = _build_user_prompt(grant, org_profile, impact_data, match_rationale)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",  # Cost-efficient for drafting
            max_tokens=4000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        return _parse_sections(raw, grant, org_profile)

    except Exception as exc:
        logger.error("Application drafting failed: %s", exc)
        return _template_draft(grant, org_profile)


def _build_user_prompt(
    grant: dict,
    org_profile: dict,
    impact_data: Optional[dict],
    match_rationale: Optional[str],
) -> str:
    org_name = org_profile.get("org_name", "Our Organization")
    mission = org_profile.get("mission", "")
    programs = "; ".join(org_profile.get("programs", []))
    geography = org_profile.get("geography", "")
    budget = org_profile.get("annual_budget", 0)
    budget_str = f"${budget:,}" if budget else "confidential"

    grant_name = grant.get("grant_name", "Grant Opportunity")
    funder = grant.get("funder", "Funder")
    focus_areas = ", ".join(grant.get("focus_areas", []))
    amount_max = grant.get("amount_max", 0)
    amount_str = f"${amount_max:,}" if amount_max else "not specified"
    deadline = grant.get("deadline_pattern", "see website")
    geo_restrictions = grant.get("geographic_restrictions", "global")
    funder_notes = grant.get("notes", "")

    impact_str = ""
    if impact_data:
        impact_str = "\n\nIMPACT DATA TO INCORPORATE:\n"
        for key, value in impact_data.items():
            impact_str += f"- {key}: {value}\n"

    rationale_str = ""
    if match_rationale:
        rationale_str = f"\n\nMATCH ANALYSIS:\n{match_rationale}"

    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    return f"""Generate a complete grant application draft for the following opportunity.

ORGANIZATION:
- Name: {org_name}
- Mission: {mission}
- Programs: {programs}
- Geography: {geography}
- Annual budget: {budget_str}

GRANT OPPORTUNITY:
- Grant name: {grant_name}
- Funder: {funder}
- Focus areas: {focus_areas}
- Maximum award: {amount_str}
- Deadline pattern: {deadline}
- Geographic restrictions: {geo_restrictions}
- Funder notes: {funder_notes}

Today's date: {today}
{impact_str}{rationale_str}

Generate ALL FOUR sections below. Use exact section headers.

## LETTER OF INQUIRY

Write a complete LOI (600–900 words). Structure:
1. Letterhead block (org name, date, formal header)
2. Salutation to {funder} Grants Review Committee
3. Opening: purpose and why {funder} is the ideal partner
4. The Need: problem the org addresses
5. The Fit: direct connection to funder's focus areas
6. The Impact: specific outcomes and measurable goals
7. Closing: funding ask, call to action, signature

## EXECUTIVE SUMMARY

Write a 150-word executive summary for the cover page.
Include: org name, grant requested amount, core program, primary outcomes.

## PROGRAM NARRATIVE OUTLINE

Write a structured outline with bullet points for each section:
- Problem Statement
- Proposed Approach
- Evaluation Plan
- Organizational Capacity
- Sustainability Plan

## BUDGET NARRATIVE STUB

Write a 100-word budget narrative paragraph explaining how funds will be deployed.
Do not include actual numbers — this is a narrative template.
"""


def _parse_sections(raw: str, grant: dict, org_profile: dict) -> dict:
    """Parse the LLM output into structured sections."""
    sections = {
        "loi": "",
        "executive_summary": "",
        "program_narrative_outline": "",
        "budget_narrative_stub": "",
    }

    # Split on section headers
    markers = {
        "loi": "## LETTER OF INQUIRY",
        "executive_summary": "## EXECUTIVE SUMMARY",
        "program_narrative_outline": "## PROGRAM NARRATIVE OUTLINE",
        "budget_narrative_stub": "## BUDGET NARRATIVE STUB",
    }

    lines = raw.split("\n")
    current_section = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

    for line in lines:
        matched = False
        for key, marker in markers.items():
            if line.strip().upper().startswith(marker.upper().lstrip("# ").strip()):
                flush()
                current_section = key
                current_lines = []
                matched = True
                break
        if not matched and current_section is not None:
            current_lines.append(line)

    flush()

    # If parsing failed, store raw output in loi
    if not any(sections.values()):
        sections["loi"] = raw

    return {
        **sections,
        "metadata": {
            "grant_name": grant.get("grant_name"),
            "funder": grant.get("funder"),
            "org_name": org_profile.get("org_name"),
            "amount_max": grant.get("amount_max"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _template_draft(grant: dict, org_profile: dict) -> dict:
    """Minimal template fallback when LLM is unavailable."""
    org_name = org_profile.get("org_name", "Our Organization")
    funder = grant.get("funder", "Funder")
    grant_name = grant.get("grant_name", "Grant")
    focus_areas = ", ".join(grant.get("focus_areas", []))
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")

    loi = f"""# Letter of Inquiry — {grant_name}

**{org_name}**
{today}

Dear {funder} Grants Review Committee,

We write to express our intent to apply for the {grant_name}. Our organization's work
directly advances the priorities outlined in your funding guidelines.

[Complete this section with your mission statement and the specific problem you address.]

Our approach aligns with your focus on {focus_areas}.

[Describe your program approach and anticipated outcomes.]

We welcome the opportunity to submit a full proposal and discuss partnership.

Sincerely,
{org_name}
"""

    return {
        "loi": loi,
        "executive_summary": f"[Complete executive summary for {org_name}'s application to {funder}.]",
        "program_narrative_outline": (
            "- Problem Statement: [Describe the problem]\n"
            "- Proposed Approach: [Describe your program]\n"
            "- Evaluation Plan: [Describe how you'll measure success]\n"
            "- Organizational Capacity: [Describe your team and track record]\n"
            "- Sustainability Plan: [Describe long-term funding strategy]"
        ),
        "budget_narrative_stub": (
            "[Describe how grant funds will be deployed across personnel, "
            "program expenses, and overhead.]"
        ),
        "metadata": {
            "grant_name": grant_name,
            "funder": funder,
            "org_name": org_name,
            "amount_max": grant.get("amount_max"),
            "generated_at": today,
        },
    }
