"""
Re-engagement sequence generator for supporters at HIGH or LAPSED churn risk.

Generates personalized re-engagement messages using Claude with a static
system prompt (placed first for cache optimization).

Three tone options:
- IMPACT_UPDATE: what we accomplished since you last gave
- URGENT_NEED: current campaign that needs support now
- PERSONAL: from the executive director

Requires ANTHROPIC_API_KEY in environment.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

from .models import ChurnRisk, Donor

logger = logging.getLogger(__name__)


class ReengagementTone(str, Enum):
    IMPACT_UPDATE = "IMPACT_UPDATE"   # What we accomplished since last gift
    URGENT_NEED = "URGENT_NEED"       # Current campaign that needs support
    PERSONAL = "PERSONAL"             # From the executive director


# Static system prompt placed first for cache optimization.
# This block never changes — the LLM should cache it across calls.
_SYSTEM_PROMPT = """You are a compassionate nonprofit fundraising writer for an animal advocacy organization.
Your role is to write authentic, non-manipulative re-engagement emails that reconnect
lapsed supporters with the movement's work.

Core principles:
- Lead with impact and progress, not guilt
- Be specific about animals helped and campaigns won
- Respect the supporter's autonomy — never pressure
- Use movement terminology: "supporters" not "donors", "campaign" not "project",
  "farmed animals" not "livestock", "slaughterhouse" not "processing facility"
- Never use speciesist idioms or language that normalizes animal harm
- Tone: warm, direct, honest

Format: Return JSON with exactly these keys:
{
  "subject_line": "string (max 60 chars)",
  "preview_text": "string (max 90 chars, shown in email client after subject)",
  "body_markdown": "string (full email body in markdown, 150-250 words)"
}"""


def generate_reengagement_email(
    donor: Donor,
    tone: ReengagementTone,
    org_name: str,
    recent_wins: Optional[list[str]] = None,
    current_campaign: Optional[str] = None,
    ed_name: Optional[str] = None,
) -> dict:
    """
    Generate a personalized re-engagement email for a lapsed supporter.

    Args:
        donor: Donor with HIGH or LAPSED churn_risk. Must have churn_risk set.
        tone: Which re-engagement angle to use.
        org_name: Name of the coalition partner organization.
        recent_wins: Optional list of recent campaign victories (for IMPACT_UPDATE tone).
        current_campaign: Optional campaign description (for URGENT_NEED tone).
        ed_name: Executive director's name (for PERSONAL tone).

    Returns:
        Dict with subject_line, preview_text, body_markdown.

    Raises:
        ValueError: If donor churn_risk is LOW or MEDIUM (not worth re-engagement cost).
        RuntimeError: If LLM call fails and no fallback applies.
    """
    if donor.churn_risk not in (ChurnRisk.HIGH, ChurnRisk.LAPSED):
        raise ValueError(
            f"Re-engagement is for HIGH/LAPSED donors only. Got: {donor.churn_risk}"
        )

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; returning template fallback")
        return _template_fallback(donor, tone, org_name)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; returning template fallback")
        return _template_fallback(donor, tone, org_name)

    client = anthropic.Anthropic(api_key=api_key)

    topic = donor.preferred_campaign_topic or "animal welfare"
    days_lapsed = donor.days_since_last_donation

    user_prompt = _build_user_prompt(
        tone=tone,
        org_name=org_name,
        topic=topic,
        days_lapsed=days_lapsed,
        recent_wins=recent_wins,
        current_campaign=current_campaign,
        ed_name=ed_name,
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",  # Cheapest model sufficient for email copy
            max_tokens=600,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        import json
        # Strip markdown code fences if present
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except Exception as exc:
        logger.error("Re-engagement LLM call failed: %s", exc)
        return _template_fallback(donor, tone, org_name)


def _build_user_prompt(
    tone: ReengagementTone,
    org_name: str,
    topic: str,
    days_lapsed: int,
    recent_wins: Optional[list[str]],
    current_campaign: Optional[str],
    ed_name: Optional[str],
) -> str:
    base = (
        f"Organization: {org_name}\n"
        f"Supporter's area of interest: {topic}\n"
        f"Days since last gift: {days_lapsed}\n\n"
    )

    if tone == ReengagementTone.IMPACT_UPDATE:
        wins_text = ""
        if recent_wins:
            wins_text = "Recent victories:\n" + "\n".join(f"- {w}" for w in recent_wins)
        return (
            base
            + f"Tone: IMPACT_UPDATE\n"
            + wins_text
            + "\nWrite an email reconnecting this supporter by sharing what the organization "
            + "has accomplished since they last gave. Focus on animals helped and campaigns won."
        )

    if tone == ReengagementTone.URGENT_NEED:
        campaign_text = current_campaign or "an active campaign to reduce farmed animal suffering"
        return (
            base
            + f"Tone: URGENT_NEED\n"
            + f"Current campaign: {campaign_text}\n"
            + "Write an email that shares a compelling current campaign and invites the supporter "
            + "to re-engage. Be specific about what's happening and why their support matters now."
        )

    if tone == ReengagementTone.PERSONAL:
        signer = ed_name or "the Executive Director"
        return (
            base
            + f"Tone: PERSONAL\n"
            + f"Signed by: {signer}\n"
            + "Write a personal email from the executive director. Warm, direct, honest. "
            + "Acknowledge the gap without guilt-tripping. Share why the work matters right now."
        )

    return base + "Write a warm re-engagement email."


def _template_fallback(donor: Donor, tone: ReengagementTone, org_name: str) -> dict:
    """Non-LLM fallback template when API is unavailable."""
    topic = donor.preferred_campaign_topic or "animal welfare"

    if tone == ReengagementTone.URGENT_NEED:
        subject = f"We need you back — {topic} campaign needs support"
        body = (
            f"Hi,\n\n"
            f"We haven't heard from you in a while, and we miss you.\n\n"
            f"Right now, {org_name} is in the middle of a critical campaign on {topic}. "
            f"Your past support made a real difference, and we'd love to have you back.\n\n"
            f"Every action counts. Will you join us again?\n\n"
            f"With gratitude,\n{org_name}"
        )
    elif tone == ReengagementTone.PERSONAL:
        subject = f"A personal note from {org_name}"
        body = (
            f"Hi,\n\n"
            f"I wanted to reach out personally. You've been part of our community, "
            f"and your past support has meant so much to the animals we fight for.\n\n"
            f"I hope you'll consider rejoining us. The work continues, and every voice matters.\n\n"
            f"Warmly,\n{org_name}"
        )
    else:  # IMPACT_UPDATE
        subject = f"Look what we've accomplished together — {org_name}"
        body = (
            f"Hi,\n\n"
            f"We've been making real progress on {topic}, and we wanted to share it with you.\n\n"
            f"Your past gifts have been part of every victory. We'd love to have you back "
            f"as we continue this work.\n\n"
            f"Thank you for everything you've done.\n\n"
            f"{org_name}"
        )

    return {
        "subject_line": subject[:60],
        "preview_text": f"We miss you — see what {org_name} has been up to.",
        "body_markdown": body,
    }
