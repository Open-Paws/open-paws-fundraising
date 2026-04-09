"""
Impact reporting: campaign outcomes → donor-facing impact statements.

Takes campaign outcomes (animals helped, legislative wins, corporate commitments)
and maps them to donor-facing impact statements for stewardship and re-engagement.

Integrates with churn prediction:
- Donors with HIGH churn risk receive impact reports before re-engagement sequences
- Impact data is org-controlled — orgs decide what to share with supporters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CampaignOutcome:
    """A concrete outcome from an advocacy campaign."""

    campaign_name: str
    outcome_type: str  # "animals_helped", "policy_win", "corporate_commitment", "direct_action"
    description: str
    metric_value: Optional[float] = None  # e.g. 50000 for animals_helped
    metric_unit: Optional[str] = None     # e.g. "farmed animals", "hens", "pigs"
    date_achieved: Optional[date] = None


@dataclass
class DonorImpactReport:
    """
    Impact report for a specific supporter.

    Generated before re-engagement sequences for HIGH/LAPSED churn risk donors.
    Connects their donation history to specific campaign outcomes.
    """

    donor_id: str
    org_id: str
    org_name: str
    period_start: date
    period_end: date
    outcomes: list[CampaignOutcome] = field(default_factory=list)
    total_donated_period: float = 0.0
    impact_statements: list[str] = field(default_factory=list)
    headline: str = ""


def build_impact_report(
    donor_id: str,
    org_id: str,
    org_name: str,
    total_donated: float,
    campaign_outcomes: list[CampaignOutcome],
    period_start: date,
    period_end: date,
) -> DonorImpactReport:
    """
    Build a donor impact report from campaign outcomes.

    Generates human-readable impact statements mapping donor's contribution
    to specific campaign victories. Used before re-engagement sequences.

    Args:
        donor_id: Pseudonymous donor identifier.
        org_id: Coalition partner org identifier.
        org_name: Human-readable org name for display.
        total_donated: Total amount donated by this supporter in the period.
        campaign_outcomes: List of campaign outcomes to report on.
        period_start: Start of reporting period.
        period_end: End of reporting period.

    Returns:
        DonorImpactReport with generated impact_statements and headline.
    """
    report = DonorImpactReport(
        donor_id=donor_id,
        org_id=org_id,
        org_name=org_name,
        period_start=period_start,
        period_end=period_end,
        outcomes=campaign_outcomes,
        total_donated_period=total_donated,
    )

    statements = []
    for outcome in campaign_outcomes:
        statement = _outcome_to_statement(outcome, total_donated)
        if statement:
            statements.append(statement)

    report.impact_statements = statements
    report.headline = _build_headline(campaign_outcomes, org_name)

    return report


def _outcome_to_statement(outcome: CampaignOutcome, total_donated: float) -> str:
    """Convert a single campaign outcome to a donor-facing impact statement."""
    if outcome.outcome_type == "animals_helped":
        if outcome.metric_value and outcome.metric_unit:
            return (
                f"Your support helped {int(outcome.metric_value):,} {outcome.metric_unit} "
                f"through {outcome.campaign_name}."
            )
        return f"Your support contributed to helping animals through {outcome.campaign_name}."

    if outcome.outcome_type == "policy_win":
        return (
            f"Your support helped win a policy victory: {outcome.description} "
            f"({outcome.campaign_name})."
        )

    if outcome.outcome_type == "corporate_commitment":
        return (
            f"Your support pressured a corporation to commit: {outcome.description}. "
            f"Campaign: {outcome.campaign_name}."
        )

    if outcome.outcome_type == "direct_action":
        return (
            f"Your support enabled direct action: {outcome.description} "
            f"({outcome.campaign_name})."
        )

    return f"Your support contributed to {outcome.campaign_name}: {outcome.description}."


def _build_headline(outcomes: list[CampaignOutcome], org_name: str) -> str:
    """Build a summary headline for the impact report."""
    if not outcomes:
        return f"Your support is making a difference for {org_name}."

    # Count animals helped
    total_animals = sum(
        int(o.metric_value)
        for o in outcomes
        if o.outcome_type == "animals_helped" and o.metric_value
    )

    policy_wins = [o for o in outcomes if o.outcome_type == "policy_win"]
    corporate_wins = [o for o in outcomes if o.outcome_type == "corporate_commitment"]

    if total_animals > 0:
        return (
            f"Together, we helped {total_animals:,} animals "
            f"and won {len(policy_wins) + len(corporate_wins)} "
            f"campaign{'s' if len(policy_wins) + len(corporate_wins) != 1 else ''}."
        )

    if policy_wins or corporate_wins:
        count = len(policy_wins) + len(corporate_wins)
        return f"Together, we won {count} campaign victory{'' if count == 1 else 'ies'} for animals."

    return f"Your support drove real change through {len(outcomes)} campaign{'s' if len(outcomes) != 1 else ''}."


def format_impact_report_email(report: DonorImpactReport) -> dict:
    """
    Format an impact report as email content for stewardship sequences.

    Returns dict with subject_line, preview_text, body_markdown.
    This output feeds directly into re-engagement sequences.
    """
    period_str = (
        f"{report.period_start.strftime('%B %Y')} – {report.period_end.strftime('%B %Y')}"
    )

    subject = f"Your impact at {report.org_name}: {report.headline}"
    preview = f"See what your support helped accomplish {period_str}."

    body_lines = [
        f"# Your Impact Report — {period_str}",
        "",
        f"**{report.headline}**",
        "",
        "Here's what your support made possible:",
        "",
    ]

    for statement in report.impact_statements:
        body_lines.append(f"- {statement}")

    body_lines += [
        "",
        f"Your contribution during this period: **${report.total_donated_period:,.2f}**",
        "",
        "Every dollar goes directly to this work. Thank you.",
        "",
        f"With gratitude,  ",
        f"{report.org_name}",
    ]

    return {
        "subject_line": subject[:80],
        "preview_text": preview[:90],
        "body_markdown": "\n".join(body_lines),
    }


def priority_donors_for_impact_report(
    donors: list,  # list[Donor] — avoid circular import
    min_churn_probability: float = 0.40,
) -> list:
    """
    Filter donors who should receive an impact report before re-engagement.

    Logic: donors with MEDIUM or higher churn risk get an impact report
    sent before the re-engagement sequence (softens the ask with proof of impact).

    Args:
        donors: List of Donor objects with churn_risk set.
        min_churn_probability: Minimum churn probability to include.

    Returns:
        Filtered list of donors sorted by churn_probability descending.
    """
    eligible = [
        d for d in donors
        if d.churn_probability is not None
        and d.churn_probability >= min_churn_probability
    ]
    return sorted(eligible, key=lambda d: d.churn_probability or 0.0, reverse=True)
