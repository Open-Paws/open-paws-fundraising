"""
Donor segmentation logic.

Segments supporters into MAJOR / MID_LEVEL / GRASSROOTS / LAPSED / NEW
based on behavioral signals. No PII involved.
"""

from __future__ import annotations

from .models import Donor, DonorSegment


def classify_segment(
    total_donations_12mo: float,
    donation_count_12mo: int,
    days_since_last_donation: int,
    days_since_first_donation: int,
) -> DonorSegment:
    """
    Classify a supporter into a segment based on behavioral signals.

    Rules applied in priority order:
    1. LAPSED — no donation in 12+ months
    2. NEW — first donation within 90 days
    3. MAJOR — >$1000 in last 12 months
    4. MID_LEVEL — $100–$999 in last 12 months
    5. GRASSROOTS — any active donor below $100
    """
    if days_since_last_donation >= 365:
        return DonorSegment.LAPSED

    if days_since_first_donation <= 90:
        return DonorSegment.NEW

    if total_donations_12mo >= 1000:
        return DonorSegment.MAJOR

    if total_donations_12mo >= 100:
        return DonorSegment.MID_LEVEL

    return DonorSegment.GRASSROOTS


def segment_cohort(donors: list[Donor]) -> dict[DonorSegment, list[Donor]]:
    """Group donors by segment. Returns dict keyed by DonorSegment."""
    result: dict[DonorSegment, list[Donor]] = {seg: [] for seg in DonorSegment}
    for donor in donors:
        result[donor.segment].append(donor)
    return result


def priority_order(donors: list[Donor]) -> list[Donor]:
    """
    Sort donors for outreach prioritization.

    Priority: HIGH churn risk > MEDIUM > LAPSED > LOW.
    Within each risk tier, sort by total_donations_12mo descending
    (major donors get attention first).
    """
    risk_rank = {
        "HIGH": 0,
        "LAPSED": 1,
        "MEDIUM": 2,
        "LOW": 3,
        None: 4,
    }
    return sorted(
        donors,
        key=lambda d: (
            risk_rank.get(d.churn_risk.value if d.churn_risk else None, 4),
            -d.total_donations_12mo,
        ),
    )
