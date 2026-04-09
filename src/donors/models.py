"""
Domain models for donor churn prediction.

Behavioral signals only — no PII stored here.
All donor_id values are pseudonymous identifiers assigned by the coalition
partner's CRM before data reaches this platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class ChurnRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    LAPSED = "LAPSED"


class DonorSegment(str, Enum):
    MAJOR = "MAJOR"           # >$1000/year
    MID_LEVEL = "MID_LEVEL"   # $100–999/year
    GRASSROOTS = "GRASSROOTS" # <$100/year
    LAPSED = "LAPSED"         # No donation in 12+ months
    NEW = "NEW"               # First donation within 90 days


@dataclass
class Donor:
    """
    A supporter's behavioral profile scoped to one coalition partner org.

    Contains only aggregated behavioral signals — never raw PII.
    The coalition partner's CRM assigns donor_id before sending data here.
    """

    donor_id: str               # Pseudonymous ID — no PII stored here
    org_id: str                 # Which coalition partner org this supporter belongs to
    segment: DonorSegment

    # Behavioral signals (no PII)
    total_donations_12mo: float
    donation_count_12mo: int
    days_since_last_donation: int
    average_gift_size: float
    campaigns_engaged: int      # How many campaigns they've opened/clicked
    preferred_campaign_topic: Optional[str]

    # Engagement signals
    open_rate_last_90d: float = 0.0
    click_rate_last_90d: float = 0.0
    is_recurring_donor: bool = False

    # Churn prediction output (populated by ChurnPredictor.predict)
    churn_risk: Optional[ChurnRisk] = None
    churn_probability: Optional[float] = None
    recommended_action: Optional[str] = None


@dataclass
class DonorCohort:
    """Summary statistics for a group of donors within one org."""

    org_id: str
    total_donors: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    lapsed_count: int
    predicted_churn_rate: float
    average_risk_score: float
    as_of_date: date
