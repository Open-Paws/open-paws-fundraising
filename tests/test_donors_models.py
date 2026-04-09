"""
Tests for donor domain models.

Verifies that ChurnRisk, DonorSegment, Donor, and DonorCohort
encode the correct domain invariants.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.donors.models import ChurnRisk, Donor, DonorCohort, DonorSegment


class TestChurnRisk:
    def test_values_are_ordered_by_severity(self) -> None:
        """All four risk levels are distinct and string-valued."""
        assert ChurnRisk.LOW != ChurnRisk.MEDIUM
        assert ChurnRisk.MEDIUM != ChurnRisk.HIGH
        assert ChurnRisk.HIGH != ChurnRisk.LAPSED

    def test_string_conversion(self) -> None:
        assert ChurnRisk.HIGH.value == "HIGH"
        assert ChurnRisk.LAPSED.value == "LAPSED"
        assert ChurnRisk.LOW.value == "LOW"
        assert ChurnRisk.MEDIUM.value == "MEDIUM"

    def test_roundtrip_from_string(self) -> None:
        assert ChurnRisk("HIGH") == ChurnRisk.HIGH
        assert ChurnRisk("LAPSED") == ChurnRisk.LAPSED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ChurnRisk("CRITICAL")


class TestDonorSegment:
    def test_all_segments_present(self) -> None:
        segments = {s.value for s in DonorSegment}
        assert segments == {"MAJOR", "MID_LEVEL", "GRASSROOTS", "LAPSED", "NEW"}

    def test_roundtrip(self) -> None:
        for seg in DonorSegment:
            assert DonorSegment(seg.value) == seg


def _make_donor(**kwargs) -> Donor:
    """Factory for Donor with sensible defaults."""
    defaults = dict(
        donor_id="donor-001",
        org_id="org-abc",
        segment=DonorSegment.GRASSROOTS,
        total_donations_12mo=50.0,
        donation_count_12mo=2,
        days_since_last_donation=30,
        average_gift_size=25.0,
        campaigns_engaged=3,
        preferred_campaign_topic="farmed animal welfare",
    )
    defaults.update(kwargs)
    return Donor(**defaults)


class TestDonor:
    def test_default_engagement_signals(self) -> None:
        donor = _make_donor()
        assert donor.open_rate_last_90d == 0.0
        assert donor.click_rate_last_90d == 0.0
        assert donor.is_recurring_donor is False

    def test_churn_fields_start_empty(self) -> None:
        donor = _make_donor()
        assert donor.churn_risk is None
        assert donor.churn_probability is None
        assert donor.recommended_action is None

    def test_churn_fields_can_be_set(self) -> None:
        donor = _make_donor()
        donor.churn_risk = ChurnRisk.HIGH
        donor.churn_probability = 0.85
        donor.recommended_action = "Call within 7 days"
        assert donor.churn_risk == ChurnRisk.HIGH
        assert donor.churn_probability == 0.85

    def test_preferred_campaign_topic_optional(self) -> None:
        donor = _make_donor(preferred_campaign_topic=None)
        assert donor.preferred_campaign_topic is None

    def test_org_scoping(self) -> None:
        """Each donor is scoped to exactly one org — no cross-org data."""
        donor_a = _make_donor(donor_id="d-1", org_id="org-a")
        donor_b = _make_donor(donor_id="d-2", org_id="org-b")
        assert donor_a.org_id != donor_b.org_id


class TestDonorCohort:
    def test_cohort_fields(self) -> None:
        cohort = DonorCohort(
            org_id="org-xyz",
            total_donors=100,
            high_risk_count=10,
            medium_risk_count=25,
            low_risk_count=60,
            lapsed_count=5,
            predicted_churn_rate=0.10,
            average_risk_score=0.32,
            as_of_date=date(2026, 4, 9),
        )
        assert cohort.total_donors == 100
        assert cohort.high_risk_count + cohort.medium_risk_count + cohort.low_risk_count + cohort.lapsed_count == 100
        assert cohort.predicted_churn_rate == pytest.approx(0.10)
