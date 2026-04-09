"""
Tests for donor segmentation logic.

The classify_segment function encodes the org's segmentation rules.
These tests verify that rule priority order is correct and that
boundary conditions are handled.
"""

from __future__ import annotations

import pytest

from src.donors.models import ChurnRisk, Donor, DonorSegment
from src.donors.segments import classify_segment, priority_order, segment_cohort


class TestClassifySegment:
    """Each rule is tested in isolation and at its boundary."""

    def test_lapsed_takes_priority_over_all(self) -> None:
        """If days_since_last_donation >= 365, always LAPSED regardless of revenue."""
        result = classify_segment(
            total_donations_12mo=5000.0,
            donation_count_12mo=10,
            days_since_last_donation=365,
            days_since_first_donation=500,
        )
        assert result == DonorSegment.LAPSED

    def test_364_days_is_not_lapsed(self) -> None:
        result = classify_segment(
            total_donations_12mo=50.0,
            donation_count_12mo=1,
            days_since_last_donation=364,
            days_since_first_donation=500,
        )
        assert result != DonorSegment.LAPSED

    def test_new_donor_within_90_days(self) -> None:
        result = classify_segment(
            total_donations_12mo=25.0,
            donation_count_12mo=1,
            days_since_last_donation=30,
            days_since_first_donation=30,
        )
        assert result == DonorSegment.NEW

    def test_91_days_is_not_new(self) -> None:
        result = classify_segment(
            total_donations_12mo=25.0,
            donation_count_12mo=1,
            days_since_last_donation=91,
            days_since_first_donation=91,
        )
        assert result != DonorSegment.NEW

    def test_major_donor_threshold(self) -> None:
        result = classify_segment(
            total_donations_12mo=1000.0,
            donation_count_12mo=5,
            days_since_last_donation=30,
            days_since_first_donation=400,
        )
        assert result == DonorSegment.MAJOR

    def test_below_major_threshold(self) -> None:
        result = classify_segment(
            total_donations_12mo=999.99,
            donation_count_12mo=4,
            days_since_last_donation=30,
            days_since_first_donation=400,
        )
        assert result == DonorSegment.MID_LEVEL

    def test_mid_level_lower_boundary(self) -> None:
        result = classify_segment(
            total_donations_12mo=100.0,
            donation_count_12mo=2,
            days_since_last_donation=30,
            days_since_first_donation=200,
        )
        assert result == DonorSegment.MID_LEVEL

    def test_grassroots_below_100(self) -> None:
        result = classify_segment(
            total_donations_12mo=99.99,
            donation_count_12mo=2,
            days_since_last_donation=60,
            days_since_first_donation=200,
        )
        assert result == DonorSegment.GRASSROOTS

    def test_zero_donation_active_is_grassroots(self) -> None:
        result = classify_segment(
            total_donations_12mo=0.0,
            donation_count_12mo=0,
            days_since_last_donation=60,
            days_since_first_donation=200,
        )
        assert result == DonorSegment.GRASSROOTS


def _make_donor(
    donor_id: str = "d-1",
    segment: DonorSegment = DonorSegment.GRASSROOTS,
    churn_risk: ChurnRisk | None = None,
    total_donations_12mo: float = 50.0,
) -> Donor:
    d = Donor(
        donor_id=donor_id,
        org_id="org-test",
        segment=segment,
        total_donations_12mo=total_donations_12mo,
        donation_count_12mo=2,
        days_since_last_donation=30,
        average_gift_size=25.0,
        campaigns_engaged=1,
        preferred_campaign_topic=None,
    )
    d.churn_risk = churn_risk
    return d


class TestSegmentCohort:
    def test_groups_by_segment(self) -> None:
        donors = [
            _make_donor("d-1", DonorSegment.MAJOR),
            _make_donor("d-2", DonorSegment.GRASSROOTS),
            _make_donor("d-3", DonorSegment.GRASSROOTS),
        ]
        cohort = segment_cohort(donors)
        assert len(cohort[DonorSegment.MAJOR]) == 1
        assert len(cohort[DonorSegment.GRASSROOTS]) == 2

    def test_empty_segments_present(self) -> None:
        """All segment keys exist even when empty."""
        cohort = segment_cohort([])
        for seg in DonorSegment:
            assert seg in cohort
            assert cohort[seg] == []


class TestPriorityOrder:
    def test_high_risk_before_medium(self) -> None:
        high = _make_donor("d-high", churn_risk=ChurnRisk.HIGH)
        medium = _make_donor("d-medium", churn_risk=ChurnRisk.MEDIUM)
        ordered = priority_order([medium, high])
        assert ordered[0].donor_id == "d-high"

    def test_lapsed_before_medium(self) -> None:
        lapsed = _make_donor("d-lapsed", churn_risk=ChurnRisk.LAPSED)
        medium = _make_donor("d-medium", churn_risk=ChurnRisk.MEDIUM)
        ordered = priority_order([medium, lapsed])
        assert ordered[0].donor_id == "d-lapsed"

    def test_major_donor_first_within_same_risk(self) -> None:
        small = _make_donor("d-small", churn_risk=ChurnRisk.HIGH, total_donations_12mo=50.0)
        major = _make_donor("d-major", churn_risk=ChurnRisk.HIGH, total_donations_12mo=2000.0)
        ordered = priority_order([small, major])
        assert ordered[0].donor_id == "d-major"

    def test_no_risk_sorts_last(self) -> None:
        no_risk = _make_donor("d-none", churn_risk=None)
        low = _make_donor("d-low", churn_risk=ChurnRisk.LOW)
        ordered = priority_order([no_risk, low])
        assert ordered[0].donor_id == "d-low"
        assert ordered[1].donor_id == "d-none"
