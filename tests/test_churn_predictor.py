"""
Tests for the rules-based donor churn predictor.

Validates that risk bands, boundary conditions, and the fallback predictor
produce correct results without requiring a trained model.
"""

from __future__ import annotations

import pytest

from src.donors.churn_predictor import (
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    ChurnPredictor,
    _recommendation,
    _risk_band,
    _rules_based_risk,
)
from src.donors.models import ChurnRisk, Donor, DonorSegment


def _make_donor(
    days_since_last_donation: int = 60,
    open_rate_last_90d: float = 0.3,
    campaigns_engaged: int = 3,
    is_recurring_donor: bool = False,
) -> Donor:
    return Donor(
        donor_id="test-donor",
        org_id="test-org",
        segment=DonorSegment.GRASSROOTS,
        total_donations_12mo=100.0,
        donation_count_12mo=2,
        days_since_last_donation=days_since_last_donation,
        average_gift_size=50.0,
        campaigns_engaged=campaigns_engaged,
        preferred_campaign_topic=None,
        open_rate_last_90d=open_rate_last_90d,
        is_recurring_donor=is_recurring_donor,
    )


class TestRiskBand:
    def test_score_at_high_threshold(self) -> None:
        assert _risk_band(HIGH_RISK_THRESHOLD) == ChurnRisk.HIGH

    def test_score_above_high_threshold(self) -> None:
        assert _risk_band(0.95) == ChurnRisk.HIGH

    def test_score_at_medium_threshold(self) -> None:
        assert _risk_band(MEDIUM_RISK_THRESHOLD) == ChurnRisk.MEDIUM

    def test_score_between_thresholds(self) -> None:
        mid_score = (HIGH_RISK_THRESHOLD + MEDIUM_RISK_THRESHOLD) / 2
        assert _risk_band(mid_score) == ChurnRisk.MEDIUM

    def test_score_below_medium_threshold(self) -> None:
        assert _risk_band(0.10) == ChurnRisk.LOW

    def test_zero_score_is_low(self) -> None:
        assert _risk_band(0.0) == ChurnRisk.LOW


class TestRulesBasedRisk:
    def test_365_plus_days_lapsed(self) -> None:
        donor = _make_donor(days_since_last_donation=365)
        risk, prob = _rules_based_risk(donor)
        assert risk == ChurnRisk.LAPSED
        assert prob >= 0.85

    def test_180_plus_days_high_risk(self) -> None:
        donor = _make_donor(days_since_last_donation=200)
        risk, prob = _rules_based_risk(donor)
        assert risk in (ChurnRisk.HIGH, ChurnRisk.MEDIUM)

    def test_recent_active_donor_low_risk(self) -> None:
        donor = _make_donor(
            days_since_last_donation=20,
            open_rate_last_90d=0.5,
            campaigns_engaged=5,
        )
        risk, prob = _rules_based_risk(donor)
        assert risk == ChurnRisk.LOW

    def test_low_engagement_bumps_score(self) -> None:
        """Low open rate and zero campaign engagement should raise risk score."""
        engaged_donor = _make_donor(
            days_since_last_donation=60, open_rate_last_90d=0.5, campaigns_engaged=5
        )
        disengaged_donor = _make_donor(
            days_since_last_donation=60, open_rate_last_90d=0.05, campaigns_engaged=0
        )
        _, prob_engaged = _rules_based_risk(engaged_donor)
        _, prob_disengaged = _rules_based_risk(disengaged_donor)
        assert prob_disengaged > prob_engaged

    def test_recurring_donor_lower_risk(self) -> None:
        """Recurring donors are more stable — their score should be lower."""
        one_off = _make_donor(days_since_last_donation=60, is_recurring_donor=False)
        recurring = _make_donor(days_since_last_donation=60, is_recurring_donor=True)
        _, prob_one_off = _rules_based_risk(one_off)
        _, prob_recurring = _rules_based_risk(recurring)
        assert prob_recurring <= prob_one_off

    def test_probability_in_valid_range(self) -> None:
        for days in [10, 90, 180, 300, 400]:
            donor = _make_donor(days_since_last_donation=days)
            _, prob = _rules_based_risk(donor)
            assert 0.0 <= prob <= 1.0, f"prob={prob} out of range for days={days}"


class TestRecommendation:
    def test_high_risk_recurring_donor(self) -> None:
        donor = _make_donor(is_recurring_donor=True)
        donor.churn_risk = ChurnRisk.HIGH
        rec = _recommendation(donor)
        assert "recurring" in rec.lower() or "personal" in rec.lower()

    def test_high_risk_low_engagement(self) -> None:
        donor = _make_donor(open_rate_last_90d=0.05, is_recurring_donor=False)
        donor.churn_risk = ChurnRisk.HIGH
        rec = _recommendation(donor)
        assert len(rec) > 0

    def test_medium_risk_recommendation(self) -> None:
        donor = _make_donor()
        donor.churn_risk = ChurnRisk.MEDIUM
        rec = _recommendation(donor)
        assert "stewardship" in rec.lower() or "impact" in rec.lower() or "re-engagement" in rec.lower()

    def test_low_risk_recommendation(self) -> None:
        donor = _make_donor()
        donor.churn_risk = ChurnRisk.LOW
        rec = _recommendation(donor)
        assert len(rec) > 0


class TestChurnPredictor:
    def test_rules_based_fallback_populates_fields(self) -> None:
        predictor = ChurnPredictor()
        donors = [
            _make_donor(days_since_last_donation=30),
            _make_donor(days_since_last_donation=400),
        ]
        result = predictor.predict(donors)
        for d in result:
            assert d.churn_risk is not None
            assert d.churn_probability is not None
            assert d.recommended_action is not None

    def test_lapsed_donor_classified_correctly(self) -> None:
        predictor = ChurnPredictor()
        lapsed = _make_donor(days_since_last_donation=400)
        result = predictor.predict([lapsed])
        assert result[0].churn_risk == ChurnRisk.LAPSED

    def test_returns_same_count_as_input(self) -> None:
        predictor = ChurnPredictor()
        donors = [_make_donor(days_since_last_donation=d) for d in [10, 60, 200, 400]]
        result = predictor.predict(donors)
        assert len(result) == 4

    def test_load_raises_on_missing_model_dir(self) -> None:
        predictor = ChurnPredictor()
        with pytest.raises(FileNotFoundError):
            predictor.load("/nonexistent/path/models")
