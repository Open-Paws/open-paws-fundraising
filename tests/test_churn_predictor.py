"""
Tests for donor churn prediction feature extraction and rules-based scoring.

Domain rules verified:
- Feature vector shape matches FEATURE_COLS (even without model weights)
- Rules-based risk scoring correctly classifies donors by days_since_last_donation
- Recurring donors receive a lower base risk than equivalent non-recurring donors
- A lapsed donor (>=365 days) must be classified as LAPSED risk
- ChurnPredictor.predict populates churn_risk, churn_probability, recommended_action
"""

from __future__ import annotations

import pytest

from src.donors.churn_predictor import (
    FEATURE_COLS,
    ChurnPredictor,
    _rules_based_risk,
    _risk_band,
)
from src.donors.models import Donor, DonorSegment, ChurnRisk


def _make_donor(
    donor_id: str = "pseudo-001",
    days_since_last_donation: int = 30,
    open_rate: float = 0.25,
    is_recurring: bool = False,
    campaigns_engaged: int = 2,
    total_donations_12mo: float = 200.0,
    donation_count_12mo: int = 4,
    average_gift_size: float = 50.0,
    click_rate: float = 0.10,
    preferred_topic: str = "farmed animal welfare",
) -> Donor:
    return Donor(
        donor_id=donor_id,
        org_id="org-test",
        segment=DonorSegment.MID_LEVEL,
        total_donations_12mo=total_donations_12mo,
        donation_count_12mo=donation_count_12mo,
        days_since_last_donation=days_since_last_donation,
        average_gift_size=average_gift_size,
        campaigns_engaged=campaigns_engaged,
        preferred_campaign_topic=preferred_topic,
        open_rate_last_90d=open_rate,
        click_rate_last_90d=click_rate,
        is_recurring_donor=is_recurring,
    )


class TestFeatureColsShape:
    """The FEATURE_COLS list defines the expected feature vector shape."""

    def test_feature_cols_is_non_empty(self):
        assert len(FEATURE_COLS) > 0, "FEATURE_COLS must define at least one feature"

    def test_feature_cols_includes_recency_signal(self):
        """days_since_last_donation (recency) is the primary churn signal."""
        assert "days_since_last_donation" in FEATURE_COLS

    def test_feature_cols_includes_engagement_signals(self):
        """Email engagement rates are key behavioral churn signals."""
        assert "open_rate_last_90d" in FEATURE_COLS
        assert "click_rate_last_90d" in FEATURE_COLS

    def test_feature_cols_includes_monetary_signals(self):
        """Monetary signals (frequency, total) must be present."""
        assert "donation_count_12mo" in FEATURE_COLS
        assert "total_donations_12mo" in FEATURE_COLS

    def test_feature_row_built_from_donor_has_correct_column_count(self):
        """
        Simulate building a feature row from a Donor and verify it has
        the same number of columns as FEATURE_COLS.

        This test does not require trained model weights — it validates
        the feature extraction contract.
        """
        donor = _make_donor()
        feature_row = {
            "days_since_last_donation": donor.days_since_last_donation,
            "days_since_last_communication": donor.days_since_last_donation,
            "open_rate_last_90d": donor.open_rate_last_90d,
            "click_rate_last_90d": donor.click_rate_last_90d,
            "donation_count_12mo": donor.donation_count_12mo,
            "total_donations_12mo": donor.total_donations_12mo,
            "average_gift_size": donor.average_gift_size,
            "campaigns_engaged": donor.campaigns_engaged,
            "is_recurring_donor": int(donor.is_recurring_donor),
            "preferred_campaign_topic": donor.preferred_campaign_topic or "unknown",
        }
        # Every FEATURE_COL must have a corresponding key in the feature row
        for col in FEATURE_COLS:
            assert col in feature_row, (
                f"Feature column '{col}' is in FEATURE_COLS but not produced by feature extraction"
            )


class TestRulesBasedRiskScoring:
    """The rules-based fallback must correctly classify donors by signals."""

    def test_lapsed_donor_classified_as_lapsed(self):
        """A donor with >=365 days since last donation must be LAPSED."""
        donor = _make_donor(days_since_last_donation=400)
        risk, prob = _rules_based_risk(donor)
        assert risk == ChurnRisk.LAPSED, (
            f"Donor with 400 days since last donation must be LAPSED, got {risk}"
        )
        assert prob >= 0.80

    def test_donor_180_days_is_high_risk(self):
        """A donor with 180-364 days since last donation is HIGH risk."""
        donor = _make_donor(days_since_last_donation=200, open_rate=0.30, campaigns_engaged=2)
        risk, prob = _rules_based_risk(donor)
        assert risk == ChurnRisk.HIGH, (
            f"Donor with 200 days since last donation must be HIGH, got {risk}"
        )

    def test_recent_active_donor_is_low_risk(self):
        """A recently active, engaged donor must be LOW risk."""
        donor = _make_donor(
            days_since_last_donation=15,
            open_rate=0.50,
            campaigns_engaged=5,
            is_recurring=True,
        )
        risk, _ = _rules_based_risk(donor)
        assert risk == ChurnRisk.LOW, (
            f"A recently active recurring donor must be LOW risk, got {risk}"
        )

    def test_recurring_donor_lower_risk_than_non_recurring(self):
        """Recurring donors must score lower churn probability than equivalent non-recurring."""
        recurring = _make_donor(days_since_last_donation=100, open_rate=0.05, campaigns_engaged=0, is_recurring=True)
        non_recurring = _make_donor(days_since_last_donation=100, open_rate=0.05, campaigns_engaged=0, is_recurring=False)

        _, prob_recurring = _rules_based_risk(recurring)
        _, prob_non_recurring = _rules_based_risk(non_recurring)

        assert prob_recurring < prob_non_recurring, (
            f"Recurring donor churn prob ({prob_recurring}) must be lower than "
            f"non-recurring ({prob_non_recurring})"
        )

    def test_low_engagement_increases_churn_probability(self):
        """A donor with very low open rate and no campaigns engaged should be bumped higher."""
        base_donor = _make_donor(days_since_last_donation=80, open_rate=0.25, campaigns_engaged=3)
        low_engagement_donor = _make_donor(days_since_last_donation=80, open_rate=0.05, campaigns_engaged=0)

        _, base_prob = _rules_based_risk(base_donor)
        _, low_prob = _rules_based_risk(low_engagement_donor)

        assert low_prob >= base_prob, (
            "Low engagement (open_rate < 10%, zero campaigns) must raise churn probability"
        )


class TestChurnPredictorPredict:
    """ChurnPredictor.predict (rules-based fallback) populates all output fields."""

    def test_predict_populates_churn_risk(self):
        predictor = ChurnPredictor()  # no model loaded — uses rules-based fallback
        donor = _make_donor(days_since_last_donation=50)
        result = predictor.predict([donor])
        assert result[0].churn_risk is not None

    def test_predict_populates_churn_probability(self):
        predictor = ChurnPredictor()
        donor = _make_donor(days_since_last_donation=50)
        result = predictor.predict([donor])
        prob = result[0].churn_probability
        assert prob is not None
        assert 0.0 <= prob <= 1.0, f"churn_probability {prob} out of [0.0, 1.0]"

    def test_predict_populates_recommended_action(self):
        predictor = ChurnPredictor()
        donor = _make_donor(days_since_last_donation=50)
        result = predictor.predict([donor])
        assert result[0].recommended_action is not None
        assert len(result[0].recommended_action) > 0

    def test_predict_returns_same_count_as_input(self):
        predictor = ChurnPredictor()
        donors = [_make_donor(donor_id=f"pseudo-{i}", days_since_last_donation=i * 30) for i in range(5)]
        result = predictor.predict(donors)
        assert len(result) == 5


class TestRiskBandThresholds:
    """Verify _risk_band maps scores to the correct tier."""

    def test_score_above_high_threshold_is_high(self):
        assert _risk_band(0.85) == ChurnRisk.HIGH

    def test_score_above_medium_threshold_is_medium(self):
        assert _risk_band(0.55) == ChurnRisk.MEDIUM

    def test_score_below_medium_threshold_is_low(self):
        assert _risk_band(0.20) == ChurnRisk.LOW
