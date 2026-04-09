"""
Tests for the impact reporting module.

Domain rules verified:
- build_impact_report must produce correct aggregates from donor activity data
- Impact statements must reference real campaign outcomes
- Headline must reflect the number of animals helped
- priority_donors_for_impact_report must filter by churn probability correctly
- format_impact_report_email must produce non-empty subject, preview, and body
"""

from __future__ import annotations

from datetime import date

import pytest

from src.impact.reporting import (
    CampaignOutcome,
    build_impact_report,
    format_impact_report_email,
    priority_donors_for_impact_report,
)
from src.donors.models import Donor, DonorSegment, ChurnRisk


def _make_donor(donor_id: str, churn_prob: float, segment: DonorSegment = DonorSegment.MID_LEVEL) -> Donor:
    d = Donor(
        donor_id=donor_id,
        org_id="org-sanctuary",
        segment=segment,
        total_donations_12mo=250.0,
        donation_count_12mo=4,
        days_since_last_donation=60,
        average_gift_size=62.5,
        campaigns_engaged=3,
        preferred_campaign_topic="farmed animal welfare",
    )
    d.churn_probability = churn_prob
    d.churn_risk = ChurnRisk.HIGH if churn_prob >= 0.70 else ChurnRisk.MEDIUM
    return d


class TestBuildImpactReport:
    """Verify build_impact_report computes correct aggregates."""

    def test_report_contains_donor_id(self):
        report = build_impact_report(
            donor_id="pseudo-abc",
            org_id="org-001",
            org_name="Sanctuary Friends",
            total_donated=150.0,
            campaign_outcomes=[],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert report.donor_id == "pseudo-abc"

    def test_report_total_donated_period_set_correctly(self):
        report = build_impact_report(
            donor_id="pseudo-xyz",
            org_id="org-002",
            org_name="Farmed Animal Alliance",
            total_donated=500.0,
            campaign_outcomes=[],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert report.total_donated_period == 500.0

    def test_animals_helped_outcome_produces_impact_statement(self):
        """An animals_helped outcome must generate a statement mentioning the metric value."""
        outcome = CampaignOutcome(
            campaign_name="Hen Liberation Campaign",
            outcome_type="animals_helped",
            description="Freed hens from battery cages via corporate commitment",
            metric_value=50_000,
            metric_unit="hens",
            date_achieved=date(2025, 6, 1),
        )
        report = build_impact_report(
            donor_id="pseudo-hen",
            org_id="org-003",
            org_name="Hen Alliance",
            total_donated=200.0,
            campaign_outcomes=[outcome],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert len(report.impact_statements) == 1
        stmt = report.impact_statements[0]
        assert "50,000" in stmt or "50000" in stmt, (
            "Impact statement must include the metric value (number of hens)"
        )
        assert "hen" in stmt.lower()

    def test_policy_win_outcome_produces_impact_statement(self):
        outcome = CampaignOutcome(
            campaign_name="Ag-Gag Repeal Initiative",
            outcome_type="policy_win",
            description="Repealed ag-gag statute in state legislature",
            date_achieved=date(2025, 3, 15),
        )
        report = build_impact_report(
            donor_id="pseudo-policy",
            org_id="org-004",
            org_name="Policy Alliance",
            total_donated=300.0,
            campaign_outcomes=[outcome],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert len(report.impact_statements) == 1
        assert "policy" in report.impact_statements[0].lower() or "Ag-Gag" in report.impact_statements[0]

    def test_multiple_outcomes_produce_multiple_statements(self):
        outcomes = [
            CampaignOutcome(
                campaign_name="Campaign A",
                outcome_type="animals_helped",
                description="Helped farmed animals",
                metric_value=10_000,
                metric_unit="farmed animals",
            ),
            CampaignOutcome(
                campaign_name="Campaign B",
                outcome_type="corporate_commitment",
                description="Retailer committed to cage-free",
            ),
        ]
        report = build_impact_report(
            donor_id="pseudo-multi",
            org_id="org-005",
            org_name="Multi-Campaign Alliance",
            total_donated=400.0,
            campaign_outcomes=outcomes,
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert len(report.impact_statements) == 2

    def test_headline_mentions_animals_when_animals_helped(self):
        outcome = CampaignOutcome(
            campaign_name="Sanctuary Rescue",
            outcome_type="animals_helped",
            description="Rescued pigs from factory farm",
            metric_value=200,
            metric_unit="pigs",
        )
        report = build_impact_report(
            donor_id="pseudo-pig",
            org_id="org-006",
            org_name="Pig Sanctuary",
            total_donated=100.0,
            campaign_outcomes=[outcome],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert "200" in report.headline or "animal" in report.headline.lower(), (
            "Headline must mention the animal count when animals_helped outcome is present"
        )

    def test_no_outcomes_produces_fallback_headline(self):
        report = build_impact_report(
            donor_id="pseudo-empty",
            org_id="org-007",
            org_name="Animal Sanctuary",
            total_donated=50.0,
            campaign_outcomes=[],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert len(report.headline) > 0, "Headline must not be empty even with no outcomes"


class TestFormatImpactReportEmail:
    """Verify email formatting output structure."""

    def test_email_format_produces_subject_preview_body(self):
        outcome = CampaignOutcome(
            campaign_name="Farmed Animal Campaign",
            outcome_type="animals_helped",
            description="Improved conditions for farmed animals",
            metric_value=1_000,
            metric_unit="farmed animals",
        )
        report = build_impact_report(
            donor_id="pseudo-email",
            org_id="org-008",
            org_name="Farmed Animal Alliance",
            total_donated=150.0,
            campaign_outcomes=[outcome],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        email = format_impact_report_email(report)
        assert "subject_line" in email
        assert "preview_text" in email
        assert "body_markdown" in email
        assert len(email["subject_line"]) > 0
        assert len(email["body_markdown"]) > 0

    def test_subject_line_length_within_limit(self):
        report = build_impact_report(
            donor_id="pseudo-subject",
            org_id="org-009",
            org_name="Sanctuary Alliance",
            total_donated=200.0,
            campaign_outcomes=[],
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        email = format_impact_report_email(report)
        assert len(email["subject_line"]) <= 80, (
            "Subject line must not exceed 80 characters"
        )


class TestPriorityDonorsFilter:
    """Verify priority_donors_for_impact_report filters and sorts correctly."""

    def test_donors_below_threshold_excluded(self):
        """Donors below min_churn_probability must not appear in the priority list."""
        low_risk = _make_donor("pseudo-low", churn_prob=0.20)
        high_risk = _make_donor("pseudo-high", churn_prob=0.80)
        result = priority_donors_for_impact_report([low_risk, high_risk], min_churn_probability=0.40)
        ids = [d.donor_id for d in result]
        assert "pseudo-low" not in ids
        assert "pseudo-high" in ids

    def test_result_sorted_by_churn_probability_descending(self):
        donors = [
            _make_donor("pseudo-a", churn_prob=0.55),
            _make_donor("pseudo-b", churn_prob=0.90),
            _make_donor("pseudo-c", churn_prob=0.45),
        ]
        result = priority_donors_for_impact_report(donors, min_churn_probability=0.40)
        probs = [d.churn_probability for d in result]
        assert probs == sorted(probs, reverse=True), (
            "Priority donors must be sorted by churn_probability descending"
        )

    def test_empty_donor_list_returns_empty(self):
        result = priority_donors_for_impact_report([], min_churn_probability=0.40)
        assert result == []
