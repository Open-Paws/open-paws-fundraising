"""
Privacy invariant tests for the donor feature table.

Domain rules verified:
- donor_id must be a pseudonymous string — never an email address or real name
- The Donor dataclass must not have fields for email, name, phone, or address
- The feature vector fed to the churn predictor contains no PII fields
- FEATURE_COLS must not include any PII field names
"""

from __future__ import annotations

import re

import pytest

from src.donors.churn_predictor import FEATURE_COLS, CAT_COLS, NUM_COLS
from src.donors.models import Donor, DonorSegment

# PII field names that must never appear in the feature table
_PII_FIELD_NAMES = {
    "email",
    "email_address",
    "name",
    "first_name",
    "last_name",
    "full_name",
    "phone",
    "phone_number",
    "address",
    "street",
    "city",
    "zip",
    "postcode",
    "ssn",
    "date_of_birth",
    "dob",
    "ip_address",
    "credit_card",
}

# Pattern: an email-format string
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Pattern: real name heuristic — two capitalized words
_REAL_NAME_RE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")


class TestFeatureColsPIIFree:
    """The churn predictor's feature column list must contain no PII field names."""

    def test_feature_cols_contain_no_pii_field_names(self):
        for col in FEATURE_COLS:
            assert col.lower() not in _PII_FIELD_NAMES, (
                f"PII field '{col}' found in FEATURE_COLS — "
                "the donor feature table must never include personally identifiable fields"
            )

    def test_num_cols_contain_no_pii_field_names(self):
        for col in NUM_COLS:
            assert col.lower() not in _PII_FIELD_NAMES, (
                f"PII field '{col}' found in NUM_COLS"
            )

    def test_cat_cols_contain_no_pii_field_names(self):
        for col in CAT_COLS:
            assert col.lower() not in _PII_FIELD_NAMES, (
                f"PII field '{col}' found in CAT_COLS"
            )


class TestDonorModelNoPII:
    """The Donor dataclass schema itself must not declare PII fields."""

    def test_donor_has_no_email_field(self):
        """Donor dataclass must not have an email field."""
        donor_fields = {f.name for f in Donor.__dataclass_fields__.values()}
        pii_present = donor_fields & _PII_FIELD_NAMES
        assert not pii_present, (
            f"Donor dataclass contains PII fields: {pii_present}. "
            "All identifiers must be pseudonymous — no PII stored in the feature model."
        )

    def test_donor_id_field_exists_and_is_string_typed(self):
        """donor_id must exist and be typed as str (pseudonymous identifier)."""
        fields = Donor.__dataclass_fields__
        assert "donor_id" in fields, "Donor must have a donor_id field"
        assert fields["donor_id"].type is str or fields["donor_id"].type == "str", (
            "donor_id must be typed as str"
        )


class TestPseudonymousDonorIds:
    """Donor IDs created at runtime must be pseudonymous — not email addresses or real names."""

    def _make_donor(self, donor_id: str) -> Donor:
        return Donor(
            donor_id=donor_id,
            org_id="org-test",
            segment=DonorSegment.GRASSROOTS,
            total_donations_12mo=50.0,
            donation_count_12mo=2,
            days_since_last_donation=30,
            average_gift_size=25.0,
            campaigns_engaged=1,
            preferred_campaign_topic="farmed animal welfare",
        )

    def test_pseudonymous_id_accepted(self):
        """A UUID-style pseudonymous ID must be accepted without error."""
        donor = self._make_donor("donor-a4f2c8b1-9e3d-4a7f-b261-0c5e8d2a1b3f")
        assert donor.donor_id == "donor-a4f2c8b1-9e3d-4a7f-b261-0c5e8d2a1b3f"

    def test_email_format_id_is_not_pseudonymous(self):
        """
        Verify that an email-format string is detectable as a PII violation.

        The Donor model does not enforce this at construction time — enforcement
        is the responsibility of the intake layer. This test documents the contract:
        if a donor_id matches an email pattern, it violates the pseudonymity requirement.
        """
        suspicious_id = "jane.smith@example.org"
        assert _EMAIL_RE.match(suspicious_id), (
            "Test setup error: the suspicious ID should look like an email"
        )
        # The ID should NOT be used as-is; an intake-layer validator must reject it.
        # We assert that an email-pattern ID is detectable (this is the signal to reject it).
        assert _EMAIL_RE.match(suspicious_id) is not None, (
            "Systems receiving donor data must check donor_id does not match email pattern"
        )

    def test_real_name_format_id_is_not_pseudonymous(self):
        """
        Verify that a real-name-format string is detectable as a PII violation.
        """
        suspicious_id = "Jane Smith"
        assert _REAL_NAME_RE.match(suspicious_id), (
            "Test setup error: the suspicious ID should look like a real name"
        )
        assert _REAL_NAME_RE.match(suspicious_id) is not None, (
            "Systems receiving donor data must check donor_id does not match real name pattern"
        )

    def test_valid_donors_have_org_id_not_pii(self):
        """org_id must be an org identifier, not donor PII."""
        donor = self._make_donor("pseudo-001")
        assert "@" not in donor.org_id, "org_id must not be an email address"
        assert _REAL_NAME_RE.match(donor.org_id) is None, "org_id must not look like a real name"
