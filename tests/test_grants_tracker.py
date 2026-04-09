"""
Tests for the grant application pipeline tracker.

Uses an in-memory SQLite database to isolate tests from filesystem.
All tracker operations are scoped to org_id for coalition data isolation.
"""

from __future__ import annotations

import pytest

from src.grants.tracker import ApplicationStatus, GrantTracker


@pytest.fixture
def tracker(tmp_path) -> GrantTracker:
    """Tracker backed by a temporary database — isolated per test."""
    return GrantTracker(org_id="org-test", db_path=tmp_path / "test.db")


@pytest.fixture
def other_tracker(tmp_path) -> GrantTracker:
    """Tracker for a different org on the same database."""
    return GrantTracker(org_id="org-other", db_path=tmp_path / "test.db")


class TestApplicationStatus:
    def test_all_statuses_present(self) -> None:
        statuses = {s.value for s in ApplicationStatus}
        assert statuses == {"DRAFTING", "SUBMITTED", "UNDER_REVIEW", "AWARDED", "DECLINED"}

    def test_roundtrip(self) -> None:
        assert ApplicationStatus("AWARDED") == ApplicationStatus.AWARDED


class TestAddApplication:
    def test_add_returns_integer_id(self, tracker: GrantTracker) -> None:
        app_id = tracker.add_application(
            grant_id="grant-001",
            funder_name="Open Philanthropy",
            grant_name="Farm Animal Welfare",
        )
        assert isinstance(app_id, int)
        assert app_id > 0

    def test_new_application_has_drafting_status(self, tracker: GrantTracker) -> None:
        tracker.add_application(
            grant_id="grant-001",
            funder_name="Open Philanthropy",
            grant_name="Farm Animal Welfare",
        )
        pipeline = tracker.get_pipeline()
        assert len(pipeline) == 1
        assert pipeline[0]["status"] == "DRAFTING"

    def test_duplicate_grant_raises(self, tracker: GrantTracker) -> None:
        tracker.add_application(
            grant_id="grant-dup",
            funder_name="Funder A",
            grant_name="Grant A",
        )
        with pytest.raises(ValueError, match="already exists"):
            tracker.add_application(
                grant_id="grant-dup",
                funder_name="Funder A",
                grant_name="Grant A",
            )

    def test_optional_fields_stored(self, tracker: GrantTracker) -> None:
        tracker.add_application(
            grant_id="grant-opt",
            funder_name="Funder B",
            grant_name="Grant B",
            amount_requested=50000.0,
            notes="Apply by Q3",
        )
        all_apps = tracker.get_all()
        assert all_apps[0]["amount_requested"] == pytest.approx(50000.0)
        assert all_apps[0]["notes"] == "Apply by Q3"


class TestOrgIsolation:
    def test_different_orgs_see_only_own_applications(
        self, tracker: GrantTracker, other_tracker: GrantTracker
    ) -> None:
        """Coalition data isolation: each org sees only its own pipeline."""
        tracker.add_application("grant-shared-id", "Funder X", "Grant X")
        other_tracker.add_application("grant-shared-id", "Funder X", "Grant X")

        assert len(tracker.get_pipeline()) == 1
        assert len(other_tracker.get_pipeline()) == 1
        assert tracker.get_pipeline()[0]["org_id"] == "org-test"
        assert other_tracker.get_pipeline()[0]["org_id"] == "org-other"


class TestUpdateStatus:
    def test_update_changes_status(self, tracker: GrantTracker) -> None:
        app_id = tracker.add_application("grant-upd", "Funder", "Grant")
        tracker.update_status(app_id, ApplicationStatus.SUBMITTED)
        all_apps = tracker.get_all()
        assert all_apps[0]["status"] == "SUBMITTED"

    def test_update_amount_awarded(self, tracker: GrantTracker) -> None:
        app_id = tracker.add_application("grant-award", "Funder", "Grant")
        tracker.update_status(
            app_id, ApplicationStatus.AWARDED, amount_awarded=25000.0
        )
        all_apps = tracker.get_all()
        assert all_apps[0]["amount_awarded"] == pytest.approx(25000.0)


class TestGetPipeline:
    def test_pipeline_excludes_decided_applications(self, tracker: GrantTracker) -> None:
        """AWARDED and DECLINED applications are not in the active pipeline."""
        active_id = tracker.add_application("grant-active", "F1", "G1")
        awarded_id = tracker.add_application("grant-done", "F2", "G2")
        tracker.update_status(awarded_id, ApplicationStatus.AWARDED)

        pipeline = tracker.get_pipeline()
        assert len(pipeline) == 1
        assert pipeline[0]["grant_id"] == "grant-active"

    def test_empty_pipeline(self, tracker: GrantTracker) -> None:
        assert tracker.get_pipeline() == []


class TestPipelineValue:
    def test_pipeline_value_zero_with_no_amounts(self, tracker: GrantTracker) -> None:
        tracker.add_application("grant-no-amt", "Funder", "Grant")
        tracker.update_status(1, ApplicationStatus.SUBMITTED)
        assert tracker.pipeline_value() == pytest.approx(0.0)

    def test_pipeline_value_sums_submitted_and_under_review(
        self, tracker: GrantTracker
    ) -> None:
        id1 = tracker.add_application(
            "grant-v1", "F1", "G1", amount_requested=10000.0
        )
        id2 = tracker.add_application(
            "grant-v2", "F2", "G2", amount_requested=20000.0
        )
        tracker.update_status(id1, ApplicationStatus.SUBMITTED)
        tracker.update_status(id2, ApplicationStatus.UNDER_REVIEW)
        assert tracker.pipeline_value() == pytest.approx(30000.0)

    def test_pipeline_value_excludes_awarded(self, tracker: GrantTracker) -> None:
        id1 = tracker.add_application(
            "grant-won", "F1", "G1", amount_requested=50000.0
        )
        tracker.update_status(id1, ApplicationStatus.AWARDED, amount_awarded=50000.0)
        assert tracker.pipeline_value() == pytest.approx(0.0)


class TestWinRate:
    def test_win_rate_zero_with_no_decisions(self, tracker: GrantTracker) -> None:
        result = tracker.win_rate()
        assert result["total_decided"] == 0
        assert result["win_rate"] == pytest.approx(0.0)

    def test_win_rate_calculation(self, tracker: GrantTracker) -> None:
        id1 = tracker.add_application("grant-w1", "F1", "G1")
        id2 = tracker.add_application("grant-w2", "F2", "G2")
        id3 = tracker.add_application("grant-w3", "F3", "G3")

        tracker.update_status(id1, ApplicationStatus.AWARDED, amount_awarded=10000.0)
        tracker.update_status(id2, ApplicationStatus.AWARDED, amount_awarded=5000.0)
        tracker.update_status(id3, ApplicationStatus.DECLINED)

        result = tracker.win_rate()
        assert result["total_decided"] == 3
        assert result["awarded"] == 2
        assert result["declined"] == 1
        assert result["win_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert result["total_awarded_value"] == pytest.approx(15000.0)
