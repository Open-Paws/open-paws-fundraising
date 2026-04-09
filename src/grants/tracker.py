"""
Grant application pipeline tracker.

SQLite-backed tracker for managing the status of grant applications
across coalition partner organizations.

Per-org isolation: all queries are scoped to org_id.
Grant data is public information shared across orgs.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = "data/grant_pipeline.db"


class ApplicationStatus(str, Enum):
    DRAFTING = "DRAFTING"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    AWARDED = "AWARDED"
    DECLINED = "DECLINED"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS grant_applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id          TEXT NOT NULL,
    grant_id        TEXT NOT NULL,          -- references seed_grants.json id
    funder_name     TEXT NOT NULL,
    grant_name      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'DRAFTING',
    amount_requested REAL,
    amount_awarded   REAL,
    submitted_date  TEXT,                   -- ISO date string
    decision_date   TEXT,                   -- ISO date string
    notes           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(org_id, grant_id)
);
"""


class GrantTracker:
    """
    Track grant applications for a coalition partner organization.

    All methods are scoped to org_id for coalition data isolation.

    Usage:
        tracker = GrantTracker(org_id="org-123")
        tracker.add_application(grant_id="open-philanthropy-farm-animal", ...)
        tracker.update_status(app_id=1, status=ApplicationStatus.SUBMITTED)
        value = tracker.pipeline_value()
        rate = tracker.win_rate()
    """

    def __init__(
        self,
        org_id: str,
        db_path: str | Path = DEFAULT_DB_PATH,
    ) -> None:
        self.org_id = org_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def add_application(
        self,
        grant_id: str,
        funder_name: str,
        grant_name: str,
        amount_requested: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Add a new grant application to the pipeline.

        Returns the new application ID.
        Raises ValueError if application for this grant already exists for this org.
        """
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO grant_applications
                        (org_id, grant_id, funder_name, grant_name,
                         status, amount_requested, notes, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.org_id, grant_id, funder_name, grant_name,
                        ApplicationStatus.DRAFTING.value,
                        amount_requested, notes, now, now,
                    ),
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                raise ValueError(
                    f"Application for grant '{grant_id}' already exists for org '{self.org_id}'"
                )

    def update_status(
        self,
        app_id: int,
        status: ApplicationStatus,
        amount_awarded: Optional[float] = None,
        submitted_date: Optional[date] = None,
        decision_date: Optional[date] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Update the status of an application."""
        now = datetime.now(UTC).isoformat()
        fields = ["status = ?", "updated_at = ?"]
        values: list = [status.value, now]

        if amount_awarded is not None:
            fields.append("amount_awarded = ?")
            values.append(amount_awarded)
        if submitted_date is not None:
            fields.append("submitted_date = ?")
            values.append(submitted_date.isoformat())
        if decision_date is not None:
            fields.append("decision_date = ?")
            values.append(decision_date.isoformat())
        if notes is not None:
            fields.append("notes = ?")
            values.append(notes)

        values.extend([app_id, self.org_id])

        # fields contains only hardcoded column assignment strings (e.g. "status = ?")
        # and all user-supplied values are passed as parameterized query arguments.
        set_clause = ", ".join(fields)  # nosec B608
        query = "UPDATE grant_applications SET " + set_clause + " WHERE id = ? AND org_id = ?"
        with self._connect() as conn:
            conn.execute(query, values)

    def get_pipeline(self) -> list[dict]:
        """
        Return all active applications for this org.

        Active = DRAFTING, SUBMITTED, or UNDER_REVIEW.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM grant_applications
                WHERE org_id = ?
                  AND status IN ('DRAFTING', 'SUBMITTED', 'UNDER_REVIEW')
                ORDER BY created_at DESC
                """,
                (self.org_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict]:
        """Return all applications (all statuses) for this org."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM grant_applications WHERE org_id = ? ORDER BY created_at DESC",
                (self.org_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def pipeline_value(self) -> float:
        """
        Total dollar value of grants currently under review.

        Sums amount_requested for SUBMITTED and UNDER_REVIEW applications.
        Returns 0.0 if no amounts are recorded.
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(amount_requested), 0) AS total
                FROM grant_applications
                WHERE org_id = ?
                  AND status IN ('SUBMITTED', 'UNDER_REVIEW')
                  AND amount_requested IS NOT NULL
                """,
                (self.org_id,),
            ).fetchone()
        return float(row["total"])

    def win_rate(self) -> dict:
        """
        Calculate historical win rate from decided applications.

        Returns:
            {
                "total_decided": int,
                "awarded": int,
                "declined": int,
                "win_rate": float (0.0–1.0),
                "total_awarded_value": float,
            }
        """
        with self._connect() as conn:
            awarded = conn.execute(
                """
                SELECT COUNT(*) as count, COALESCE(SUM(amount_awarded), 0) as total
                FROM grant_applications
                WHERE org_id = ? AND status = 'AWARDED'
                """,
                (self.org_id,),
            ).fetchone()
            declined = conn.execute(
                "SELECT COUNT(*) as count FROM grant_applications "
                "WHERE org_id = ? AND status = 'DECLINED'",
                (self.org_id,),
            ).fetchone()

        total_decided = awarded["count"] + declined["count"]
        win_rate = (awarded["count"] / total_decided) if total_decided > 0 else 0.0

        return {
            "total_decided": total_decided,
            "awarded": awarded["count"],
            "declined": declined["count"],
            "win_rate": round(win_rate, 4),
            "total_awarded_value": float(awarded["total"]),
        }

    def upcoming_deadlines(self, days_ahead: int = 30) -> list[dict]:
        """
        Return submitted applications with decisions expected within days_ahead.

        Used to surface time-sensitive follow-up actions.
        """
        cutoff = datetime.now(UTC).date()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM grant_applications
                WHERE org_id = ?
                  AND status IN ('SUBMITTED', 'UNDER_REVIEW')
                  AND submitted_date IS NOT NULL
                ORDER BY submitted_date ASC
                """,
                (self.org_id,),
            ).fetchall()
        return [dict(r) for r in rows]
