"""
FastAPI server for the fundraising intelligence platform.

All endpoints require X-API-Key header authenticated against FUNDRAISING_API_KEYS.
No cross-org data access is possible — all queries are scoped to org_id.

Endpoints:
    GET  /donors/at-risk          — HIGH/LAPSED donors needing attention
    POST /donors/analyze          — Run churn prediction on uploaded donor data
    GET  /grants/match            — Match grants to org profile
    POST /grants/draft            — Generate grant application draft
    GET  /grants/pipeline         — Current application pipeline
    POST /grants/pipeline         — Add application to pipeline
    PATCH /grants/pipeline/{id}   — Update application status
    GET  /forecast/revenue        — 12-month revenue projection
    GET  /health                  — Health check
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Format: "key1:org_id_1,key2:org_id_2"
FUNDRAISING_API_KEYS: dict[str, str] = {
    k: v for k, v in (
        entry.split(":", 1) for entry in
        os.environ.get("FUNDRAISING_API_KEYS", "").split(",") if ":" in entry
    )
}

app = FastAPI(
    title="Open Paws Fundraising Intelligence API",
    description=(
        "Fundraising intelligence platform for animal advocacy organizations. "
        "Predicts donor churn, matches grants, drafts applications, forecasts revenue."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten per environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Org authentication ────────────────────────────────────────────────────────

def require_api_key(x_api_key: str = Header(None)) -> str:
    """
    Validate X-API-Key header and return the mapped org_id.

    Keys are configured via FUNDRAISING_API_KEYS env var (format: key1:org_id_1,key2:org_id_2).
    All endpoints are scoped to the resolved org_id — no cross-org data access is possible.
    """
    if not x_api_key or x_api_key not in FUNDRAISING_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return FUNDRAISING_API_KEYS[x_api_key]  # returns org_id


def require_org(org_id: str = Depends(require_api_key)) -> str:
    return org_id


# ── Request/response models ───────────────────────────────────────────────────

class DonorRecord(BaseModel):
    donor_id: str
    segment: str = "GRASSROOTS"
    total_donations_12mo: float = 0.0
    donation_count_12mo: int = 0
    days_since_last_donation: int = 365
    average_gift_size: float = 0.0
    campaigns_engaged: int = 0
    preferred_campaign_topic: Optional[str] = None
    open_rate_last_90d: float = 0.0
    click_rate_last_90d: float = 0.0
    is_recurring_donor: bool = False


class DonorAnalyzeRequest(BaseModel):
    donors: list[DonorRecord] = Field(..., min_length=1, max_length=10000)


class OrgProfile(BaseModel):
    org_name: str
    mission: str
    programs: list[str] = []
    geography: str = "US"
    annual_budget: Optional[int] = None
    previous_funders: list[str] = []


class GrantMatchRequest(BaseModel):
    org_profile: OrgProfile
    min_score: float = 0.1
    top_n: int = 10


class GrantDraftRequest(BaseModel):
    grant_id: str
    org_profile: OrgProfile
    impact_data: Optional[dict] = None


class PipelineAddRequest(BaseModel):
    grant_id: str
    funder_name: str
    grant_name: str
    amount_requested: Optional[float] = None
    notes: Optional[str] = None


class PipelineUpdateRequest(BaseModel):
    status: str
    amount_awarded: Optional[float] = None
    notes: Optional[str] = None


class ForecastRequest(BaseModel):
    monthly_totals: list[float] = Field(..., min_length=3, max_length=120)
    horizon: int = Field(default=12, ge=1, le=24)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health_check() -> dict:
    return {"status": "ok", "service": "open-paws-fundraising"}


# ── Donors ────────────────────────────────────────────────────────────────────

@app.post("/donors/analyze", tags=["donors"])
def analyze_donors(
    request: DonorAnalyzeRequest,
    org_id: str = Depends(require_org),
) -> dict:
    """
    Run churn prediction on a batch of donor records.

    Returns each donor with churn_risk, churn_probability, and recommended_action.
    Falls back to rules-based scoring if no trained model is loaded.
    """
    from src.donors.churn_predictor import ChurnPredictor
    from src.donors.models import Donor, DonorSegment

    predictor = ChurnPredictor()

    # Attempt to load pre-trained model
    try:
        predictor.load("models/")
    except FileNotFoundError:
        logger.info("No trained model found; using rules-based fallback for org %s", org_id)

    donors = []
    for rec in request.donors:
        try:
            segment = DonorSegment(rec.segment)
        except ValueError:
            segment = DonorSegment.GRASSROOTS

        donors.append(Donor(
            donor_id=rec.donor_id,
            org_id=org_id,
            segment=segment,
            total_donations_12mo=rec.total_donations_12mo,
            donation_count_12mo=rec.donation_count_12mo,
            days_since_last_donation=rec.days_since_last_donation,
            average_gift_size=rec.average_gift_size,
            campaigns_engaged=rec.campaigns_engaged,
            preferred_campaign_topic=rec.preferred_campaign_topic,
            open_rate_last_90d=rec.open_rate_last_90d,
            click_rate_last_90d=rec.click_rate_last_90d,
            is_recurring_donor=rec.is_recurring_donor,
        ))

    scored = predictor.predict(donors)

    return {
        "org_id": org_id,
        "total_donors": len(scored),
        "donors": [
            {
                "donor_id": d.donor_id,
                "churn_risk": d.churn_risk.value if d.churn_risk else None,
                "churn_probability": d.churn_probability,
                "recommended_action": d.recommended_action,
                "segment": d.segment.value,
            }
            for d in scored
        ],
        "summary": {
            "high_risk": sum(1 for d in scored if d.churn_risk and d.churn_risk.value in ("HIGH", "LAPSED")),
            "medium_risk": sum(1 for d in scored if d.churn_risk and d.churn_risk.value == "MEDIUM"),
            "low_risk": sum(1 for d in scored if d.churn_risk and d.churn_risk.value == "LOW"),
        },
    }


@app.get("/donors/at-risk", tags=["donors"])
def get_at_risk_donors(
    limit: int = 50,
    org_id: str = Depends(require_org),
) -> dict:
    """
    Return a placeholder response. In production, this reads from your CRM
    integration and runs churn scoring on the stored donor cohort.

    POST /donors/analyze with your donor data to score a batch immediately.
    """
    return {
        "org_id": org_id,
        "message": (
            "POST to /donors/analyze with your donor behavioral data to score donors. "
            "This endpoint will return cached at-risk results once a CRM integration is configured."
        ),
        "next_step": "POST /donors/analyze",
    }


# ── Grants ────────────────────────────────────────────────────────────────────

@app.post("/grants/match", tags=["grants"])
def match_grants(
    request: GrantMatchRequest,
    org_id: str = Depends(require_org),
) -> dict:
    """
    Match the org profile against the grant database.

    Returns ranked list of grant opportunities with match scores and rationale.
    """
    from src.grants.matcher import GrantMatcher

    profile_dict = request.org_profile.model_dump()
    profile_dict["org_id"] = org_id

    matcher = GrantMatcher()
    matches = matcher.top_matches(
        org_profile=profile_dict,
        n=request.top_n,
        min_score=request.min_score,
    )

    return {
        "org_id": org_id,
        "total_matches": len(matches),
        "matches": matches,
    }


@app.post("/grants/draft", tags=["grants"])
def draft_grant_application(
    request: GrantDraftRequest,
    org_id: str = Depends(require_org),
) -> dict:
    """
    Generate a grant application draft for a matched grant.

    Returns LOI, executive summary, program narrative outline, and budget narrative.
    Requires OPEN_PAWS_API_KEY environment variable.
    """
    import json
    from pathlib import Path
    from src.grants.drafter import draft_application

    # Load grant from seed data
    seed_path = Path(__file__).parent.parent / "grants" / "seed_grants.json"
    with open(seed_path, encoding="utf-8") as f:
        all_grants = json.load(f)

    grant = next((g for g in all_grants if g["id"] == request.grant_id), None)
    if grant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Grant '{request.grant_id}' not found in grant database.",
        )

    org_profile = request.org_profile.model_dump()
    org_profile["org_id"] = org_id

    draft = draft_application(
        grant=grant,
        org_profile=org_profile,
        impact_data=request.impact_data,
    )

    return {"org_id": org_id, "draft": draft}


@app.get("/grants/pipeline", tags=["grants"])
def get_grant_pipeline(org_id: str = Depends(require_org)) -> dict:
    """Return the current grant application pipeline for this org."""
    from src.grants.tracker import GrantTracker

    tracker = GrantTracker(org_id=org_id)
    pipeline = tracker.get_pipeline()
    pipeline_value = tracker.pipeline_value()
    win_rate = tracker.win_rate()

    return {
        "org_id": org_id,
        "pipeline": pipeline,
        "pipeline_value": pipeline_value,
        "win_rate": win_rate,
    }


@app.post("/grants/pipeline", tags=["grants"])
def add_to_pipeline(
    request: PipelineAddRequest,
    org_id: str = Depends(require_org),
) -> dict:
    """Add a grant application to the pipeline tracker."""
    from src.grants.tracker import GrantTracker

    tracker = GrantTracker(org_id=org_id)
    try:
        app_id = tracker.add_application(
            grant_id=request.grant_id,
            funder_name=request.funder_name,
            grant_name=request.grant_name,
            amount_requested=request.amount_requested,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return {"org_id": org_id, "application_id": app_id, "status": "DRAFTING"}


@app.patch("/grants/pipeline/{application_id}", tags=["grants"])
def update_pipeline_status(
    application_id: int,
    request: PipelineUpdateRequest,
    org_id: str = Depends(require_org),
) -> dict:
    """Update the status of a grant application."""
    from src.grants.tracker import ApplicationStatus, GrantTracker

    try:
        new_status = ApplicationStatus(request.status)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status '{request.status}'. Valid: {[s.value for s in ApplicationStatus]}",
        )

    tracker = GrantTracker(org_id=org_id)
    tracker.update_status(
        app_id=application_id,
        status=new_status,
        amount_awarded=request.amount_awarded,
        notes=request.notes,
    )

    return {"org_id": org_id, "application_id": application_id, "status": new_status.value}


# ── Forecasting ───────────────────────────────────────────────────────────────

@app.post("/forecast/revenue", tags=["forecasting"])
def revenue_forecast(
    request: ForecastRequest,
    org_id: str = Depends(require_org),
) -> dict:
    """
    Generate a 12-month revenue forecast from historical monthly donation totals.

    Input: list of monthly totals (oldest first, minimum 3 months).
    Returns projections with confidence intervals, trend direction, and notes.
    """
    from src.forecasting.revenue_forecast import RevenueForecaster

    forecaster = RevenueForecaster()
    try:
        result = forecaster.forecast(
            monthly_totals=request.monthly_totals,
            horizon=request.horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    return {"org_id": org_id, **result}
