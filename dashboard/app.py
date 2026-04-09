"""
Streamlit fundraising intelligence dashboard.

Tabs:
    1. Donor Health — churn risk breakdown, donors needing attention
    2. Grant Pipeline — applications, pipeline value, win rate, deadlines
    3. Revenue Forecast — 12-month projection with confidence bands
    4. Impact Metrics — campaign outcomes mapped to donor segments

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Open Paws Fundraising Intelligence",
    page_icon="🐾",
    layout="wide",
)

st.title("Open Paws Fundraising Intelligence")
st.caption(
    "Coalition fundraising platform for animal advocacy organizations. "
    "Data is isolated per organization — no cross-org visibility."
)

# ── Org selection ─────────────────────────────────────────────────────────────
org_id = st.sidebar.text_input(
    "Organization ID",
    value="demo-org",
    help="Your coalition partner org identifier",
)
st.sidebar.caption("All data is scoped to your org ID.")

tab_donors, tab_grants, tab_forecast, tab_impact = st.tabs([
    "Donor Health",
    "Grant Pipeline",
    "Revenue Forecast",
    "Impact Metrics",
])

# ── Tab 1: Donor Health ───────────────────────────────────────────────────────
with tab_donors:
    st.header("Donor Health")
    st.write(
        "Upload supporter behavioral data to predict churn risk. "
        "No PII required — pseudonymous IDs and behavioral signals only."
    )

    uploaded = st.file_uploader(
        "Upload donor behavioral data (JSON)",
        type=["json"],
        help=(
            "JSON array of donor records with fields: donor_id, segment, "
            "total_donations_12mo, donation_count_12mo, days_since_last_donation, etc."
        ),
    )

    if uploaded:
        try:
            data = json.load(uploaded)
            if not isinstance(data, list):
                st.error("Expected a JSON array of donor records.")
            else:
                from src.donors.churn_predictor import ChurnPredictor
                from src.donors.models import Donor, DonorSegment
                from src.donors.segments import priority_order

                predictor = ChurnPredictor()
                try:
                    predictor.load("models/")
                    st.success("Loaded trained churn model.")
                except FileNotFoundError:
                    st.info("No trained model found — using rules-based scoring.")

                donors = []
                for rec in data:
                    try:
                        seg = DonorSegment(rec.get("segment", "GRASSROOTS"))
                    except ValueError:
                        seg = DonorSegment.GRASSROOTS
                    donors.append(Donor(
                        donor_id=rec.get("donor_id", "unknown"),
                        org_id=org_id,
                        segment=seg,
                        total_donations_12mo=float(rec.get("total_donations_12mo", 0)),
                        donation_count_12mo=int(rec.get("donation_count_12mo", 0)),
                        days_since_last_donation=int(rec.get("days_since_last_donation", 365)),
                        average_gift_size=float(rec.get("average_gift_size", 0)),
                        campaigns_engaged=int(rec.get("campaigns_engaged", 0)),
                        preferred_campaign_topic=rec.get("preferred_campaign_topic"),
                        open_rate_last_90d=float(rec.get("open_rate_last_90d", 0)),
                        click_rate_last_90d=float(rec.get("click_rate_last_90d", 0)),
                        is_recurring_donor=bool(rec.get("is_recurring_donor", False)),
                    ))

                scored = predictor.predict(donors)
                prioritized = priority_order(scored)

                # Summary metrics
                high = sum(1 for d in scored if d.churn_risk and d.churn_risk.value in ("HIGH", "LAPSED"))
                medium = sum(1 for d in scored if d.churn_risk and d.churn_risk.value == "MEDIUM")
                low = sum(1 for d in scored if d.churn_risk and d.churn_risk.value == "LOW")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Supporters", len(scored))
                col2.metric("High Risk", high, delta=None, delta_color="inverse")
                col3.metric("Medium Risk", medium)
                col4.metric("Low Risk", low)

                # Risk distribution
                import pandas as pd
                risk_df = pd.DataFrame({
                    "Risk Level": ["High/Lapsed", "Medium", "Low"],
                    "Count": [high, medium, low],
                })
                st.bar_chart(risk_df.set_index("Risk Level"))

                # Priority table
                st.subheader("Supporters Needing Attention")
                attention = [
                    d for d in prioritized
                    if d.churn_risk and d.churn_risk.value in ("HIGH", "LAPSED", "MEDIUM")
                ][:50]

                if attention:
                    rows = []
                    for d in attention:
                        rows.append({
                            "Donor ID": d.donor_id,
                            "Risk": d.churn_risk.value if d.churn_risk else "—",
                            "Probability": f"{d.churn_probability:.0%}" if d.churn_probability else "—",
                            "Days Since Gift": d.days_since_last_donation,
                            "12mo Total": f"${d.total_donations_12mo:,.0f}",
                            "Segment": d.segment.value,
                            "Recommended Action": d.recommended_action or "—",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.success("No high-priority supporters identified.")

        except Exception as exc:
            st.error(f"Error processing donor data: {exc}")
    else:
        st.info(
            "Upload a JSON file to score your supporter base. "
            "See the API at `POST /donors/analyze` for programmatic access."
        )

        # Show sample format
        with st.expander("Sample donor record format"):
            st.json([{
                "donor_id": "pseudonymous-id-001",
                "segment": "MID_LEVEL",
                "total_donations_12mo": 350.0,
                "donation_count_12mo": 4,
                "days_since_last_donation": 95,
                "average_gift_size": 87.50,
                "campaigns_engaged": 3,
                "preferred_campaign_topic": "farmed animal welfare",
                "open_rate_last_90d": 0.42,
                "click_rate_last_90d": 0.18,
                "is_recurring_donor": False,
            }])


# ── Tab 2: Grant Pipeline ─────────────────────────────────────────────────────
with tab_grants:
    st.header("Grant Pipeline")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Grant Matching")
        mission_input = st.text_area(
            "Organization mission",
            placeholder="e.g. We run undercover investigations of factory farms and corporate campaigns...",
            height=100,
        )
        programs_input = st.text_area(
            "Programs (one per line)",
            placeholder="Cage-free corporate campaigns\nFarmed animal investigations\nAnimal welfare policy",
            height=80,
        )
        geography_input = st.selectbox("Geography", ["US", "global", "Europe", "UK"])
        budget_input = st.number_input("Annual budget ($)", min_value=0, value=500000, step=50000)

        if st.button("Find Matching Grants"):
            if not mission_input:
                st.warning("Enter your organization's mission to find grants.")
            else:
                from src.grants.matcher import GrantMatcher

                org_profile = {
                    "org_id": org_id,
                    "org_name": "Your Organization",
                    "mission": mission_input,
                    "programs": [p.strip() for p in programs_input.split("\n") if p.strip()],
                    "geography": geography_input,
                    "annual_budget": budget_input,
                }
                matcher = GrantMatcher()
                matches = matcher.top_matches(org_profile, n=10)

                if matches:
                    for match in matches:
                        g = match["grant"]
                        score = match["match_score"]
                        with st.expander(
                            f"**{g['funder']}** — {g['grant_name']} "
                            f"(score: {score:.0%})"
                        ):
                            st.write(match["rationale"])
                            amt_min = g.get("amount_min", 0)
                            amt_max = g.get("amount_max", 0)
                            if amt_min and amt_max:
                                st.write(f"**Amount:** ${amt_min:,} – ${amt_max:,}")
                            st.write(f"**Deadline:** {g.get('deadline_pattern', 'varies')}")
                            st.write(f"**Geography:** {g.get('geographic_restrictions', 'global')}")
                            if g.get("url"):
                                st.write(f"**URL:** {g['url']}")
                else:
                    st.info("No strong matches found. Try broadening your mission description.")

    with col_left:
        st.subheader("Application Tracker")

        from src.grants.tracker import ApplicationStatus, GrantTracker

        tracker = GrantTracker(org_id=org_id)
        pipeline = tracker.get_pipeline()
        pipeline_value = tracker.pipeline_value()
        win_rate_data = tracker.win_rate()

        c1, c2, c3 = st.columns(3)
        c1.metric("Pipeline Value", f"${pipeline_value:,.0f}")
        c2.metric("Win Rate", f"{win_rate_data['win_rate']:.0%}")
        c3.metric("Total Awarded", f"${win_rate_data['total_awarded_value']:,.0f}")

        if pipeline:
            import pandas as pd
            pipeline_df = pd.DataFrame(pipeline)[
                ["funder_name", "grant_name", "status", "amount_requested", "created_at"]
            ]
            pipeline_df.columns = ["Funder", "Grant", "Status", "Amount Requested", "Added"]
            st.dataframe(pipeline_df, use_container_width=True)
        else:
            st.info(
                "No active applications. Use the grant matching tool to find opportunities, "
                "then add them to your pipeline via POST /grants/pipeline."
            )

        # Add application form
        with st.expander("Add application to pipeline"):
            import json as _json
            seed_path = Path(__file__).parent.parent / "src" / "grants" / "seed_grants.json"
            with open(seed_path, encoding="utf-8") as f:
                all_grants = _json.load(f)

            grant_options = {f"{g['funder']} — {g['grant_name']}": g for g in all_grants}
            selected = st.selectbox("Select grant", list(grant_options.keys()))
            amount = st.number_input("Amount to request ($)", min_value=0, value=50000)
            add_notes = st.text_input("Notes")

            if st.button("Add to Pipeline"):
                g = grant_options[selected]
                try:
                    app_id = tracker.add_application(
                        grant_id=g["id"],
                        funder_name=g["funder"],
                        grant_name=g["grant_name"],
                        amount_requested=float(amount) if amount else None,
                        notes=add_notes or None,
                    )
                    st.success(f"Added to pipeline (application #{app_id})")
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))


# ── Tab 3: Revenue Forecast ───────────────────────────────────────────────────
with tab_forecast:
    st.header("Revenue Forecast")
    st.write("Enter historical monthly donation totals to generate a 12-month projection.")

    default_history = "12000, 9500, 11000, 15000, 8000, 13000, 10500, 9000, 11500, 18000, 22000, 14000"
    history_input = st.text_area(
        "Monthly donation totals (comma-separated, oldest first)",
        value=default_history,
        help="Minimum 6 months recommended. 18+ months enables seasonality detection.",
    )

    horizon = st.slider("Forecast horizon (months)", min_value=3, max_value=24, value=12)

    if st.button("Generate Forecast"):
        try:
            monthly_totals = [float(x.strip()) for x in history_input.split(",") if x.strip()]
            if len(monthly_totals) < 3:
                st.error("Need at least 3 months of history.")
            else:
                from src.forecasting.revenue_forecast import RevenueForecaster

                forecaster = RevenueForecaster()
                result = forecaster.forecast(monthly_totals, horizon=horizon)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Projected Annual Total",
                    f"${result['annual_total_projection']:,.0f}",
                )
                col2.metric(
                    "Lower Bound (80%)",
                    f"${result['annual_total_lower']:,.0f}",
                )
                col3.metric(
                    "Upper Bound (80%)",
                    f"${result['annual_total_upper']:,.0f}",
                )

                # Chart
                import pandas as pd
                history_df = pd.DataFrame({
                    "Month": list(range(1, len(monthly_totals) + 1)),
                    "Actual": monthly_totals,
                    "Type": ["Historical"] * len(monthly_totals),
                })
                forecast_start = len(monthly_totals) + 1
                forecast_df = pd.DataFrame({
                    "Month": list(range(forecast_start, forecast_start + horizon)),
                    "Actual": result["projections"],
                    "Type": ["Forecast"] * horizon,
                    "Lower": [ci["lower"] for ci in result["confidence_intervals"]],
                    "Upper": [ci["upper"] for ci in result["confidence_intervals"]],
                })

                st.subheader("Monthly Projections")
                chart_data = pd.concat([
                    history_df[["Month", "Actual"]].rename(columns={"Actual": "Actual/Forecast"}),
                    forecast_df[["Month", "Actual"]].rename(columns={"Actual": "Actual/Forecast"}),
                ])
                st.line_chart(chart_data.set_index("Month"))

                # Trend and notes
                trend_icon = {"up": "↑", "down": "↓", "flat": "→"}.get(result["trend_direction"], "→")
                st.caption(f"Trend: {trend_icon} {result['trend_direction'].upper()}")
                st.info(result["notes"])

                if result["seasonal_pattern_detected"]:
                    st.success("Year-end giving spike pattern detected in your history.")

        except ValueError as exc:
            st.error(f"Error: {exc}")
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")


# ── Tab 4: Impact Metrics ─────────────────────────────────────────────────────
with tab_impact:
    st.header("Impact Metrics")
    st.write(
        "Record campaign outcomes to generate donor-facing impact statements. "
        "Impact reports are sent to HIGH/LAPSED risk donors before re-engagement sequences."
    )

    st.subheader("Add Campaign Outcome")
    with st.form("add_outcome"):
        campaign_name = st.text_input("Campaign name", placeholder="Cage-Free Corporate Campaign 2026")
        outcome_type = st.selectbox(
            "Outcome type",
            ["animals_helped", "policy_win", "corporate_commitment", "direct_action"],
        )
        description = st.text_area(
            "Description",
            placeholder="Major food retailer committed to 100% cage-free eggs by 2026",
        )
        metric_value = st.number_input("Metric value (optional)", min_value=0, value=0)
        metric_unit = st.text_input("Metric unit (optional)", placeholder="farmed animals")
        submitted = st.form_submit_button("Add Outcome")

    if "outcomes" not in st.session_state:
        st.session_state.outcomes = []

    if submitted and campaign_name and description:
        from src.impact.reporting import CampaignOutcome
        from datetime import date

        outcome = CampaignOutcome(
            campaign_name=campaign_name,
            outcome_type=outcome_type,
            description=description,
            metric_value=float(metric_value) if metric_value > 0 else None,
            metric_unit=metric_unit or None,
            date_achieved=date.today(),
        )
        st.session_state.outcomes.append(outcome)
        st.success(f"Added: {campaign_name}")

    if st.session_state.outcomes:
        st.subheader("Impact Summary")
        from src.impact.reporting import build_impact_report, format_impact_report_email
        from datetime import date

        report = build_impact_report(
            donor_id="sample-supporter",
            org_id=org_id,
            org_name="Your Organization",
            total_donated=1200.0,
            campaign_outcomes=st.session_state.outcomes,
            period_start=date(date.today().year, 1, 1),
            period_end=date.today(),
        )

        st.metric("Headline", report.headline)

        st.subheader("Impact Statements")
        for stmt in report.impact_statements:
            st.write(f"- {stmt}")

        email = format_impact_report_email(report)
        with st.expander("Preview email"):
            st.write(f"**Subject:** {email['subject_line']}")
            st.write(f"**Preview:** {email['preview_text']}")
            st.markdown(email["body_markdown"])

        if st.button("Clear outcomes"):
            st.session_state.outcomes = []
            st.rerun()
    else:
        st.info("Add campaign outcomes above to generate impact statements.")
