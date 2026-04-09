# open-paws-fundraising

Fundraising intelligence platform for animal advocacy organizations. Shared service for coalition partners.

## What it does

- **Donor churn prediction** — CatBoost (or sklearn fallback) model scoring supporters by churn risk using behavioral signals only (no PII stored in feature table)
- **Grant matching** — Keyword + optional semantic matching of org profiles against 20+ real animal welfare grant opportunities
- **Application drafting** — LLM-powered LOI, executive summary, and narrative outline generation
- **Application tracking** — SQLite pipeline tracker with win rate analytics
- **Revenue forecasting** — 12-month projection with confidence intervals using exponential smoothing
- **Impact reporting** — Campaign outcome → donor-facing impact statement generation

## Architecture

```
src/
├── donors/         # Churn prediction + re-engagement
├── grants/         # Matching, drafting, tracking
├── forecasting/    # Revenue time-series
├── impact/         # Outcome → reporting loop
└── api/            # FastAPI (org-authenticated)
dashboard/          # Streamlit intelligence dashboard
docs/               # Grant database documentation
```

## Coalition Use

Each coalition partner org operates in full isolation — no cross-org donor data visibility. The grant database is shared, so all orgs benefit from the same grant discovery and seeding work.

## Quick Start

```bash
pip install -e ".[all]"
cp .env.example .env  # add your ANTHROPIC_API_KEY

# Run API
uvicorn src.api.server:app --reload

# Run dashboard
streamlit run dashboard/app.py
```

## Privacy

Donor behavioral features contain zero PII. All donor identifiers are pseudonymous `donor_id` strings assigned by the coalition partner's CRM. This platform never receives or stores names, email addresses, or contact information.

## Security

See `CLAUDE.md` for the three-adversary threat model applied to this codebase.

## Open Issues

See [GitHub Issues](https://github.com/Open-Paws/open-paws-fundraising/issues) for the current backlog.
