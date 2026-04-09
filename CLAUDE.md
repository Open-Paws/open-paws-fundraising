# open-paws-fundraising — Agent Instructions

Fundraising intelligence platform for animal advocacy organizations.
Predicts donor churn, matches grants, drafts applications, forecasts revenue.
Available to all coalition partner orgs as a shared service.

## Every Session

Read the strategy repo before starting any task:

```bash
gh api repos/Open-Paws/open-paws-strategy/contents/priorities.md --jq '.content' | base64 -d
gh api repos/Open-Paws/open-paws-strategy/contents/closed-decisions.md --jq '.content' | base64 -d
```

## Architecture

- `src/donors/` — Donor churn prediction + re-engagement (DonorPulse CatBoost patterns)
- `src/grants/` — Grant matching (semantic search) + application drafting (LLM)
- `src/forecasting/` — Revenue forecasting (TimesFM time-series patterns)
- `src/impact/` — Impact measurement → donor reporting loop
- `src/api/` — FastAPI serving org-authenticated endpoints
- `dashboard/` — Streamlit fundraising intelligence dashboard

## Coalition Use

This is a shared service for coalition partner orgs.
Each org has isolated data (no cross-org donor visibility).
Grant database is shared — all orgs benefit from the same grant discovery.

## Privacy

- Donor data: never shared between orgs, never used for cross-org analysis
- Grant data: public information, shared freely
- Impact data: org controls what to share with supporters
- All donor identifiers are pseudonymous — no PII stored in behavioral tables

## Security

Three-adversary threat model applies:
1. State surveillance — activist identity data may be subpoena target
2. Industry infiltration — coalitions include orgs with varying risk profiles
3. AI model bias — grant matching LLMs may encode speciesist defaults

Per-org API key authentication. All org_id scoping enforced at query level.
Never log or retain donor PII.

## Domain Language

- "activist" or "supporter" not "user"
- "organization" or "coalition partner" not "client"
- "campaign" not "project"
- "movement" not "sector" or "industry"
- "farmed animal" not "livestock"
- "factory farm" not "farm" or "production facility"

## Task Routing

| You're doing... | Read... |
|-----------------|---------|
| Churn model training/inference | `src/donors/churn_predictor.py`, `src/donors/models.py` |
| Adding grants to database | `src/grants/seed_grants.json`, `docs/grant-database.md` |
| Grant matching logic | `src/grants/matcher.py` |
| Application drafting | `src/grants/drafter.py` |
| Revenue forecasting | `src/forecasting/revenue_forecast.py` |
| Impact reporting | `src/impact/reporting.py` |
| API endpoints | `src/api/server.py` |
| Dashboard | `dashboard/app.py` |

## Running

```bash
# API server
uvicorn src.api.server:app --reload

# Dashboard
streamlit run dashboard/app.py

# Grant matching CLI
python -m src.grants.matcher --org-profile profile.json

# Train churn model
python -m src.donors.churn_predictor train --input data/donor_events.csv --output models/

# Revenue forecast
python -m src.forecasting.revenue_forecast --input data/monthly_donations.csv
```

## Quality Gates

```bash
pip install "git+https://github.com/Open-Paws/desloppify.git#egg=desloppify[full]"
desloppify scan --path .
desloppify next
```

Minimum score: 85 before any PR merge.

## Open Issues

See GitHub Issues for current work items. Key ones:
- #1: Integrate CatBoost for production-grade churn model
- #2: Add TimesFM 2.5 for higher-accuracy revenue forecasting
- #3: Build semantic grant matching with sentence-transformers
- #4: Add monthly grant database refresh automation
- #5: Build supporter re-engagement campaign automation
- #6: Add Bringing-money-into-the-movement strategic frameworks

Speciesist language scan:
```bash
semgrep --config semgrep-no-animal-violence.yaml .
```

All PRs must pass CI before merge.