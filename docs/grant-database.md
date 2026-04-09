# Grant Database

This document describes the grant database seeded in `src/grants/seed_grants.json`.

## Overview

25 real grant opportunities curated for animal advocacy organizations.
The database covers:

- Farmed animal welfare (corporate campaigns, investigations, policy)
- Companion animal welfare (shelters, spay-neuter, rescue)
- Alternative proteins and food systems
- Animal law and legal advocacy
- International funders (US, UK, Europe, global)
- Small grassroots grants ($500) to major foundation grants ($5M)

## Funders Included

| Funder | Focus | Range | Geography |
|--------|-------|-------|-----------|
| Open Philanthropy | Farmed animal welfare, movement building | $50K–$5M | Global |
| Wilks Foundation | Investigations, corporate campaigns | $100K–$2M | Global |
| Pew Charitable Trusts | Factory farm reform, food policy | $100K–$1M | US |
| ASPCA | Companion animal welfare, anti-cruelty | $10K–$100K | US |
| Humane Society Foundation | Animal protection programs | $5K–$50K | US |
| Maddie's Fund | Shelter medicine, no-kill programs | $25K–$1M | US |
| Animal Charity Evaluators | Evidence-based advocacy | $10K–$500K | Global |
| New Harvest | Cellular agriculture research | $20K–$300K | Global |
| Good Food Institute | Alternative protein research | $50K–$500K | Global |
| Mercy For Animals | Corporate campaigns | $10K–$150K | Global |
| Animal Legal Defense Fund | Animal law, ag-gag challenges | $5K–$100K | US |
| Faunalytics | Advocacy research | $5K–$50K | Global |
| Albert Schweitzer Foundation | Farmed animal welfare, vegan outreach | $5K–$50K | Global |
| World Animal Protection | Cross-species campaigns | $20K–$200K | Global |
| FOUR PAWS International | Animal welfare projects | $10K–$150K | Global |
| Global Animal Law Foundation | Animal law, legislation | $5K–$75K | Global |
| Vogt Foundation | Animal rights movement | $10K–$200K | US/Europe |
| Farm Sanctuary | Community advocacy | $1K–$15K | US |
| The Pollination Project | Grassroots projects | $500–$1.5K | Global |
| Marisla Foundation | Marine/aquatic animals | $25K–$150K | US |
| Animal Welfare Trust | Cross-species welfare | £5K–£50K | UK |
| Doris Day Animal Foundation | Companion animals | $1K–$10K | US |
| PetSmart Charities | Companion animal rescue | $5K–$75K | US/Canada |
| Humane Farming Association | Anti-factory-farming | $5K–$50K | US |
| Prevention of Animal Cruelty in Europe | EU policy | €10K–€100K | Europe |

## Maintenance

The grant database should be refreshed at minimum quarterly to catch new programs,
updated deadlines, and changed award amounts.

See GitHub Issue #4 for the automated refresh plan.

## Adding Grants

To add a new grant to the database, append an entry to `seed_grants.json` following this schema:

```json
{
  "id": "unique-slug",
  "funder": "Funder Organization Name",
  "grant_name": "Grant Program Name",
  "focus_areas": ["keyword1", "keyword2"],
  "amount_min": 10000,
  "amount_max": 100000,
  "deadline_pattern": "rolling | annual | quarterly | invitation",
  "geographic_restrictions": "global | US only | Europe only | UK only",
  "url": "https://...",
  "notes": "Key information about this funder's priorities and process."
}
```

Use lowercase, hyphenated slugs for `id` that include the funder name.

Focus areas should use movement terminology:
- "farmed animal welfare" not "livestock welfare"
- "factory farm" not "farm"
- "slaughterhouse" not "processing facility"
