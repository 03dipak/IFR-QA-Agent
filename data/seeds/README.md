# Discovery seed inputs (drafts, owner review pending)

These files feed the Phase 1 bulk-acquisition discovery step
(`src/ingestion/discovery.py`).

## Files

| File | Purpose | Owner / review |
| --- | --- | --- |
| `companies_pilot.csv` | Companies file for the first real discovery test (2 companies). | Project owner + team; reviewed with mentor before discovery run. |
| `companies_v1.csv` | Draft companies file for the v1 tier (15-25 companies; v1 tier is the ~100-150 document batch). | Project owner + team; reviewed with mentor before use. |
| `discovery_pilot.json` | Discovery config (fiscal years / doc types / listing urls) for the pilot. | Project owner + team; reviewed with mentor. |
| `discovery_v1.json` | Discovery config for the v1 draft batch. | Project owner + team; reviewed with mentor. |
| `README.md` | This file. | Project owner + team; reviewed with mentor. |

## Format (as consumed by discovery.py)

Companies CSV columns (required by the validator):

- `company_name`, `scrip_code`, `isin`, `company_id`
- `verification_status` (`VERIFIED` | `NEEDS_VERIFICATION`)
- `source_url`, `checked_on`, `notes`

Discovery config JSON keys: `fiscal_years` (list, e.g. `["FY2024"]`),
`doc_types` (list, e.g. `["annual", "quarterly", "disclosure"]`),
`listing_urls` (map of scrip code -> listing page URL).

`discovery.py` accepts extra CSV columns (via `csv.DictReader`) and ignores
them, so no sidecar file is needed for the added columns.

## Sourcing rules followed

- BSE scrip code, ISIN and fiscal year-end values were **not** taken from
  memory; where they could not be verified in this step they are left EMPTY and
  `verification_status` is set to `NEEDS_VERIFICATION`.
- Never guessed values: a wrong BSE scrip/ISIN is not acceptable.

## Commands

Validate a companies file without network access:

```
uv run -m src.ingestion.discovery --companies <companies.csv> --config <discovery.json> --validate
```

Discover filings for a companies file (refuses when the file contains
`NEEDS_VERIFICATION` rows unless `--allow-needs-verification` is passed):

```
uv run -m src.ingestion.discovery --companies <companies.csv> --config <discovery.json> --out <candidates.csv>
```
