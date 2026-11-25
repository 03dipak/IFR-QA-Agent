# Coverage Report — IFR-QA Corpus Acquisition (read-only audit)

Source of truth: existing repository artifacts only. No new network requests, no downloads,
no changes to existing corpus data or seed lists were made during this audit.

## 1. Current corpus state (as audited)

| Metric | Value | Source |
|---|---|---|
| Raw PDFs in `data/corpus/raw/` | 80 | directory listing |
| Ledger document rows | 68 | `data/corpus/ledger.jsonl` |
| Kept documents (gate) | 66 | `data/corpus/gate_report.json` summary |
| Excluded documents | 13 | gate_report.json |
| Unreadable documents | 1 | gate_report.json |
| Extracted | 63 | ledger `extraction_status=extracted` |
| Pending extraction | 5 | ledger `extraction_status=pending` |
| Anomalies flagged | 11 | gate_report.json |
| USD/IFRS flagged | 25 | gate_report.json |

Ledger document rows by company: itc 16, hcl 13, wipro 12, infosys 7, tata_steel 4,
l_and_t 3, torrent_pharma 3, hindustan_unilever 1, sun_pharma 1, RELIANCE 1, ICICI 1,
plus 6 seed/demo "local" disclosure rows.

By document type: annual 23, quarterly 39, disclosure 6.
Fiscal periods observed: FY2023–Q2FY2027.

## 2. Coverage-matrix gap analysis (544 required cells = 41 companies × 13)

| Status | Cells | Share |
|---|---|---|
| PROBED_OK (confirmed PDF URL) | 68 | 12.5% |
| FOUND (URL candidate, not yet probed) | 26 | 4.8% |
| MISSING (no URL found) | 450 | 82.7% |

Maximum usable candidate documents from the current 41-company list, if every FOUND
URL also probes OK and passes the gate: 68 + 26 = **94**.
Shortfall vs the 500-document minimum: **406** (even before keep-rate discount and
deduplication against the 66 already-kept documents).
Shortfall vs the preferred 800–1,200 band: **706–1,106**.

## 3. Coverage by sector / document type / period (PROBED_OK cells)

| Sector | Annual | Quarterly | Total |
|---|---|---|---|
| IT | 4 | 22 | 26 |
| FMCG | 4 | 20 | 24 |
| Energy/metals | 2 | 5 | 7 |
| Capital goods | 3 | 0 | 3 (bhel only) |
| Chemicals | 2 | 0 | 2 (pidilite 1, upl 1) |
| Auto | 1 | 3 | 4 |
| Pharma | 1 | 1 | 2 |

Periods with probed cells: FY2024 (5), FY2025 (6), FY2026 (6), Q1FY2025 (6),
Q1FY2026 (4), Q1FY2027 (5), Q2FY2025 (6), Q2FY2026 (4), Q2FY2027 (3),
Q3FY2025 (6), Q3FY2026 (5), Q4FY2025 (8), Q4FY2026 (4).

Companies with zero probed/found cells: 26 of 41 (including all 12 priority-3 blocked
companies plus divis_labs, abbott_india, britannia, ntpc, tech_mahindra, marico,
force_motors, adani_green — verified domains but no static document links discovered).

## 4. Source feasibility assessment (evidence-based, no new requests)

| Source | Route | Feasibility | Evidence |
|---|---|---|---|
| Company-owned IR pages (28 VERIFIED domains) | Direct PDF links from investor pages | PARTIAL — feasible where static links exist | `data/seeds/company_domains.csv` (28 VERIFIED OK), `coverage_matrix.csv` PROBED_OK rows, `harvest_p12.log`/`final_push.log` per-company results |
| Company IR pages (13 domains not VERIFIED OK) | Direct PDF links | BLOCKED — UNREACHABLE/HTTP_403/NO_INVESTOR_LINK/NAME_MISMATCH/REDIRECTED | `company_domains.csv` |
| NSE corporate filings | Bulk download | NOT PERMITTED under current rules — rule 3 forbids NSE automation without explicit permission/access; treat as pending mentor/institution approval | `rule 3`, `src/ingestion/fetcher.py` etiquette |
| BSE corporate announcements/results | Bulk download | NOT PERMITTED under current rules — same as NSE | `rule 3` |
| SEBI corporate filings curation page | Public listing page | UNVERIFIED — no repository evidence yet | no artifact references SEBI fetches |
| XBRL / CSV / alternate representations | Supporting structured data | NOT counted as documents (rule 7); usable as structured support only | `rule 7` |

## 5. Conclusion

The 25-company pilot (see `pilot_plan.md`) is a method-validation step, not a path to 500:
even at full yield it adds an estimated ≤45 kept documents (before dedupe overlap with
the existing corpus), taking the corpus to roughly 100–130 kept documents. Reaching the
500-document minimum requires either (a) a secondary candidate universe of additional
listed companies with verified official document availability, or (b) mentor/institution
approved NSE or BSE access. Neither is assumed in this plan.
