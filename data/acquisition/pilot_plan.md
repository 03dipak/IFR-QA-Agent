# Pilot Plan — 25-Company Acquisition Pilot (Phase 0: proposal only)

**Status:** proposal. No downloads, no corpus changes, no modifications to existing
acquisition records, seed lists, or production code are made by this document.

## 1. Objective

Validate the end-to-end acquisition method (discover → probe → download → gate → extract
→ dedupe → register) on a small, sector-balanced set before requesting broader expansion
or NSE/BSE access. The pilot alone is NOT expected to reach the 500-document minimum
(see coverage_report.md: max 94 candidate URLs across the pilot, ≈45 estimated kept
before dedupe overlap).

## 2. Pilot universe — 25 companies, all seven sectors

See `data/acquisition/company_universe.csv`.

| Sector | Companies |
|---|---|
| IT | infosys, wipro, hcl, tech_mahindra, coforge |
| FMCG | itc, britannia, dabur, marico |
| Energy/metals | tata_steel, ntpc, jsw_steel, hindalco, adani_green |
| Capital goods | l_and_t, bhel |
| Chemicals | pidilite, upl |
| Auto | tvs_motor, eicher_motors, ashok_leyland, force_motors |
| Pharma | divis_labs, abbott_india, torrent_pharma |

Selection rule: verified or document-evidenced official domain from
`data/seeds/company_domains.csv` / `coverage_matrix.csv`, preference for companies with
existing probed URLs or ledger documents (itc, wipro, infosys, hcl, tata_steel,
l_and_t, torrent_pharma) so the pipeline can be validated end-to-end on known-good
sources. ISIN and BSE scrip code are NOT verified for any pilot company (marked
NOT_VERIFIED in company_universe.csv) and must be resolved before exchange-level
attribution is claimed.

Abbott India has a December year-end — confirm acceptance before including its periods.

## 3. Access-method assessment per source (read-only evidence)

| Source | Method | Permission status | Basis |
|---|---|---|---|
| Company IR pages | Direct PDF attachment URLs from verified investor pages | Permitted (public, no login observed; honest UA IFR-QA-Agent/0.1, 1 req/s, robots.txt respected by fetcher) | 28 VERIFIED domains in company_domains.csv; PROBED_OK rows in coverage_matrix.csv |
| NSE corporate filings | Not attempted | BLOCKED — rule 3: no NSE automation without explicit permission/access | task rule 3 |
| BSE corporate announcements/results | Not attempted | BLOCKED — rule 3 | task rule 3 |
| SEBI corporate filings | Not attempted | UNVERIFIED — no repository evidence | no prior SEBI fetch artifacts |
| XBRL/CSV alternates | Not counted as documents | Supporting structured data only (rule 7) | task rule 7 |

## 4. Phases and gates

**Phase 0 — now (this proposal):** repository audit, source feasibility, pilot company
universe, master-ledger schema, this plan. Approval gate: user sign-off on pilot list,
access methods, and rate limits.

**Phase 1 — controlled acquisition (after approval):** run the existing pipeline
(`uv run -m src.ingestion.pipeline bulk --candidates <pilot candidates.csv>
--storage-dir data/corpus/raw`) against pilot candidate URLs only. Constraints:
global 1 req/s (fetcher default), per-domain throttle, exponential backoff capped at
60 s, no aggressive retry of blocked access (403/404/451 treated permanent), robots.txt
honoured, no browser spoofing, no NSE/BSE JSON endpoints. Record every attempt and
outcome; quarantine failures row-by-row (pipeline policy: skip-and-report).

**Phase 2 — validation:** gate_report + reconcile on the new raw PDFs; apply KEPT rules
(official provenance, downloaded, company+period identified, readable, checksummed,
non-duplicate, extracted except documented short filings); deduplicate against existing
66 kept documents by (company, period, consolidated/standalone, reporting basis).

**Phase 3 — expansion decision:** if measured kept-yield per company supports it, propose
a secondary candidate universe (additional listed companies across the seven sectors;
illustrative verified-domain names from the swap history: mahindra_mahindra, tata_motors,
bajaj_auto, hero_motocorp — document access previously blocked, re-evaluate only) and/or
request mentor/institution-mediated NSE academic access.

## 5. Estimated yield (planning figures, not guarantees)

- Confirmed downloadable URLs in pilot: 68 (PROBED_OK).
- Candidate URLs pending probe: 26 (FOUND).
- Estimated kept after gate (observed keep rate 66/80 ≈ 0.8, pre-dedupe): ≤45.
- Existing kept corpus: 66.
- Post-pilot projected kept: roughly 100–130, before dedupe against existing corpus.
- Gap to 500 minimum: ≈400+ — expansion beyond the pilot is required; this plan does not
  claim the capstone corpus requirement is met.

## 6. Deliverables after pilot execution (not now)

- `data/corpus/raw/` additions (PDFs only), checksummed.
- `data/corpus/ledger.jsonl` append-only rows with the master-ledger schema fields.
- Updated `data/corpus/gate_report.json` and reconcile output.
- Coverage by sector/company/type/period; duplicates and revisions; access failures with
  reasons; financial QA coverage statement.

## 7. Explicitly out of scope for this phase

- No downloads or bulk acquisition yet.
- No modification of `data/seeds/*.csv`, `data/corpus/ledger.jsonl`,
  `data/corpus/gate_report.json`, or production code.
- No NSE/BSE/SEBI fetching.
- No XBRL counting as documents.
