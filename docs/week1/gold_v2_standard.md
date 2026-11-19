# Gold v2 standard (draft, <=1 page)

Basis: `docs/gold_verification_protocol.md` (docs repo) + gold schema 2.0 as used by
`gold_v1.4`. gold_v2 is a NEW frozen set; `gold_v1.4` and `corpus freeze v1/v2` are
never edited (baselines are pinned to their hashes).

## Row schema
- Reuse schema 2.0 fields; every answer row needs: `question`, `expected_answer`,
  `expected_value`/`expected_unit`, `evidence[]` (document_id + 1-indexed pdf_page +
  printed excerpt), `category`, `company_id`, `period`, `doc_type`, `variant`.
- Every answer row must be **re-computed from the printed table** on the cited page
  (arithmetic check); every abstain row must be **verified absent** from the company's
  frozen corpus docs (abstain keyword-hit report + reviewer confirmation).
- Rows are signed with `signed_sha256` only after review; pilot rows ship unsigned and
  marked `unverified`.
- `gold_set_version: gold_v2.0`; new freeze reference recorded next to gold_hash.

## Coverage targets
| Target | Rule |
|---|---|
| Per company minimum | >= 4 rows per company with annual rows; companies with < 4 are dropped or topped up |
| Annual vs quarterly share | keep ~70/30 once quarterly statement maps confirmed (v1.4 is 100/0) |
| Question-type mix | ~20/15/20/15/15/15 per protocol (single_fact, numeric_from_table, cross_company, cross_period, multi_hop, abstain) |
| Abstain share | ~40% (v1.4: 43.1% — compliant) |
| No company > 5% of rows | v1.4 violates: `multi` 8.2%, `hcl` 5.6%, `ambig` 5.1% |

## Dev/holdout split
- Deterministic stratified split (25% holdout, seeded, splitter version pinned),
  persisted to the manifest beside the dataset; holdout rows are never used for
  prompt, retrieval or reflection-loop tuning until the final report.

## v1.4 coverage vs these targets (measured 2026-11-13)
- 195 rows, 63 company labels, all `doc_type=annual`, periods FY2024/25/26 (69 `n/a`).
- 175 self_reviewed / 20 peer_reviewed — below the 2-reviewer bar for sign-off.
- Question-type gaps (company x category): cross_company missing in 50/63 companies,
  cross_period 55/63, single_fact 55/63, numeric_from_table 58/63, multi_hop 53/63 —
  v1.4 is dense on few companies, sparse everywhere else.
- Quarterly coverage: 0 rows; blocked on quarterly statement map (see readiness note).
