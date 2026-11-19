# gold_v2 Plan Stats (combined, in-memory)

**Inputs:** `gold_public.jsonl` (v1.4) minus {G-140..G-146, G-170, G-012, G-013} + `gold_v2_pilot.jsonl` + `gold_v2_rewrites.jsonl` + `gold_v2_absence_drafts.jsonl` (V-001..V-007) + `gold_v2_quarterly.jsonl`. Dedupe: by `question_id` and by near-identical question text (Jaccard ≥ 0.9, first-seen kept). 250 raw → **236** after dedupe.

## Totals
| Metric | Count | Share |
|---|---:|---:|
| Total rows | 236 | 100% |
| Answer | 125 | 53.0% |
| Abstain | 111 | 47.0% |

## Abstain split (by subtype)
| Subtype | Count |
|---|---:|
| not_in_corpus | 64 |
| near_miss_absent | 23 |
| out_of_scope | 17 |
| company_not_in_corpus | 7 |

## Doc-type split
| Doc type | Count |
|---|---:|
| annual | 186 |
| quarterly | 50 |

## Question-type split
| Category | Count |
|---|---:|
| unanswerable_abstain | 111 |
| single_fact_lookup | 33 |
| same_company_cross_period | 29 |
| cross_company_comparison | 24 |
| numeric_from_table | 24 |
| multi_hop | 15 |

## Per-company rows (top)
adani_green 13, biocon 11, deepak_nitrite 10, aurobindo_pharma 10, bharat_electronics 10, bhel 10, hcl 9, britannia 9, asian_paints 9, multi 9, ambig 9, RELIANCE 8, others ≤8.

**>5% of total (236):** adani_green — 13 rows (5.51%). None others above 5%.

## Anomaly resolution (plan review)

### 1. Deduped-out rows (14, all annual cross_company_comparison, none quarterly/abstain)
| Dropped | Matched | Jaccard | Both quarterly | Both abstain |
|---|---|---:|---|---|
| G-104 | G-103 | 0.909 | no | no |
| G-108 | G-102 | 1.000 | no | no |
| G-109 | G-103 | 1.000 | no | no |
| G-110 | G-103 | 0.909 | no | no |
| G-113 | G-112 | 0.919 | no | no |
| G-115 | G-114 | 0.946 | no | no |
| G-118 | G-112 | 1.000 | no | no |
| G-119 | G-112 | 0.919 | no | no |
| G-120 | G-114 | 1.000 | no | no |
| G-202 | G-102 | 0.905 | no | no |
| G-203 | G-102 | 0.905 | no | no |
| G-204 | G-105 | 0.909 | no | no |
| G-205 | G-112 | 0.944 | no | no |
| G-206 | G-112 | 0.944 | no | no |

### 2. company=multi / ambig rows (18)
| id | company | origin | cat | question |
|---|---|---|---|---|
| G-147 | multi | gold_public.jsonl | unanswerable_abstain | What does 'mid-cap' mean? |
| G-148 | multi | gold_public.jsonl | unanswerable_abstain | Recommend three mutual funds. |
| G-149 | multi | gold_public.jsonl | unanswerable_abstain | What is the difference between a bear and a bull market? |
| G-150 | multi | gold_public.jsonl | unanswerable_abstain | How do you make masala chai at home? |
| G-151 | multi | gold_public.jsonl | unanswerable_abstain | Describe a good Marathi poem about Mumbai monsoons. |
| G-152 | multi | gold_public.jsonl | unanswerable_abstain | Translate 'equity research' into Hindi. |
| G-153 | multi | gold_public.jsonl | unanswerable_abstain | What is a good book to read for beginner traders? |
| G-154 | multi | gold_public.jsonl | unanswerable_abstain | Explain what a Python dict is in one sentence. |
| G-171 | ambig | gold_public.jsonl | unanswerable_abstain | Compare the revenue of both firms. |
| G-172 | ambig | gold_public.jsonl | unanswerable_abstain | Which one did better in FY2024? |
| G-173 | ambig | gold_public.jsonl | unanswerable_abstain | Give me the balance sheet of X and Y. |
| G-174 | ambig | gold_public.jsonl | unanswerable_abstain | Did the company report higher profit or lower profit, exactly as printed, in Rs. crore, for FY2026? |
| G-175 | ambig | gold_public.jsonl | unanswerable_abstain | Compare Firm A's EBITDA and Firm B's EBITDA for FY2024; report both in Rs. crore. |
| G-176 | ambig | gold_public.jsonl | unanswerable_abstain | State which company had the larger FY2026 revenue, exactly as printed in Rs. crore; do not invent names. |
| G-177 | ambig | gold_public.jsonl | unanswerable_abstain | List the cash from operations for both companies for FY2025; report both in Rs. crore, exactly as printed. |
| G-178 | ambig | gold_public.jsonl | unanswerable_abstain | Which of the two firms grew faster in FY2026? Give the reported growth rate as printed as a percentage. |
| G-179 | ambig | gold_public.jsonl | unanswerable_abstain | State the FY2024 net worth of each company as printed in Rs. crore. |
| G-182 | multi | gold_public.jsonl | unanswerable_abstain | Which mutual funds should I pick for my retirement? |

### 3. adani_green (13 rows)
| id | origin | category | subtype | question (trunc) |
|---|---|---|---|---|
| G-039 | gold_public.jsonl | single_fact_lookup | - | Reading the standalone results of Adani Green for the year ended 31 March 2026 as filed, give me total assets… |
| G-040 | gold_public.jsonl | single_fact_lookup | - | In the standalone results of Adani Green for the year ended 31 March 2026, what was total equity… |
| G-041 | gold_public.jsonl | single_fact_lookup | - | For the standalone results of Adani Green for the year ended 31 March 2026: state total liabilities… |
| G-061 | gold_public.jsonl | numeric_from_table | - | On the standalone statement page of Adani Green for the year ended 31 March 2026: give profit for the year… |
| G-062 | gold_public.jsonl | numeric_from_table | - | On Adani Green's statement page for the year ended 31 March 2026 (consolidated), report revenue from operations… |
| G-133 | gold_public.jsonl | unanswerable_abstain | out_of_scope | Excerpt from Adani Green's FY2026 annual report: what was the company's reported average cost per kWh… |
| G-184 | gold_public.jsonl | unanswerable_abstain | not_in_corpus | Based on the FY2026 results, should I invest in Adani Green? |
| G-189 | gold_public.jsonl | unanswerable_abstain | not_in_corpus | Would you recommend Adani Green shares for a growth portfolio? |
| G-199 | gold_public.jsonl | unanswerable_abstain | not_in_corpus | Estimate Adani Green's FY2028 capacity addition in MW, as you best can. |
| P-004 | gold_v2_pilot.jsonl | single_fact_lookup | - | What was Adani Green Energy's standalone revenue from operations for Q1 FY2026 as filed? |
| P-005 | gold_v2_pilot.jsonl | single_fact_lookup | - | What was Adani Green Energy's standalone total profit (loss) for the period in Q1 FY2026? |
| P-009 | gold_v2_pilot.jsonl | unanswerable_abstain | not_in_corpus | What credit rating did Adani Green Energy disclose in its Q1 FY2026 quarterly filing? |
| P-010 | gold_v2_pilot.jsonl | unanswerable_abstain | not_in_corpus | What full-year revenue guidance did Adani Green Energy give in its Q1 FY2026 filing? |

**Propose 2 cuts:** G-184 and G-189 — redundant investment-advice/recommendation abstains (same intent, both not_in_corpus) for the same company.

### 4. Proposed cuts to reach ~42% abstain (from company_not_in_corpus and near-duplicate template abstains only)
Baseline: 236 rows, 111 abstain (47.0%). Eligible pools: `company_not_in_corpus` (7) + near-duplicate template abstains / `near_miss_absent` V-drafts & quarterly (23) = 30.
- Cut all 7 `company_not_in_corpus`: G-140, G-141, G-142, G-143, G-144, G-145, G-146.
- Cut 7 `near_miss_absent` V-drafts (annual template): V-001..V-007.
- Cut 6 quarterly template abstains (1 per company, second of pair): QT-033..QT-038.

**Total cuts: 20 (all abstain). Projected:** 216 rows, 91 abstain / 125 answer — **abstain 42.1%**.
