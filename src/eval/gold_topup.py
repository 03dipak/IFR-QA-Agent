"""Generate the 90 UNREVIEWED draft rows to top up the gold set to LLD quota.

All companies are in the frozen slice. No existing row is edited.
Usage: python -m src.eval.gold_topup  -> writes data/eval/datasets/packs/drafts_topup.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

GOLD = Path("data/eval/datasets/gold_data.jsonl")
OUT = Path("data/eval/datasets/packs/drafts_topup.jsonl")

NOW = "2026-11-08T00:00:00Z"

MACRO = [
    ("hindalco", "Hindalco", "f4ea9e50b6468ce6b9021358972174a56db1a5c0d177893bd98ed989e087bce2",
     "HINDALCO FY2026", 16, "29,440"),
    ("infosys", "Infosys", "602f04ec2abfacbf28ca8ff6e592db51942c620af4c12287d3a717edb6cd078f",
     "INFOSYS FY2026", 16, "29,440"),
    ("itc", "ITC", "2a51bc137e460a2549785e784465e6fe1aa7bcd2ca1077d6e2d2465efdd27acc",
     "ITC FY2025",    64, "26528.82"),
    ("coal_india", "Coal India", "9b63fc3453960fa251c3596f6f1268274d3a159e335acbe9528ca1e3c76e67b1",
     "COAL_INDIA FY2025", 26, "27.81"),
    ("bosch", "Bosch", "39ba05d9e676f4d76442a2c9450d9c78dab6d9009ead5a0c648eb6d7074d43c9",
     "BOSCH FY2026",  20, "20,034"),
    ("cummins_india", "Cummins India", "f3b09714e17ff1afe3bf3584cc7f84870cedd361f8ad06c39d4735dcac1d3b50",
     "CUMMINS FY2026", 29, "12,143.19"),
]


def ev(did, pg, role, excerpt):
    return [{"document_id": did, "pdf_page": pg, "evidence_role": role,
             "excerpt": excerpt, "snippet_hash": None, "version": None}]


def row(i, question, answer, evidence, company, metric, period, bkt, abstain=False):
    cat = bkt if bkt in ("matrix", "neutralitytrap", "multi_turn") else bkt
    expected_behaviour = "answer"
    if bkt in ("out_of_domain", "not_in_corpus_company", "figure_absent",
                "ambiguous", "advice", "forecast"):
        expected_behaviour = "abstain"
    else:
        if bkt == "multi_turn":
            expected_behaviour = "answer"
    return {
        "row_id": f"D-{i:03d}", "question_id": None,
        "category": cat, "company_id": company,
        "period": period, "fiscal_year": period,
        "doc_type": "annual", "variant": "consolidated",
        "metric": metric, "question_text": question, "question": question,
        "expected_answer": answer, "answer_text": answer,
        "expected_value": None,
        "expected_unit": "money",
        "expected_unit_raw": "Rs. in crore",
        "expected_formula": None, "standalone_question": None,
        "evidence": evidence,
        "evidence_span_text": None,
        "document_id": None, "pdf_page_index": None,
        "printed_page_label": None, "table_refs": None,
        "extraction_checksum": None,
        "must_include": [], "must_not_include": [],
        "abstention_subtype": None,
        "expected_behaviour": expected_behaviour,
        "depends_on": None,
        "relevance_rule": "primary",
        "owner": "gold-topup-generator",
        "annotated_at": NOW, "re_annotated_at": None,
        "review_date": None, "review_status": "unreviewed",
        "lifecycle_status": "active",
        "notes": "top-up draft, unreviewed",
        "gold_schema_version": "2.0",
        "gold_set_version": "gold_v1",
        "row_revision": 0, "status": "active",
        "reviewer": None, "reviewers": [],
        "signed_sha256": None,
        "split": "dev", "sub_answers": None,
        "version": None,
        "bucket": bkt,
    }


def build_rows():
    out = []
    r = 200

    def next_id():
        nonlocal r
        r += 1
        return r

    # --- matrix (10) ---------------------------------------------------------
    for k in range(10):
        i = next_id()
        a_idx = k % len(MACRO)
        b_idx = (k + 1) % len(MACRO)
        if b_idx == a_idx:
            b_idx = (k + 2) % len(MACRO)
        a = MACRO[a_idx]
        b = MACRO[b_idx]
        q = (f"Create a comparison matrix for {a[1]} and {b[1]} for FY2026 "
             f"(consolidated, as printed); report both companies' Net profit "
             f"(consolidated, as printed) in Rs. in crore.")
        out.append(row(i, q,
                       f"{a[0]}={a[5]}; {b[0]}={b[5]}",
                       ev(a[2], a[4], "primary", f"Net profit {a[5]}")
                       + ev(b[2], b[4], "comparison", f"Net profit {b[5]}"),
                       company=f"{a[0]}|{b[0]}", metric="net profit",
                       period="FY2026", bkt="matrix"))

    # --- neutrality trap (10) ------------------------------------------------
    for k in range(10):
        i = next_id()
        a_idx = k % len(MACRO)
        b_idx = (k + 1) % len(MACRO)
        if b_idx == a_idx:
            b_idx = (k + 2) % len(MACRO)
        a = MACRO[a_idx]
        b = MACRO[b_idx]
        q = (f"For the year ended 31 March 2026 (consolidated, as printed): "
             f"which of {a[1]} and {b[1]} did better? If you would be expected "
             f"to name one as 'better', give the matrix but do not rank; give "
             f"the printed figures in Rs. in crore.")
        out.append(row(i, q,
                       f"{a[0]}={a[5]}; {b[0]}={b[5]} (neutral, no winner)",
                       ev(a[2], a[4], "primary", f"Net profit {a[5]}")
                       + ev(b[2], b[4], "comparison", f"Net profit {b[5]}"),
                       company=f"{a[0]}|{b[0]}", metric="net profit",
                       period="FY2026", bkt="neutralitytrap"))

    # --- multi_turn follow-up (5 pairs) --------------------------------------
    for k in range(5):
        i1 = next_id()
        i2 = next_id()
        a_idx = k % len(MACRO)
        a = MACRO[a_idx]
        q1 = (f"For the consolidated results of {a[1]} for the year ended "
              f"31 March 2026, what was the company's reported net profit in "
              f"Rs. in crore, exactly as printed?")
        ans1 = f"{a[0]}={a[5]}"
        q2 = (f"And what was the same figure for the prior financial year "
              f"(FY2025) for {a[1]}, as printed in Rs. in crore?")
        ans2 = f"{a[0]}=N/A (not pinned)"
        out.append(row(i1, q1, ans1,
                       ev(a[2], a[4], "primary", f"Net profit {a[5]}"),
                       company=a[0], metric="net profit",
                       period="FY2026", bkt="multi_turn"))
        out.append(row(i2, q2, ans2,
                       ev(a[2], a[4], "primary", f"FY2025 figure"),
                       company=a[0], metric="net profit",
                       period="FY2025", bkt="multi_turn"))

    # --- figure-absent (5 new) -----------------------------------------------
    figure_absent_extra = [
        ("For the consolidated standalone results of Coal India for FY2026 in Rs. in crore: what was the company's reported dividend paid, exactly as printed?", "ABSTAIN", "metric=dividend", "coal_india"),
        ("For Indian Oil Corporation for FY2024 consolidated: what were their audit fees, exactly as printed in Rs. lakh?", "ABSTAIN", "metric=audit_fees", "ioc"),
        ("Excerpt from Adani Green's FY2026 annual report: what was the company's reported average cost per kWh, exactly as printed in Rs., and is it printed for any segment?", "ABSTAIN", "metric=cost_per_kwh", "adani_green"),
        ("State the statutory auditor's name and their report date for ITC's FY2027 audited standalone results, exactly as printed in Rs. million.", "ABSTAIN", "metric=auditor", "itc"),
        ("For Maruti Suzuki FY2026 results: what was the company's Q1 gross margin as a percentage, exactly as printed?", "ABSTAIN", "metric=gross_margin", "maruti_suzuki"),
        ("For Wipro FY2025: what was Wipro's FY2025 percentage contribution from Russia, exactly as printed?", "ABSTAIN", "metric=russia_pct", "wipro"),
        ("For Coal India: state the FY2026 percentage of washeries capacity used, exactly as printed as a ratio.", "ABSTAIN", "metric=washery_use", "coal_india_washery"),
        ("For Hindustan Unilever FY2025: state the FY2025 net carbon emissions printed, in kg CO2e, exactly as printed.", "ABSTAIN", "metric=co2", "hindustan_unilever_co2"),
        ("For ITC FY2026 results: state the ITC Life Sciences subsidiary's gross profit for FY2026 in Rs. crore, exactly as printed.", "ABSTAIN", "metric=life_sciences_gp", "itc_life_sciences"),
    ]
    for q, ans, tag, comp in figure_absent_extra:
        i = next_id()
        out.append(row(i, q, ans,
                       ev("", -1, "", "not-extracted"),
                       company=comp, metric=tag.split("=")[1],
                       period="n/a", bkt="figure_absent"))

    # --- out-of-domain (10) --------------------------------------------------
    out_of_domain = [
        ("What is the recipe for masala dosa?", "ABSTAIN", "food_recipe", "multi"),
        ("Explain the Pythagorean theorem in one sentence.", "ABSTAIN", "theorem", "multi"),
        ("Write a haiku about rain in Mumbai.", "ABSTAIN", "haiku", "multi"),
        ("List three songs by A. R. Rahman.", "ABSTAIN", "songs", "multi"),
        ("What is the capital of Brazil?", "ABSTAIN", "capital_city", "multi"),
        ("Translate 'the stock market is volatile' into Tamil.", "ABSTAIN", "translation", "multi"),
        ("Explain what a neural network is in one sentence.", "ABSTAIN", "neural_net", "multi"),
        ("What does 'mid-cap' mean?", "ABSTAIN", "mid_cap_def", "multi"),
        ("Recommend three mutual funds.", "ABSTAIN", "mf_pick", "multi"),
        ("What is the difference between a bear and a bull market?", "ABSTAIN", "bull_bear", "multi"),
        ("How do you make masala chai at home?", "ABSTAIN", "tea_recipe", "multi"),
        ("Describe a good Marathi poem about Mumbai monsoons.", "ABSTAIN", "poem_monsoons", "multi"),
        ("Translate 'equity research' into Hindi.", "ABSTAIN", "translation_eq", "multi"),
        ("What is a good book to read for beginner traders?", "ABSTAIN", "trading_book", "multi"),
        ("Explain what a Python dict is in one sentence.", "ABSTAIN", "python_dict", "multi"),
    ]
    for q, ans, tag, comp in out_of_domain:
        i = next_id()
        out.append(row(i, q, ans,
                       ev("", -1, "", "off-domain"),
                       company=comp, metric=tag,
                       period="n/a", bkt="out_of_domain"))
    # --- not-in-corpus company (10) -------------------------------------------
    not_in_corpus = [
        ("For Hindustan Unilever Ltd (HUL) for FY2026: what was HUL's embedded carbon offset programme contribution for FY2025, exactly as printed in their FY2024-24C sustainability report?", "ABSTAIN", "cbcd_contrib", "hindustan_unilever"),
        ("For Dr. Reddy's Laboratories FY2024 results: state the FY2024 revenue from Russia as printed, in Rs. million.", "ABSTAIN", "russia_rev", "dr_reddys"),
        ("For UltraTech Cement FY2026: what was UltraTech's FY2026 capital expenditure for the East zone, exactly as printed in Rs. crore?", "ABSTAIN", "east_capex", "ultratech"),
        ("For Reliance Retail Ventures Ltd (a Reliance subsidiary) FY2024-FY2025 results: what were the retail segment's net revenue as printed in Rs. crore?", "ABSTAIN", "retail_rev", "reliance_retail"),
        ("What was Adnoc Distribution (ADNOC)'s FY2024 EBITDA as printed?", "ABSTAIN", "adnoc_ebitda", "adnoc"),
        ("State the FY2024 dividend per share as printed for Faar Bharat? Note the company is not in the corpus.", "ABSTAIN", "dps", "faar_bharat"),
        ("What is the FY2026 headline number for Consolidated EBITDA for Tata Projects Limited, as printed in Rs. crore?", "ABSTAIN", "tata_projects_ebitda", "tata_projects"),
        ("For Tata Motors FY2026 results (not Tata Steel): what was Tata Motors' JLR segment EBITDA, exactly as printed in Rs. crore?", "ABSTAIN", "jlr_ebitda", "tata_motors"),
        ("For Coal India: report the Q1FY2025 consolidated EBITDA margin for the washed coal business, exactly as printed as a percentage.", "ABSTAIN", "washed_margin", "coal_india_mine"),
        ("State the FY2026 audit report date for State Bank of India (SBI) as printed in the auditor's opinion note.", "ABSTAIN", "sbi_audit_date", "sbi"),
        ("State the FY2026 contributed equity for Bharti Airtel as printed in their consolidated balance sheet, in Rs. crore.", "ABSTAIN", "airtel_equity", "bharti_airtel"),
        ("State the FY2026 audited sales volume (tonnes) for Tata Steel as printed in their annual report, exactly as printed.", "ABSTAIN", "tata_steel_tonnes", "tata_steel_tonnes"),
        ("For Adani Total Gas FY2026 results: report the FY2026 gas volumes as printed, in MMSCM.", "ABSTAIN", "adani_total_gas_vol", "adani_total_gas"),
        ("What is the FY2026 statutory Refit audit opinion for Cognizant for FY2026 (names redacted)?", "ABSTAIN", "ltm_audit_opinion", "cognizant"),
        ("State the FY2026 disclosed EBITDA for BEML as printed in their FY2026 report, in Rs. crore.", "ABSTAIN", "beml_ebitda", "beml"),
    ]
    for q, ans, tag, comp in not_in_corpus:
        i = next_id()
        out.append(row(i, q, ans,
                       ev("", -1, "", "off-corpus"),
                       company=comp, metric=tag,
                       period="n/a", bkt="not_in_corpus_company"))

    # --- ambiguous entity (5) --------------------------------------------------
    ambiguous = [
        ("Which of the two companies had the higher net profit, exactly as printed in Rs. crore, for FY2025?", "ABSTAIN", "ambig1", "ambig"),
        ("Compare the revenue of both firms.", "ABSTAIN", "ambig2", "ambig"),
        ("Which one did better in FY2024?", "ABSTAIN", "ambig3", "ambig"),
        ("Give me the balance sheet of X and Y.", "ABSTAIN", "ambig4", "ambig"),
        ("Did the company report higher profit or lower profit, exactly as printed, in Rs. crore, for FY2026?", "ABSTAIN", "ambig5", "ambig"),
        ("Compare Firm A's EBITDA and Firm B's EBITDA for FY2024; report both in Rs. crore.", "ABSTAIN", "ambig6", "ambig"),
        ("State which company had the larger FY2026 revenue, exactly as printed in Rs. crore; do not invent names.", "ABSTAIN", "ambig7", "ambig"),
        ("List the cash from operations for both companies for FY2025; report both in Rs. crore, exactly as printed.", "ABSTAIN", "ambig8", "ambig"),
        ("Which of the two firms grew faster in FY2026? Give the reported growth rate as printed as a percentage.", "ABSTAIN", "ambig9", "ambig"),
        ("State the FY2024 net worth of each company as printed in Rs. crore.", "ABSTAIN", "ambig10", "ambig"),
    ]
    for q, ans, tag, comp in ambiguous:
        i = next_id()
        out.append(row(i, q, ans,
                       ev("", -1, "", "ambig"),
                       company=comp, metric=tag,
                       period="n/a", bkt="ambiguous"))

    # --- advice refusal (5) --------------------------------------------------
    #  Advice wording: the system should not act as financial advisor
    advice = [
        ("Should I buy shares of Maruti Suzuki now?", "ABSTAIN", "adv1", "maruti"),
        ("Is this a good time to sell HCL Technologies?", "ABSTAIN", "adv2", "hcl"),
        ("Which mutual funds should I pick for my retirement?", "ABSTAIN", "adv3", "multi"),
        ("Will ITC beat the market next year, and should I hold it?", "ABSTAIN", "adv4", "itc"),
        ("Based on the FY2026 results, should I invest in Adani Green?", "ABSTAIN", "adv5", "adani_green"),
        ("Should I buy HCL Technologies shares right now at this price?", "ABSTAIN", "adv6", "hcl"),
        ("Are Maruti Suzuki shares a safe retirement investment right now?", "ABSTAIN", "adv7", "maruti_suzuki"),
        ("Should I hold ITC or sell before the dividend announcement?", "ABSTAIN", "adv8", "itc"),
        ("Is Coal India a good long-term buy based on its FY2025 results?", "ABSTAIN", "adv9", "coal_india"),
        ("Would you recommend Adani Green shares for a growth portfolio?", "ABSTAIN", "adv10", "adani_green"),
    ]
    for q, ans, tag, comp in advice:
        i = next_id()
        out.append(row(i, q, ans,
                       ev("", -1, "", "advice"),
                       company=comp, metric=tag,
                       period="n/a", bkt="advice"))

    # --- forecast refusal (5) --------------------------------------------------
    forecast = [
        ("What will Reliance report as revenue for FY2028?", "ABSTAIN", "fcst1", "reliance"),
        ("Predict Coal India's FY2027 EBITDA in Rs. crore, exactly as you can best estimate?", "ABSTAIN", "fcst2", "coal_india"),
        ("Estimate Maruti Suzuki's FY2030 dividend per share based on trend.", "ABSTAIN", "fcst3", "maruti_suzuki"),
        ("What will TCS report as operating margin for FY2027?", "ABSTAIN", "fcst4", "tcs"),
        ("Project the FY2030 net profit of HUL in Rs. crore, as you best can.", "ABSTAIN", "fcst5", "hindustan_unilever"),
        ("What will HCL Technologies report as FY2028 revenue in Rs. crore?", "ABSTAIN", "fcst6", "hcl"),
        ("What will Tata Steel report as FY2028 EBITDA in Rs. crore?", "ABSTAIN", "fcst7", "tata_steel"),
        ("Predict Maruti Suzuki's FY2029 dividend in Rs. per share.", "ABSTAIN", "fcst8", "maruti_suzuki"),
        ("What will ITC's FY2030 revenue be in Rs. crore?", "ABSTAIN", "fcst9", "itc"),
        ("Estimate Adani Green's FY2028 capacity addition in MW, as you best can.", "ABSTAIN", "fcst10", "adani_green"),
    ]
    for q, ans, tag, comp in forecast:
        i = next_id()
        out.append(row(i, q, ans,
                       ev("", -1, "", "forecast"),
                       company=comp, metric=tag,
                       period="n/a", bkt="forecast"))

    return out


def main():
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} topup rows to {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
