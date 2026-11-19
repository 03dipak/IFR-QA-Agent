"""Question templates + category mix for the Indian-filings corpus — the ported set.

Provenance (2026-11-08): templates adapted from this repo's
``src/eval/gold_candidates.py`` (authored 2026-11-08 for NSE printed
financial-results PDFs) and from the category discipline of the EDGAR gold
tooling (``financial_reporting/src/collector/gold/schema.py``; ADR-005 §8).
The EDGAR templates were NOT copied: they anchor to Item 1A/Item 7, XBRL
tagged facts, and USD figures.

Corpus adaptations:

* Indian fiscal year ends 31 March (FY2024 = year ended 2024-03-31).
* Standalone vs consolidated are separate statements in one filing; the
  basis is named INSIDE the question text of every generated row.
* Quarterly results print quarter / half-year / nine-months columns; the
  questions anchor to the stated period.
* Statement lines are Ind AS lines as printed by NSE issuers.
* Units are NOT USD; units come from the statement's unit header on the
  cited page (Rs. in crore / lakh / million / Rs. per share / %).

Kept EDGAR-set mix where it fits (ADR-005 six categories mapped):
prose → numeric-from-table/single-fact (this corpus's prose surface is the
statement notes, not risk factors); numeric / numeric_with_calculation →
numeric_from_table / multi_hop; multi_step → multi_hop; unanswerable →
unanswerable_abstain; multi_turn deferred. Cross-period and cross-company
comparison are additions specific to this multi-year, multi-company corpus.
"""

from __future__ import annotations

import hashlib

#: Category mix (fractions of the 100-row set).
MIX = {
    "cross_company_comparison": 0.20,
    "unanswerable_abstain": 0.15,
    "single_fact_lookup": 0.20,
    "numeric_from_table": 0.15,
    "same_company_cross_period": 0.15,
    "multi_hop": 0.15,
}

ABSTAIN_SUBTYPES = ("not_in_corpus", "out_of_scope")

INDAS_INCOME_LINES = (
    "revenue from operations",
    "total income",
    "total expenses",
    "profit before tax",
    "profit for the year",
    "profit for the period",
)
INDAS_BALANCE_LINES = (
    "total assets",
    "total equity",
    "total liabilities",
)
INDAS_CASHFLOW_LINES = None  # parked: cash-flow variable labels are noisy

#: Canonical unit labels derived from the unit header ON THE CITED PAGE.
_UNIT_PATTERNS = [
    (r"crore", "Rs. in crore"),
    (r"lakh", "Rs. in lakh"),
    (r"lakhs", "Rs. in lakh"),
    (r"million|mio\b", "Rs. in million"),
    (r"per share", "Rs. per share"),
    (r"percent|per cent|%", "%"),
]


def canonicalise_unit(unit_header: str, page_text: str = "") -> str | None:
    """Map a unit header (or, failing that, the page text) to a canonical label.

    Returns None when the page gives no unit evidence — the candidate is
    rejected upstream rather than guessed.
    """
    import re
    s = (unit_header or "").strip().lower()
    for pat, canon in _UNIT_PATTERNS:
        if re.search(pat, s):
            return canon
    s = (page_text or "").lower()
    for pat, canon in _UNIT_PATTERNS[:4]:
        if re.search(pat, s):
            return canon
    return None


#: period_phrase: annual FY -> "the year ended 31 march 20xx"; quarterly ->
#: "the quarter ended <month-dd>". Deterministic, no dates beyond the label.
_Q_MONTH = {"Q1": "june 30", "Q2": "september 30", "Q3": "december 31", "Q4": "march 31"}


def period_phrase(doc_type: str, period: str) -> str:
    if not period:
        return "the reporting period"
    if doc_type == "annual":
        try:
            year = int(str(period)[2:])
        except ValueError:
            return f"the year ended 31 March {period}"
        return f"the year ended 31 March {year}"
    if doc_type == "quarterly" and str(period)[:2] in _Q_MONTH:
        q = str(period)[:2]
        try:
            yy = int(str(period)[4:]) % 100
        except ValueError:
            return f"the {q} quarter of {period}"
        end = _Q_MONTH[q]
        return f"the quarter ended {end.title()} {2000 + yy:04d}"
    return f"the period {period}"


def basis_phrase(basis: str) -> str:
    return {"consolidated": "consolidated",
            "standalone": "standalone",
            "single": "single, entity-level",
            "combined": "combined, both bases"}.get(basis or "", basis or "consolidated")


_TPL_SINGLE = [
    ("In the {basis} results of {comp} for {period}, what was {metric}, "
     "reported in {unit} as printed?"),
    ("For the {basis} results of {comp} for {period}: state {metric} "
     "exactly as printed, in {unit}."),
    ("From {comp}'s {period} {basis} results, report {metric} "
     "(as printed in the statement, in {unit})."),
    ("Reading the {basis} results of {comp} for {period} as filed, give me "
     "{metric} in {unit}."),
]
_TPL_NUMERIC = [
    ("On the {basis} statement page of {comp} for {period}: give {metric} "
     "as printed, in {unit}."),
    ("On {comp}'s statement page for {period} ({basis}), report {metric} "
     "as printed, in {unit}."),
    ("From the {period} {basis} statement tables of {comp}, read "
     "{metric}, in {unit}."),
    ("Give me {metric} from {comp}'s {period} {basis} statement page, "
     "as printed, in {unit}."),
]
_TPL_CROSS_PERIOD = [
    ("For {comp} ({basis} results, as printed), by how much did {metric} "
     "change between {y1} and {y2}? Report both figures in {unit} and the "
     "signed difference."),
    ("Two-year window for {comp} ({basis}): {metric} moved from {y1} to "
     "{y2}; report both figures in {unit} and the change."),
    ("Against the {y2} {basis} results of {comp}, state {metric} for {y2} "
     "and for {y1}, both in {unit}, and the signed change."),
    ("Compare {metric} for {comp} across {y1} and {y2} ({basis}): give "
     "both figures in {unit} and the signed difference."),
]
_TPL_CROSS_COMPANY = [
    ("For {period}, {basis} (both companies): is {metric} larger for {a} "
     "or for {b}, and what are the two figures in {unit} as printed?"),
    ("Compare {a} and {b} on {metric} for {period} {basis} results "
     "(both {a} and {b} at the same basis): who reports more, and by how "
     "much, in {unit} as printed?"),
    ("Between {a} and {b}, on {metric} for {period} {basis} results, "
     "which prints the larger figure, and what are the two figures "
     "(in {unit}, as printed on each statement)?"),
    ("Which of {a} and {b} prints the higher {metric} in {period} {basis} "
     "results? State the two figures in {unit} as printed."),
]
_TPL_MULTI_HOP = [
    ("{comp} ({basis}, {period}): from the same statement page, give "
     "{metric_a} and {metric_b} as printed there, in {unit}."),
    ("On the {basis} statement page of {comp} for {period}: state both "
     "{metric_a} and {metric_b} as printed, in {unit}."),
    ("{comp}, {basis} results, {period}: consult one statement page and "
     "report {metric_a} and {metric_b} from it, in {unit}."),
    ("On {comp}'s statement page for {period} ({basis}), give me "
     "{metric_a} and {metric_b} exactly as printed there, in {unit}."),
]
_TPL_ABSTAIN = [
    ("For {comp}'s {basis} results for {period}: what was {metric}? "
     "If it is not printed, abstain."),
    ("What was {metric} for {comp} ({basis}, {period})? If the figure is "
     "not printed, abstain."),
    ("From {comp}'s {period} {basis} results: {metric} — a figure only "
     "if it is printed, otherwise abstain."),
    ("If the {basis} results of {comp} for {period} print {metric}, give "
     "it; if they do not, abstain."),
]


def _pick(templates: list[str], *salt: str) -> str:
    h = hashlib.sha256("|".join(salt).encode("utf-8")).hexdigest()
    return templates[int(h, 16) % len(templates)]


def _near_dup(a: str, b: str, threshold: float = 0.9) -> bool:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def question(kind: str, *, _avoid: list[str] | None = None, **kw) -> str | None:
    """Build the question string for ``kind`` with keyword arguments.

    ``_avoid`` lists already-accepted question strings; the template variant
    that would produce a >=0.9 token-set jaccard with any of them is skipped
    (returns None if every variant collides).
    """
    avoid = _avoid or []
    templates = _template_list(kind)
    if templates is None or kind not in _KINDS:
        raise ValueError(f"unknown question kind {kind!r}")
    salt = "|".join(f"{k}={kw[k]}" for k in sorted(kw))
    start = int(hashlib.sha256(salt.encode("utf-8")).hexdigest(), 16) % len(templates)
    for i in range(len(templates)):
        q = templates[(start + i) % len(templates)].format(**{k: v for k, v in kw.items()})
        if any(_near_dup(q, a) for a in avoid):
            continue
        return q
    return None


def _template_list(kind):
    return {
        "single_fact_lookup": _TPL_SINGLE,
        "numeric_from_table": _TPL_NUMERIC,
        "same_company_cross_period": _TPL_CROSS_PERIOD,
        "cross_company_comparison": _TPL_CROSS_COMPANY,
        "multi_hop": _TPL_MULTI_HOP,
        "unanswerable_abstain": _TPL_ABSTAIN,
    }.get(kind)


_KINDS = {"single_fact_lookup", "numeric_from_table", "same_company_cross_period",
          "cross_company_comparison", "multi_hop", "unanswerable_abstain"}
