"""Extraction QA gate.

Implements lld_ingestion_pdf.txt section 6: stratified sampling, floors,
failure action, QA metrics, two-reviewer ground truth, mixed-page scan
handling, QA report with checksum. Digit-exact on statement pages;
answer-comparison tolerance bands never apply to extraction QA.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

from ..tables.qa import TableQAMetrics  # noqa: E402
from .models import PageExtract, PageType

# Floors (proposed; calibrated at pilot). Answer-comparison tolerance bands
# never apply here.
PAGE_TEXT_FIDELITY_FLOOR = 0.95
TABLE_ROW_ACCURACY_FLOOR = 0.90
STATEMENT_DIGIT_ACCURACY_FLOOR = 0.98
PAGE_MAPPING_ACCURACY_FLOOR = 0.99
CHARACTER_ERROR_RATE_CEILING = 0.05  # 1 - fidelity floor

# Sampling
DEFAULT_SAMPLE_RATE = 0.07  # inside the 5-10% band
DEFAULT_MIN_PER_STRATUM = 2
MIN_SAMPLE_RATE = 0.05
MAX_SAMPLE_RATE = 0.10

# Systemic vs isolated threshold
SYSTEMIC_FAILURE_THRESHOLD = 0.20
TIME_SEPARATION_MIN_SECONDS = 300  # 5 minutes between two reviews


def _normalize(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Ground truth (two time-separated reviewers, disagreement -> third read)
# ---------------------------------------------------------------------------


@dataclass
class ReviewerAnnotation:
    reviewer_id: str
    reviewed_at: datetime
    reference_text: str
    statement_digits: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class GroundTruth:
    page_index: int
    expected_page_index: int | None = None
    annotations: list[ReviewerAnnotation] = field(default_factory=list)

    def is_time_separated(self, min_gap_seconds: int = TIME_SEPARATION_MIN_SECONDS) -> bool:
        if len(self.annotations) < 2:
            return False
        a = self.annotations[0].reviewed_at
        b = self.annotations[1].reviewed_at
        return abs((b - a).total_seconds()) >= min_gap_seconds

    def has_disagreement(self) -> bool:
        if len(self.annotations) < 2:
            return False
        a = _normalize(self.annotations[0].reference_text)
        b = _normalize(self.annotations[1].reference_text)
        return a != b

    def agreement(self) -> bool:
        if len(self.annotations) < 2:
            return False
        if not self.has_disagreement():
            return True
        # disagreement resolved by a third read
        return len(self.annotations) >= 3

    def reference_text(self) -> str:
        if len(self.annotations) >= 3:
            return self.annotations[2].reference_text
        if self.annotations:
            return self.annotations[0].reference_text
        return ""

    def statement_digits(self) -> list[str]:
        if len(self.annotations) >= 3:
            return self.annotations[2].statement_digits
        if self.annotations:
            return self.annotations[0].statement_digits
        return []


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def word_fidelity(extracted: str, reference: str) -> float:
    """Word-level accuracy: LCS of word sequences / len(reference words)."""
    ref_words = reference.split()
    ext_words = extracted.split()
    if not ref_words:
        return 1.0
    m, n = len(ref_words), len(ext_words)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if ref_words[i - 1] == ext_words[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[n] / m


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def character_error_rate(extracted: str, reference: str) -> float:
    """Levenshtein distance / len(reference)."""
    if not reference:
        return 0.0 if not extracted else 1.0
    return _levenshtein(extracted, reference) / len(reference)


def score_page(extracted: str, reference: str,
               statement_digits: list[str] | None = None) -> dict[str, float]:
    """Score one extracted page against a reference page (QA-kit contract).

    Reuses the canonical metric functions; no new metrics are defined here.
    """
    return {
        "word_fidelity": word_fidelity(extracted, reference),
        "character_error_rate": character_error_rate(extracted, reference),
        "statement_digit_accuracy": (
            statement_digit_accuracy(extracted, statement_digits)
            if statement_digits else 1.0
        ),
    }


def digit_tokens(text: str) -> list[str]:
    return re.findall(r"\d[\d,\.]*", text)


def statement_digit_accuracy(extracted: str, reference_digits: list[str]) -> float:
    """Digit-exact comparison on statement pages. A single digit mismatch
    fails the cell."""
    if not reference_digits:
        return 1.0
    ext_set = set(digit_tokens(extracted))
    ok = sum(1 for d in reference_digits if d in ext_set)
    return ok / len(reference_digits)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


@dataclass
class Stratum:
    key: str
    population: int
    sample_size: int


def stratified_sample(
    pages: list[PageExtract],
    seed: int,
    rate: float = DEFAULT_SAMPLE_RATE,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
    stratum_fn: Callable[[PageExtract], str] | None = None,
) -> tuple[list[PageExtract], int, list[Stratum]]:
    """Stratified sample. Returns (sampled pages, seed, strata info).

    Rate clamped to [MIN_SAMPLE_RATE, MAX_SAMPLE_RATE] (5-10% band).
    """
    if stratum_fn is None:
        stratum_fn = lambda p: p.page_type.value  # noqa: E731
    rate = max(MIN_SAMPLE_RATE, min(MAX_SAMPLE_RATE, rate))
    rng = random.Random(seed)
    strata: dict[str, list[PageExtract]] = {}
    for p in pages:
        strata.setdefault(stratum_fn(p), []).append(p)
    out: list[PageExtract] = []
    stratum_info: list[Stratum] = []
    for key in sorted(strata):
        items = strata[key]
        n = max(min_per_stratum, int(len(items) * rate))
        n = min(n, len(items))
        if n > 0 and items:
            out.extend(rng.sample(items, n))
        stratum_info.append(Stratum(key=key, population=len(items), sample_size=n))
    return out, seed, stratum_info


# ---------------------------------------------------------------------------
# Table-side QA contract (owned by tables LLD; consumed here)
# ---------------------------------------------------------------------------


# TableQAMetrics: single source of truth is src/tables/qa.py (imported at top)


# ---------------------------------------------------------------------------
# QA result + report
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    sample_size: int
    page_text_fidelity: float
    character_error_rate: float
    table_row_accuracy: float
    statement_digit_accuracy: float
    page_mapping_accuracy: float
    passed: bool
    failure_action: str  # "none" | "fix_extractor" | "exclude_document"
    notes: str = ""
    seed: int = 0
    strata: list[Stratum] = field(default_factory=list)
    table_metrics: TableQAMetrics | None = None
    scan_pages: list[int] = field(default_factory=list)
    systemic: bool = False


@dataclass
class QAReport:
    freeze_candidate_count: int
    sample_size: int
    seed: int
    strata: list[dict]
    page_text_fidelity: float
    character_error_rate: float
    table_row_accuracy: float
    statement_digit_accuracy: float
    page_mapping_accuracy: float
    passed: bool
    failure_action: str
    systemic: bool
    notes: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def checksum(self) -> str:
        payload = asdict(self)
        payload.pop("generated_at", None)
        canonical = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_safe(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


def determine_failure_action(failed_pages: int, total_pages: int,
                             statement_failed: bool = False) -> str:
    if total_pages == 0:
        return "fix_extractor"
    if statement_failed:
        return "fix_extractor"
    if (failed_pages / total_pages) >= SYSTEMIC_FAILURE_THRESHOLD:
        return "fix_extractor"
    return "exclude_document"


def run_qa(sample: list[PageExtract],
           ground_truth: dict[int, GroundTruth],
           table_metrics: TableQAMetrics | None = None,
           seed: int = 0,
           strata: list[Stratum] | None = None) -> QAResult:
    """Run the QA gate over a sampled set.

    ground_truth: pdf_page_index -> GroundTruth (two-reviewer annotations).
    """
    if not sample:
        return QAResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "fix_extractor",
                        "empty sample", seed=seed, strata=strata or [])

    fidelities: list[float] = []
    cers: list[float] = []
    stmt_ok = 0
    stmt_total = 0
    page_mapped = 0
    scan_pages: list[int] = []
    failing_pages = 0
    statement_failed = False

    for p in sample:
        gt = ground_truth.get(p.pdf_page_index)
        if gt is None:
            fidelities.append(0.0)
            cers.append(1.0)
            failing_pages += 1
            continue
        ref_text = gt.reference_text()
        ext_text = p.page_text
        f = word_fidelity(ext_text, ref_text)
        c = character_error_rate(ext_text, ref_text)
        fidelities.append(f)
        cers.append(c)
        if f < PAGE_TEXT_FIDELITY_FLOOR:
            failing_pages += 1

        if p.page_type == PageType.STATEMENT:
            ref_digits = gt.statement_digits()
            if ref_digits:
                ext_digits = digit_tokens(ext_text)
                ext_set = set(ext_digits)
                ok = sum(1 for d in ref_digits if d in ext_set)
                stmt_ok += ok
                stmt_total += len(ref_digits)
                if ok < len(ref_digits):
                    statement_failed = True

        if p.page_type == PageType.SCAN:
            scan_pages.append(p.pdf_page_index)

        # page mapping: ground truth present for this page
        if gt.expected_page_index is None or gt.expected_page_index == p.pdf_page_index:
            page_mapped += 1

    fidelity = (sum(fidelities) / len(fidelities)) if fidelities else 0.0
    cer = (sum(cers) / len(cers)) if cers else 0.0
    stmt_acc = (stmt_ok / stmt_total) if stmt_total else 1.0
    pma = (page_mapped / len(sample)) if sample else 0.0

    tm = table_metrics if table_metrics is not None else TableQAMetrics()
    table_acc = tm.table_row_accuracy

    action = determine_failure_action(failing_pages, len(sample), statement_failed)
    systemic = action == "fix_extractor"

    passed = (
        fidelity >= PAGE_TEXT_FIDELITY_FLOOR
        and cer <= CHARACTER_ERROR_RATE_CEILING
        and table_acc >= TABLE_ROW_ACCURACY_FLOOR
        and stmt_acc >= STATEMENT_DIGIT_ACCURACY_FLOOR
        and pma >= PAGE_MAPPING_ACCURACY_FLOOR
    )

    if passed:
        action = "none"
    elif systemic:
        action = "fix_extractor"
    else:
        action = "exclude_document"

    notes = ""
    if scan_pages:
        notes = "scanned (non-statement) pages recorded: " + ",".join(str(x) for x in scan_pages)

    return QAResult(
        sample_size=len(sample),
        page_text_fidelity=fidelity,
        character_error_rate=cer,
        table_row_accuracy=table_acc,
        statement_digit_accuracy=stmt_acc,
        page_mapping_accuracy=pma,
        passed=passed,
        failure_action=action,
        notes=notes,
        seed=seed,
        strata=strata or [],
        table_metrics=tm,
        scan_pages=scan_pages,
        systemic=systemic,
    )


def qa_report_from_result(result: QAResult, candidate_count: int) -> QAReport:
    return QAReport(
        freeze_candidate_count=candidate_count,
        sample_size=result.sample_size,
        seed=result.seed,
        strata=[{"key": s.key, "population": s.population, "sample_size": s.sample_size}
                for s in result.strata],
        page_text_fidelity=result.page_text_fidelity,
        character_error_rate=result.character_error_rate,
        table_row_accuracy=result.table_row_accuracy,
        statement_digit_accuracy=result.statement_digit_accuracy,
        page_mapping_accuracy=result.page_mapping_accuracy,
        passed=result.passed,
        failure_action=result.failure_action,
        systemic=result.systemic,
        notes=result.notes,
    )


def gate_passes(result: QAResult) -> bool:
    return bool(result.passed)


# ---------------------------------------------------------------------------
# QA-kit generation and scoring contract (kit is written, never scored here)
# ---------------------------------------------------------------------------


def qa_kit_stratum(page: PageExtract) -> str:
    """Stratum key: document prefix | page class | statement-like flag."""
    return (
        f"{page.document_id[:12]}|{page.page_class.value}|"
        f"stmt:{str(_is_statement_like(page.page_text)).lower()}"
    )


def _is_statement_like(text: str) -> bool:
    from .extractor import is_statement_like

    return is_statement_like(text)


def build_qa_kit_rows(
    pages: list[PageExtract],
    companies: dict[str, str],
    seed: int = 42,
    rate: float = DEFAULT_SAMPLE_RATE,
    min_per_stratum: int = DEFAULT_MIN_PER_STRATUM,
) -> tuple[list[dict], list[Stratum]]:
    """Sample pages with the existing stratified sampler and build kit rows."""
    sampled, _seed, strata = stratified_sample(
        pages, seed=seed, rate=rate, min_per_stratum=min_per_stratum,
        stratum_fn=qa_kit_stratum)
    rows: list[dict] = []
    for page in sorted(sampled, key=lambda p: (p.document_id, p.pdf_page_index)):
        rows.append({
            "document_checksum": page.document_id,
            "company": companies.get(page.document_id, ""),
            "pdf_page_index": page.pdf_page_index,
            "printed_page_label": page.printed_page_label or "",
            "page_class": page.page_class.value,
            "page_type": page.page_type.value,
            "statement_like": str(_is_statement_like(page.page_text)),
            "stratum": qa_kit_stratum(page),
            "page_text": page.page_text,
        })
    return rows, strata


def score_qa_kit_rows(
    kit_rows: list[dict],
    references: dict[tuple[str, int], str],
    statement_digits: dict[tuple[str, int], list[str]] | None = None,
) -> list[dict]:
    """Score a filled QA kit against reviewer reference text.

    ``references`` is keyed by (document_checksum, pdf_page_index). The kit is
    not scored by this module's CLI; this is the deterministic scorer only.
    """
    scored: list[dict] = []
    for row in kit_rows:
        key = (row["document_checksum"], int(row["pdf_page_index"]))
        reference = references.get(key)
        if reference is None:
            scored.append({**row, "scored": False})
            continue
        metrics = score_page(
            row.get("page_text", ""), reference,
            (statement_digits or {}).get(key))
        scored.append({**row, **metrics, "scored": True})
    return scored
