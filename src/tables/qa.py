"""Table-side extraction QA metrics.

Implements lld_tables.txt section 5: table-detection precision/recall, row
accuracy, header preservation, digit CER, F1. Owned by the tables LLD;
consumed by the ingestion QA gate (src/ingestion/qa.py) as TableQAMetrics.

This module provides the data contract and a small helper to compute the
combined gate decision; the full table-extraction implementation lands with
LLD-04..06.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TableQAMetrics:
    """Table-side QA metrics contract (owned by tables LLD)."""
    table_detection_precision: float = 1.0
    table_detection_recall: float = 1.0
    table_row_accuracy: float = 1.0
    header_preservation: float = 1.0
    digit_cer: float = 0.0
    f1: float = 1.0


@dataclass
class TableQAGroundTruth:
    """Ground truth for one table (double-annotated, disagreement escalated)."""
    table_id: str
    expected_row_count: int
    expected_header: list[str]
    expected_digits: list[str]
    detected: bool = True


@dataclass
class TableQAResult:
    detection_precision: float
    detection_recall: float
    row_accuracy: float
    header_preservation: float
    digit_cer: float
    f1: float
    passed: bool


def compute_table_qa(
    truth: list[TableQAGroundTruth],
    detected_ids: set[str],
    row_matches: dict[str, int],
    header_matches: dict[str, bool],
    digit_errors: dict[str, int],
    digit_totals: dict[str, int],
) -> TableQAResult:
    """Compute table-side QA metrics from annotated ground truth.

    truth: list of expected tables.
    detected_ids: set of table_ids the extractor found.
    row_matches: table_id -> number of correctly extracted rows.
    header_matches: table_id -> whether header preserved.
    digit_errors/digit_totals: table_id -> counts for digit CER.
    """
    expected_ids = {t.table_id for t in truth}
    tp = len(expected_ids & detected_ids)
    fp = len(detected_ids - expected_ids)
    fn = len(expected_ids - detected_ids)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    row_ok = sum(row_matches.get(t.table_id, 0) for t in truth)
    row_total = sum(t.expected_row_count for t in truth)
    row_acc = (row_ok / row_total) if row_total else 1.0

    header_ok = sum(1 for t in truth if header_matches.get(t.table_id, False))
    header_pres = (header_ok / len(truth)) if truth else 1.0

    d_err = sum(digit_errors.get(t.table_id, 0) for t in truth)
    d_tot = sum(digit_totals.get(t.table_id, 0) for t in truth)
    digit_cer = (d_err / d_tot) if d_tot else 0.0

    passed = (
        precision >= 0.90 and recall >= 0.90 and row_acc >= 0.90
        and header_pres >= 0.95 and digit_cer <= 0.02
    )
    return TableQAResult(
        detection_precision=precision,
        detection_recall=recall,
        row_accuracy=row_acc,
        header_preservation=header_pres,
        digit_cer=digit_cer,
        f1=f1,
        passed=passed,
    )
