"""Unit tests for LLD-02 extraction-QA gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.ingestion.freeze_manager import FreezeManager
from src.ingestion.models import (
    PageExtract,
    PageType,
)
from src.ingestion.qa import (
    GroundTruth,
    ReviewerAnnotation,
    character_error_rate,
    determine_failure_action,
    gate_passes,
    qa_report_from_result,
    run_qa,
    statement_digit_accuracy,
    stratified_sample,
    word_fidelity,
)


def _page(idx: int, text: str, ptype: PageType = PageType.PROSE) -> PageExtract:
    return PageExtract(
        document_id="d1",
        extraction_run_id="run1",
        pdf_page_index=idx,
        page_text=text,
        extraction_hash="h",
        page_type=ptype,
    )


def _gt(idx: int, ref: str, digits: list[str] | None = None) -> GroundTruth:
    now = datetime.now(UTC)
    ann1 = ReviewerAnnotation(reviewer_id="r1", reviewed_at=now, reference_text=ref,
                             statement_digits=digits or [])
    ann2 = ReviewerAnnotation(reviewer_id="r2", reviewed_at=now + timedelta(minutes=10),
                             reference_text=ref, statement_digits=digits or [])
    return GroundTruth(page_index=idx, annotations=[ann1, ann2])


def test_word_fidelity_exact():
    assert word_fidelity("revenue was 100", "revenue was 100") == 1.0


def test_word_fidelity_partial():
    f = word_fidelity("revenue 100", "revenue was 100 crores")
    assert 0 < f < 1.0


def test_character_error_rate_zero():
    assert character_error_rate("abc", "abc") == 0.0


def test_character_error_rate_nonzero():
    assert character_error_rate("abc", "axc") > 0


def test_statement_digit_accuracy_exact():
    assert statement_digit_accuracy("revenue 1234", ["1234"]) == 1.0


def test_statement_digit_accuracy_mismatch():
    assert statement_digit_accuracy("revenue 1235", ["1234"]) == 0.0


def test_stratified_sample_min_per_stratum():
    pages = [_page(i, f"text {i}", PageType.PROSE) for i in range(1, 21)]
    sampled, seed, strata = stratified_sample(pages, seed=42, rate=0.05, min_per_stratum=2)
    assert seed == 42
    assert len(sampled) >= 2
    assert all(s.sample_size >= 0 for s in strata)


def test_stratified_sample_reproducible():
    pages = [_page(i, f"text {i}", PageType.PROSE) for i in range(1, 31)]
    a, _, _ = stratified_sample(pages, seed=7)
    b, _, _ = stratified_sample(pages, seed=7)
    assert [p.pdf_page_index for p in a] == [p.pdf_page_index for p in b]


def test_stratified_sample_rate_clamp():
    pages = [_page(i, f"text {i}") for i in range(1, 101)]
    sampled, _, _ = stratified_sample(pages, seed=1, rate=0.01)
    # clamped up to 0.05
    assert len(sampled) >= 5


def test_run_qa_pass():
    pages = [_page(i, "revenue was 100 crores", PageType.PROSE) for i in range(1, 11)]
    gt = {p.pdf_page_index: _gt(p.pdf_page_index, "revenue was 100 crores") for p in pages}
    sampled = pages
    result = run_qa(sampled, gt, seed=1)
    assert result.passed is True
    assert result.failure_action == "none"
    assert gate_passes(result) is True


def test_run_qa_fail_low_fidelity():
    pages = [_page(i, "garbage text", PageType.PROSE) for i in range(1, 11)]
    gt = {p.pdf_page_index: _gt(p.pdf_page_index, "revenue was 100 crores") for p in pages}
    result = run_qa(pages, gt, seed=1)
    assert result.passed is False
    assert result.failure_action in ("fix_extractor", "exclude_document")


def test_determine_failure_action_systemic():
    assert determine_failure_action(5, 10) == "fix_extractor"


def test_determine_failure_action_isolated():
    assert determine_failure_action(1, 10) == "exclude_document"


def test_determine_failure_action_statement_failed():
    assert determine_failure_action(0, 10, statement_failed=True) == "fix_extractor"


def test_run_qa_empty():
    result = run_qa([], {}, seed=1)
    assert result.passed is False
    assert result.failure_action == "fix_extractor"


def test_qa_report_checksum_deterministic():
    pages = [_page(i, "revenue was 100", PageType.PROSE) for i in range(1, 6)]
    gt = {p.pdf_page_index: _gt(p.pdf_page_index, "revenue was 100") for p in pages}
    result = run_qa(pages, gt, seed=2)
    r1 = qa_report_from_result(result, 5)
    r2 = qa_report_from_result(result, 5)
    assert r1.checksum() == r2.checksum()


def test_freeze_refuses_without_qa():
    fm = FreezeManager()
    manifest = fm.create_freeze(
        freeze_tag="v1", parent_freeze_id=None, ledger_rows=[], page_extracts=[],
        extraction_config_hash="cfg", run_id="r1", config_hash="c1",
    )
    with pytest.raises(ValueError):
        fm.publish(manifest.freeze_id)


def test_freeze_refuses_qa_not_passed():
    fm = FreezeManager()
    manifest = fm.create_freeze(
        freeze_tag="v1", parent_freeze_id=None, ledger_rows=[], page_extracts=[],
        extraction_config_hash="cfg", run_id="r1", config_hash="c1",
        qa_report_checksum="abc", qa_passed=False,
    )
    with pytest.raises(ValueError):
        fm.publish(manifest.freeze_id)


def test_freeze_publishes_with_qa_pass():
    fm = FreezeManager()
    manifest = fm.create_freeze(
        freeze_tag="v1", parent_freeze_id=None, ledger_rows=[], page_extracts=[],
        extraction_config_hash="cfg", run_id="r1", config_hash="c1",
        qa_report_checksum="abc", qa_passed=True,
    )
    from src.ingestion.freeze_manager import CandidateAccounting
    fm.validate_freeze(manifest.freeze_id, CandidateAccounting(0, 0, 0, 0), [], [])
    manifest.manifest_checksum = manifest.compute_manifest_checksum()
    fm.freezes[manifest.freeze_id] = manifest
    published = fm.publish(manifest.freeze_id)
    assert published.status == "published"


def test_ground_truth_agreement_two_reviewers():
    gt = _gt(1, "same text", ["100"])
    assert gt.agreement() is True
    assert gt.is_time_separated() is True


def test_ground_truth_disagreement_requires_third():
    now = datetime.now(UTC)
    ann1 = ReviewerAnnotation(reviewer_id="r1", reviewed_at=now, reference_text="a")
    reviewed_at = now + timedelta(minutes=10)
    ann2 = ReviewerAnnotation(reviewer_id="r2", reviewed_at=reviewed_at, reference_text="b")
    gt = GroundTruth(page_index=1, annotations=[ann1, ann2])
    assert gt.has_disagreement() is True
    assert gt.agreement() is False
    reviewed_at3 = now + timedelta(minutes=20)
    ann3 = ReviewerAnnotation(reviewer_id="r3", reviewed_at=reviewed_at3, reference_text="b")
    gt.annotations.append(ann3)
    assert gt.agreement() is True
    assert gt.reference_text() == "b"


def test_scan_pages_recorded():
    pages = [_page(1, "scan text", PageType.SCAN)]
    gt = {1: _gt(1, "scan text")}
    result = run_qa(pages, gt, seed=1)
    assert 1 in result.scan_pages
