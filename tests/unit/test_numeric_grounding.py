"""Unit tests for numeric grounding: expected_formula + lakh/crore normalisation."""

from __future__ import annotations

import json

from src.eval import gold_check as gchk

BHEL_DOC = "b" * 64
BRIT_DOC = "c" * 64
OTHER_DOC = "d" * 64
DP_DOC = "e" * 64


def _seed_pages(tmp_path, monkeypatch):
    pagedir = tmp_path / "pe"
    pagedir.mkdir()
    (pagedir / f"{BHEL_DOC}.pages.jsonl").write_text(
        json.dumps({"pdf_page_index": 3,
                    "page_text": "Nature of report Consolidated Revenue from "
                                 "operations 54869100000"}) + "\n")
    (pagedir / f"{BRIT_DOC}.pages.jsonl").write_text(
        json.dumps({"pdf_page_index": 3,
                    "page_text": "Rs. in lakh Revenue from operations 5,40,463.00"}) + "\n")
    (pagedir / f"{DP_DOC}.pages.jsonl").write_text(
        json.dumps({"pdf_page_index": 2,
                    "page_text": "Revenue from operations 18898800000"}) + "\n")
    monkeypatch.setattr(gchk, "PAGE_DIR", str(pagedir))


def _row(rid, doc, answer, value, formula=None):
    return {"id": rid, "category": "numeric_from_table", "company_id": "co_a",
            "expected_behaviour": "answer", "expected_unit": "money",
            "expected_answer": answer, "expected_value": value,
            "expected_formula": formula,
            "question": "consolidated revenue from operations FY2025?",
            "evidence": [{"document_id": doc, "pdf_page": 3, "evidence_role": "primary",
                          "excerpt": "Revenue from operations"}]}


def test_absolute_print_matches(tmp_path, monkeypatch):
    _seed_pages(tmp_path, monkeypatch)
    row = _row("G-901", BHEL_DOC, "54869100000 INR", 54869100000)
    fails, _ = gchk.check_numeric_grounded([row])
    assert fails == []


def test_lakh_print_normalises(tmp_path, monkeypatch):
    _seed_pages(tmp_path, monkeypatch)
    row = _row("G-902", BRIT_DOC, "54046300000 INR", 54046300000)
    fails, _ = gchk.check_numeric_grounded([row])
    assert fails == []


def test_wrong_unit_scale_still_fails(tmp_path, monkeypatch):
    _seed_pages(tmp_path, monkeypatch)
    row = _row("G-903", BRIT_DOC, "54046300000 INR", 54046300000)
    row["evidence"] = [{"document_id": OTHER_DOC, "pdf_page": 3,
                        "evidence_role": "primary", "excerpt": "x"}]
    (tmp_path / "pe" / f"{OTHER_DOC}.pages.jsonl").write_text(
        json.dumps({"pdf_page_index": 3,
                    "page_text": "Rs. in crore Revenue from operations 540463.00"}) + "\n")
    fails, _ = gchk.check_numeric_grounded([row])
    assert any("no numeric token" in m for _, m in fails)


def test_formula_accepts_printed_operands(tmp_path, monkeypatch):
    _seed_pages(tmp_path, monkeypatch)
    row = _row("G-904", DP_DOC, "17776300000 INR (computed)", 17776300000,
               formula="18898800000 - 1122500000 = 17776300000")
    row["evidence"] = [
        {"document_id": DP_DOC, "pdf_page": 2, "evidence_role": "primary",
         "excerpt": "Revenue 18898800000"},
        {"document_id": BRIT_DOC, "pdf_page": 3, "evidence_role": "primary",
         "excerpt": "Revenue 1122500000"},
    ]
    # formula operands must be printed somewhere; computed result need not be
    (tmp_path / "pe" / f"{BRIT_DOC}.pages.jsonl").write_text(
        json.dumps({"pdf_page_index": 3,
                    "page_text": "Revenue from operations 1122500000"}) + "\n")
    fails, _ = gchk.check_numeric_grounded([row])
    assert fails == []


def test_formula_missing_operand_fails(tmp_path, monkeypatch):
    _seed_pages(tmp_path, monkeypatch)
    row = _row("G-905", BHEL_DOC, "999 INR", 999,
               formula="18898800000 - 1122500000 = 17776300000")
    fails, _ = gchk.check_numeric_grounded([row])
    assert any("no numeric token" in m for _, m in fails)
