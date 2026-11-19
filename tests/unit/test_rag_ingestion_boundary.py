"""Boundary test: rag_ingestion imports nothing from data_pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.rag_ingestion import contract

RAG_DIR = Path("src/rag_ingestion")
_FORBIDDEN = re.compile(
    r"^\s*(?:from|import)\s+(?:src\.ingestion|ingestion|manual_ingest|"
    r"extractor|reconcile|gate_report|freeze_manager|models)\b", re.M)


def test_no_imports_from_data_pipeline():
    sources = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(RAG_DIR.glob("*.py")))
    assert not _FORBIDDEN.findall(sources), (
        "rag_ingestion must not import data_pipeline modules except the "
        "contract reader")


def test_contract_reader_is_stdlib_only():
    source = (RAG_DIR / "contract.py").read_text(encoding="utf-8")
    imports = {m.group(1) or m.group(2) for m in re.finditer(
        r"^\s*(?:from ([\w.]+) import|import ([\w.]+))", source, re.M)}
    allowed = {"__future__", "json", "collections.abc", "pathlib"}
    assert imports <= allowed, f"unexpected imports in contract.py: {imports}"


def test_kept_documents_filters_and_merges(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [
        {"kind": "document", "data": {"checksum": "a", "company_id": "tcs",
                                      "gate_decision": "keep",
                                      "doc_type": "annual"}},
        {"kind": "document", "data": {"checksum": "b", "company_id": "wipro",
                                      "gate_decision": "exclude",
                                      "doc_type": "annual"}},
        {"kind": "exclusion", "data": {"candidate_id": "x"}},
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                      encoding="utf-8")
    kept = contract.kept_documents(ledger)
    assert [d["checksum"] for d in kept] == ["a"]
    assert kept[0]["company_id"] == "tcs"


def test_extract_paths_and_readers(tmp_path):
    doc_id = "d" * 64
    pages = tmp_path / "extract" / "page_extracts" / f"{doc_id}.pages.jsonl"
    pages.parent.mkdir(parents=True)
    rows = [{"document_id": doc_id, "pdf_page_index": 1, "page_text": "hi"},
            {"document_id": doc_id, "pdf_page_index": 2, "page_text": ""}]
    pages.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
    assert contract.page_extracts_path(tmp_path, doc_id) == pages
    assert contract.page_text(tmp_path, doc_id, 1) == "hi"
    assert contract.page_text(tmp_path, doc_id, 9) == ""
    assert contract.read_table_records(tmp_path, doc_id) == []


def test_read_freeze_manifest(tmp_path):
    manifest = tmp_path / "freeze_manifest.jsonl"
    row = {"document_id": "d" * 64, "checksum": "d" * 64,
           "company_id": "tcs", "doc_type": "annual", "fiscal_year": "FY2026"}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert contract.read_freeze_manifest(manifest) == [row]
