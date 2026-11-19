"""Read-only view of persisted page/table JSONL (no writes, no stored-file change)."""

from __future__ import annotations

import json

from src.ingestion.view import render_view


def test_render_view_prints_page_text_and_tables(tmp_path):
    doc = "abc123"
    (tmp_path / f"{doc}.pages.jsonl").write_text(json.dumps(
        {"document_id": doc, "pdf_page_index": 3, "page_text": "Cash flows were fine", "page_type": "statement",
         "printed_page_label": None, "extraction_hash": "h", "extraction_run_id": "r", "page_class": "TEXT",
         "usable_as_evidence": True, "has_table": True, "ufffd_count": 0, "character_count": 18,
         "extraction_status": "extracted"}
    ) + "\n", encoding="utf-8")
    (tmp_path / f"{doc}.tables.jsonl").write_text(json.dumps(
        {"table_logical_key": "k", "page_start": 3, "page_end": 3,
         "columns": [{"header_raw": "Particulars"}, {"header_raw": "FY24"}],
         "rows": [{"cells": [{"raw": "Profit"}, {"raw": "124"}]}]}
    ) + "\n", encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())
    out = render_view(tmp_path, doc, 3)
    assert "Cash flows were fine" in out
    assert "Particulars" in out and "Profit" in out and "124" in out
    assert "(no tables on this page)" not in out
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after  # no files created, none changed
