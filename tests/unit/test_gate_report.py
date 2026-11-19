"""Tests for the read-only corpus gate report (LLD-45 harness)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.ingestion import gate_report
from src.ingestion.gate_report import _anomaly_flags
from src.ingestion.models import PageClass, PageExtract, PageType


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_pdf_bytes():
    from tests.unit.test_ingestion import _make_text_pdf

    stmt = "Balance Sheet total assets 100 revenue 200 liabilities 50 cash flow 10 profit 5"
    return _make_text_pdf(
        [
            [stmt],
            ["Statement of Profit and Loss revenue 100 expenses 50 profit 25 tax 5 net 20"],
            ["Cash Flow operating 100 investing 20 financing 10 net 5 total 5"],
        ]
    )


def test_read_only_does_not_modify_corpus_or_ledger(tmp_path, capsys):
    storage = tmp_path / "raw"
    storage.mkdir()
    pdf = storage / "sample.pdf"
    pdf.write_bytes(_make_pdf_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "entries": {}}))

    before = {p.name: p.read_bytes() for p in storage.iterdir()}
    ledger = tmp_path / "ledger.jsonl"
    assert not ledger.exists()

    rc = gate_report.main(
        [
            "--storage-dir",
            str(storage),
            "--manifest",
            str(manifest),
            "--ledger",
            str(ledger),
        ]
    )
    assert rc == 0

    after = {p.name: p.read_bytes() for p in storage.iterdir()}
    assert before == after
    assert not ledger.exists()


def test_summary_reconciles(tmp_path, capsys):
    storage = tmp_path / "raw"
    storage.mkdir()
    (storage / "sample.pdf").write_bytes(_make_pdf_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "entries": {}}))

    rc = gate_report.main(
        [
            "--storage-dir",
            str(storage),
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "summary documents=1" in out
    assert "kept=1" in out
    assert "skipped=0" in out


def test_manifest_pass_fail_and_skipped(tmp_path, capsys):
    storage = tmp_path / "raw"
    storage.mkdir()
    data = _make_pdf_bytes()
    pdf = storage / "sample.pdf"
    pdf.write_bytes(data)
    cs = _sha256_bytes(data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    cs: {
                        "filename": "sample.pdf",
                        "expected_decision": "keep",
                        "expected_rule": "none",
                        "expected_page_class_counts": {
                            "blank": 0,
                            "garbled": 0,
                            "image_dominant": 0,
                            "image_only": 0,
                            "low_text": 0,
                            "text": 3,
                        },
                        "notes": "seeded",
                        "reviewed_by": "pending visual check",
                        "reviewed_on": "2026-11-03",
                    },
                    "0" * 64: {
                        "filename": "missing.pdf",
                        "expected_decision": "keep",
                        "expected_rule": "none",
                        "expected_page_class_counts": {},
                        "notes": "missing",
                        "reviewed_by": "pending visual check",
                        "reviewed_on": "2026-11-03",
                    },
                },
            }
        )
    )

    rc = gate_report.main(
        [
            "--storage-dir",
            str(storage),
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "pass=1" in out
    assert "skipped=1" in out


def test_checksum_mismatch_reported(tmp_path, capsys):
    storage = tmp_path / "raw"
    storage.mkdir()
    (storage / "sample.pdf").write_bytes(_make_pdf_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "f" * 64: {
                        "filename": "sample.pdf",
                        "expected_decision": "keep",
                        "expected_rule": "none",
                        "expected_page_class_counts": {},
                        "notes": "wrong checksum",
                        "reviewed_by": "pending visual check",
                        "reviewed_on": "2026-11-03",
                    },
                },
            }
        )
    )

    rc = gate_report.main(
        [
            "--storage-dir",
            str(storage),
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "CHECKSUM_MISMATCH" in out or "checksum_mismatch=1" in out


def test_anomaly_excluded_with_statements(monkeypatch):
    pages = [
        PageExtract(
            document_id="d",
            extraction_run_id="r",
            pdf_page_index=i,
            page_text="Balance Sheet assets 100 revenue 200 liabilities 50 cash 10 profit 5",
            extraction_hash=f"h{i}",
            page_type=PageType.STATEMENT,
            page_class=PageClass.TEXT,
            usable_as_evidence=True,
            character_count=80,
        )
        for i in range(1, 4)
    ] + [
        PageExtract(
            document_id="d",
            extraction_run_id="r",
            pdf_page_index=4,
            page_text="",
            extraction_hash="h4",
            page_type=PageType.SCAN,
            page_class=PageClass.IMAGE_ONLY,
            usable_as_evidence=False,
            character_count=0,
        )
        for _ in range(10)
    ]
    # image share > 50% -> excluded, statement-like >= 3
    monkeypatch.setattr(
        "src.ingestion.gate_report.scanned_document_rules", lambda p, d=None: ["image_share"]
    )
    flags = _anomaly_flags(Path("x.pdf"), pages, "exclude", ["image_share"], 3, "annual", "PASS")
    assert "excluded_with_statements" in flags


def test_anomaly_kept_below_minimum(monkeypatch):
    pages = [
        PageExtract(
            document_id="d",
            extraction_run_id="r",
            pdf_page_index=1,
            page_text="x",
            extraction_hash="h1",
            page_type=PageType.PROSE,
            page_class=PageClass.TEXT,
            usable_as_evidence=True,
            character_count=10,
        )
    ]
    flags = _anomaly_flags(Path("x.pdf"), pages, "keep", [], 0, "annual", "PASS")
    assert "kept_below_minimum" in flags


def test_anomaly_high_image_share():
    pages = [
        PageExtract(
            document_id="d",
            extraction_run_id="r",
            pdf_page_index=i,
            page_text="Balance Sheet assets 100 revenue 200 liabilities 50 cash 10 profit 5",
            extraction_hash=f"h{i}",
            page_type=PageType.STATEMENT,
            page_class=PageClass.TEXT,
            usable_as_evidence=True,
            character_count=80,
        )
        for i in range(1, 4)
    ] + [
        PageExtract(
            document_id="d",
            extraction_run_id="r",
            pdf_page_index=4,
            page_text="",
            extraction_hash="h4",
            page_type=PageType.SCAN,
            page_class=PageClass.IMAGE_ONLY,
            usable_as_evidence=False,
            character_count=0,
        )
    ]
    flags = _anomaly_flags(Path("x.pdf"), pages, "keep", [], 3, "annual", "PASS")
    assert "high_image_share" in flags


def test_anomaly_high_ufffd_share():
    page = PageExtract(
        document_id="d",
        extraction_run_id="r",
        pdf_page_index=1,
        page_text="x" * 100,
        extraction_hash="h1",
        page_type=PageType.PROSE,
        page_class=PageClass.TEXT,
        usable_as_evidence=True,
        character_count=100,
        ufffd_count=10,
    )
    flags = _anomaly_flags(Path("x.pdf"), [page], "keep", [], 0, "disclosure", "PASS")
    assert "high_ufffd_share" in flags


def test_anomaly_zero_pages():
    flags = _anomaly_flags(
        Path("x.pdf"), [], "exclude", ["statements_not_found"], 0, "annual", "PASS"
    )
    assert "zero_pages" in flags


def test_anomaly_manifest_mismatch():
    page = PageExtract(
        document_id="d",
        extraction_run_id="r",
        pdf_page_index=1,
        page_text="x",
        extraction_hash="h1",
        page_type=PageType.PROSE,
        page_class=PageClass.TEXT,
        usable_as_evidence=True,
        character_count=1,
    )
    flags = _anomaly_flags(Path("x.pdf"), [page], "keep", [], 0, "disclosure", "FAIL")
    assert "manifest_mismatch" in flags


def test_unreadable_pdf_flagged(tmp_path, capsys):
    storage = tmp_path / "raw"
    storage.mkdir()
    (storage / "bad.pdf").write_bytes(b"not a pdf")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "entries": {}}))

    rc = gate_report.main(
        [
            "--storage-dir",
            str(storage),
            "--manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "unreadable_pdf" in out or "anomalies=unreadable_pdf" in out


def test_config_hash_invalidates_on_threshold_change():
    from src.ingestion import gate_report as gr

    hash_before = gr.compute_config_hash()
    gr.ANOMALY_HIGH_IMAGE_SHARE_LIMIT = 0.99
    hash_after = gr.compute_config_hash()
    assert hash_before != hash_after
