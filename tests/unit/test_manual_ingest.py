"""Tests for hardened manual ingestion (dry-run by default)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tests.unit.test_ingestion import _make_text_pdf

from src.ingestion import manual_ingest


def _pdf(text):
    return _make_text_pdf([[text]])


def _make_inbox(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "infosys").mkdir(parents=True)
    (inbox / "wipro").mkdir(parents=True)
    # same filename in two folders, valid company text
    (inbox / "infosys" / "report_FY2024.pdf").write_bytes(
        _pdf("Infosys Limited annual report FY2024 revenue"))
    (inbox / "wipro" / "report_FY2024.pdf").write_bytes(
        _pdf("Wipro Limited annual report FY2024 revenue"))
    # unparsable name
    (inbox / "infosys" / "randomname.pdf").write_bytes(
        _pdf("Infosys Limited annual report"))
    # wrong-company file
    (inbox / "infosys" / "wrong_FY2024.pdf").write_bytes(
        _pdf("Wipro Limited annual report FY2024"))
    # duplicate hash (same bytes as infosys/report.pdf but new name)
    (inbox / "infosys" / "zz_dup_FY2024.pdf").write_bytes(
        (inbox / "infosys" / "report_FY2024.pdf").read_bytes())
    # out-of-window annual
    (inbox / "infosys" / "old_FY2023.pdf").write_bytes(
        _pdf("Infosys Limited annual report FY2023"))
    return inbox


def test_dry_run_reports_decisions(tmp_path):
    inbox = _make_inbox(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    # minimal coverage matrix
    with matrix.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["company_id", "document_type", "fiscal_period", "url",
                    "status", "source_page", "collected_by", "variant",
                    "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--companies-proposed", "data/seeds/companies_proposed.csv",
        "--company-universe", "data/acquisition/company_universe.csv",
    ])
    assert rc == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    decisions = [r["decision"] for r in dry]
    assert "unparsed" in decisions
    assert "quarantine" in decisions
    assert "duplicate" in decisions
    # out-of-window file still new, core_slice false
    old_rows = [r for r in dry if r["file"] == "old_FY2023.pdf"]
    assert old_rows and old_rows[0]["core_slice"] == "false"
    # no writes
    assert not ledger.exists()
    assert not any(raw.iterdir()) if raw.exists() else True


def test_apply_ingests_valid_and_skips_problems(tmp_path):
    inbox = _make_inbox(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["company_id", "document_type", "fiscal_period", "url",
                    "status", "source_page", "collected_by", "variant",
                    "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--companies-proposed", "data/seeds/companies_proposed.csv",
        "--company-universe", "data/acquisition/company_universe.csv",
        "--apply",
    ])
    assert rc == 0
    rows = [json.loads(l) for l in ledger.read_text().splitlines()]
    docs = [r for r in rows if r["kind"] == "document"]
    # valid new files: infosys/report.pdf, wipro/report.pdf, infosys/old.pdf
    # (randomname unparsed, wrong quarantine, dup duplicate)
    assert len(docs) == 3
    assert all(r["data"]["source"] == "manual" for r in docs)
    # duplicate skipped
    checksums = [r["data"]["checksum"] for r in docs]
    assert len(set(checksums)) == 3


def test_registered_folder_not_unknown(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    (inbox / "hcl" / "AR_1_HCLTECH_FY2024.pdf").write_bytes(
        _pdf("HCL Technologies Limited annual report"))
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as fh:
        csv.writer(fh).writerow(["company_id", "document_type", "fiscal_period",
                                "url", "status", "source_page", "collected_by",
                                "variant", "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
    ])
    assert rc == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    assert all(r["decision"] != "skip_unknown_company" for r in dry)


def test_ticker_mismatch_quarantined(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "jk_cement").mkdir(parents=True)
    (inbox / "jk_cement" / "AR_1_BRITANNIA_2024_2025.pdf").write_bytes(
        _pdf("Britannia Industries annual report"))
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as fh:
        csv.writer(fh).writerow(["company_id", "document_type", "fiscal_period",
                                "url", "status", "source_page", "collected_by",
                                "variant", "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
    ])
    assert rc == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    assert dry and dry[0]["decision"] == "quarantine"
    assert "ticker mismatch" in dry[0]["problem"]
    q = list(csv.DictReader((inbox / "_quarantine.csv").open()))
    assert q and "score" in q[0]


def test_five_page_name_match(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    pages = [[] for _ in range(2)] + [["HCL Technologies Limited annual report FY2024"]]
    (inbox / "hcl" / "AR_1_HCLTECH_FY2024.pdf").write_bytes(_make_text_pdf(pages))
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as fh:
        csv.writer(fh).writerow(["company_id", "document_type", "fiscal_period",
                                "url", "status", "source_page", "collected_by",
                                "variant", "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
    ])
    assert rc == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    assert dry and dry[0]["decision"] == "new"


def test_brr_and_secretarial_classification(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    (inbox / "hcl" / "BRR_SR_2024.pdf").write_bytes(_pdf("HCL sustainability report"))
    (inbox / "hcl" / "Sec_Audit_Report_2024.pdf").write_bytes(_pdf("HCL secretarial audit"))
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as fh:
        csv.writer(fh).writerow(["company_id", "document_type", "fiscal_period",
                                "url", "status", "source_page", "collected_by",
                                "variant", "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
    ])
    assert rc == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    types = {r["file"]: r["doc_type"] for r in dry}
    assert types["BRR_SR_2024.pdf"] == "sustainability_report"
    assert types["Sec_Audit_Report_2024.pdf"] == "secretarial_audit"
    assert all(r["other_report"] == "true" for r in dry)


def test_versions_and_period_from_text(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    (inbox / "hcl" / "AR_1_HCLTECH_A.pdf").write_bytes(
        _pdf("HCL Technologies Limited quarter ended 31st December 2024"))
    (inbox / "hcl" / "AR_2_HCLTECH_U.pdf").write_bytes(
        _pdf("HCL Technologies Limited quarter ended 31st December 2024 unaudited"))
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    with matrix.open("w", newline="") as fh:
        csv.writer(fh).writerow(["company_id", "document_type", "fiscal_period",
                                "url", "status", "source_page", "collected_by",
                                "variant", "core_slice"])
    rc = manual_ingest.main([
        "--inbox", str(inbox),
        "--ledger", str(ledger),
        "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
    ])
    assert rc == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    versions = {r["file"]: r["version"] for r in dry}
    assert versions["AR_1_HCLTECH_A.pdf"] == "A"
    assert versions["AR_2_HCLTECH_U.pdf"] == "U"
    periods = {r["file"]: r["period"] for r in dry}
    assert all(p == "Q3FY2025" for p in periods.values())


def _matrix(path):
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerow(["company_id", "document_type",
                                "fiscal_period", "url", "status", "source_page",
                                "collected_by", "variant", "core_slice"])


def _run(inbox, ledger, raw, matrix, *extra):
    return manual_ingest.main([
        "--inbox", str(inbox), "--ledger", str(ledger), "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
        *extra,
    ])


def test_statement_pages_are_one_indexed_and_skip_auditor():
    texts = [
        # page 1: auditor's report names every statement in the opinion
        "Independent Auditor's Report. We have audited the Standalone "
        "Financial Statements comprising the Balance Sheet, the Statement of "
        "Profit and Loss and the Statement of Cash Flows.",
        "STANDALONE FINANCIAL STATEMENTS\nBalance Sheet as at 31 March 2024",
        "Statement of Profit and Loss for the year ended 31 March 2024",
        "Cash Flow Statement for the year ended 31 March 2024",
    ]
    pages = manual_ingest.detect_statement_pages(texts)
    assert pages["standalone_bs"] == 2
    assert pages["standalone_pl"] == 3
    assert pages["standalone_cf"] == 4


def test_incremental_cache_skips_rehash(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    (inbox / "hcl" / "AR_1_HCLTECH_FY2024.pdf").write_bytes(
        _pdf("HCL Technologies Limited annual report FY2024"))
    ledger, raw, matrix = tmp_path / "l.jsonl", tmp_path / "raw", tmp_path / "m.csv"
    _matrix(matrix)
    assert _run(inbox, ledger, raw, matrix) == 0
    cache = tmp_path / "hash_cache.json"
    assert cache.exists()

    def _boom(_path):
        raise AssertionError("unchanged file must not be re-hashed")

    monkeypatch.setattr(manual_ingest, "sha256_file", _boom)
    assert _run(inbox, ledger, raw, matrix) == 0
    dry = list(csv.DictReader((inbox / "_dry_run.csv").open()))
    assert dry and dry[0]["decision"] == "new"


def test_apply_withdraws_removed_file_and_keeps_raw(tmp_path):
    from src.ingestion.ledger import Ledger

    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    pdf = inbox / "hcl" / "AR_1_HCLTECH_FY2024.pdf"
    pdf.write_bytes(_pdf("HCL Technologies Limited annual report FY2024"))
    ledger, raw, matrix = tmp_path / "l.jsonl", tmp_path / "raw", tmp_path / "m.csv"
    _matrix(matrix)
    assert _run(inbox, ledger, raw, matrix, "--apply") == 0
    checksum = [*Ledger(ledger).rows][0]
    assert (raw / f"{checksum}.pdf").exists()

    pdf.unlink()
    assert _run(inbox, ledger, raw, matrix, "--apply") == 0
    loaded = Ledger(ledger)
    assert loaded.withdrawn
    assert loaded.withdrawn[-1].reason == "file removed from inbox"
    assert loaded.withdrawn[-1].raw_kept is True
    assert (raw / f"{checksum}.pdf").exists()


def test_same_period_new_hash_becomes_version_and_supersedes(tmp_path):
    from src.ingestion.ledger import Ledger

    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    pdf = inbox / "hcl" / "AR_1_HCLTECH_FY2024.pdf"
    pdf.write_bytes(_pdf("HCL Technologies Limited annual report FY2024 v1"))
    ledger, raw, matrix = tmp_path / "l.jsonl", tmp_path / "raw", tmp_path / "m.csv"
    _matrix(matrix)
    assert _run(inbox, ledger, raw, matrix, "--apply") == 0
    first = [*Ledger(ledger).rows][0]

    pdf.write_bytes(_pdf("HCL Technologies Limited annual report FY2024 v2"))
    assert _run(inbox, ledger, raw, matrix, "--apply") == 0
    loaded = Ledger(ledger)
    docs = list(loaded.rows)
    assert len(docs) == 2
    assert loaded.revisions
    assert loaded.revisions[-1].from_document_id == first
    assert loaded.revisions[-1].to_document_id in docs


def _ledger_docs(ledger):
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    return [r for r in rows if r["kind"] == "document"], \
           [r for r in rows if r["kind"] == "withdrawn"]


def test_apply_resumes_and_scopes_by_company_and_batch(tmp_path):
    inbox = _make_inbox(tmp_path)
    ledger, raw, matrix = tmp_path / "l.jsonl", tmp_path / "raw", tmp_path / "m.csv"
    _matrix(matrix)

    # --company: only infosys (2 ingestable docs) is touched.
    assert _run(inbox, ledger, raw, matrix, "--apply", "--company", "infosys") == 0
    docs, _ = _ledger_docs(ledger)
    assert docs and all(d["data"]["company_id"] == "infosys" for d in docs)
    n_infosys = len(docs)

    # Rerun is a no-op: the ledger is the resume point.
    assert _run(inbox, ledger, raw, matrix, "--apply", "--company", "infosys") == 0
    docs, _ = _ledger_docs(ledger)
    assert len(docs) == n_infosys

    # --batch 2 --batch-size 1 selects wipro (sorted after infosys).
    assert _run(inbox, ledger, raw, matrix, "--apply",
                "--batch", "2", "--batch-size", "1") == 0
    docs, withdrawn = _ledger_docs(ledger)
    assert len(docs) == n_infosys + 1
    assert any(d["data"]["company_id"] == "wipro" for d in docs)

    # A filtered run must not withdraw the company it did not scan.
    assert _run(inbox, ledger, raw, matrix, "--apply", "--company", "infosys") == 0
    _, withdrawn = _ledger_docs(ledger)
    assert withdrawn == []

    # Out-of-range batch is an explicit error, not a silent no-op.
    assert _run(inbox, ledger, raw, matrix, "--batch", "99") == 2


def test_apply_after_dry_run_still_writes(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "hcl").mkdir(parents=True)
    (inbox / "hcl" / "AR_1_HCLTECH_FY2024.pdf").write_bytes(
        _pdf("HCL Technologies Limited annual report FY2024"))
    ledger, raw, matrix = tmp_path / "l.jsonl", tmp_path / "raw", tmp_path / "m.csv"
    _matrix(matrix)

    # Dry run populates the hash cache with a "new" row...
    assert _run(inbox, ledger, raw, matrix) == 0
    assert not ledger.exists()

    # ...but a later --apply must not just replay that cached row.
    assert _run(inbox, ledger, raw, matrix, "--apply") == 0
    docs, _ = _ledger_docs(ledger)
    assert len(docs) == 1
    assert (raw / f"{docs[0]['data']['checksum']}.pdf").exists()

    # And the applied doc is skipped on the next apply.
    assert _run(inbox, ledger, raw, matrix, "--apply") == 0
    docs, _ = _ledger_docs(ledger)
    assert len(docs) == 1



def test_copy_verified(tmp_path, monkeypatch):
    import hashlib

    data = _pdf("annual report probe")
    src = tmp_path / "src.pdf"
    src.write_bytes(data)
    chk = hashlib.sha256(data).hexdigest()
    dest = tmp_path / "raw" / f"{chk}.pdf"

    # intact copy -> True once, then False (already intact)
    assert manual_ingest._copy_verified(src, dest, chk) is True
    assert manual_ingest._copy_verified(src, dest, chk) is False
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == chk

    # pre-existing corrupt copy is replaced, not skipped
    dest.write_bytes(data[: len(data) // 2])
    assert manual_ingest._copy_verified(src, dest, chk) is True
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == chk

    # a copy that always truncates must fail loudly and leave no bad file
    dest.write_bytes(data[: len(data) // 3])

    def truncating_copy(s, d, *a, **kw):
        d.write_bytes(src.read_bytes()[: len(src.read_bytes()) // 2])

    monkeypatch.setattr(manual_ingest.shutil, "copy2", truncating_copy)
    try:
        manual_ingest._copy_verified(src, dest, chk)
    except ValueError as exc:
        assert "sha256 mismatch" in str(exc)
    else:
        raise AssertionError("expected sha256 mismatch failure")
    assert not dest.exists()


# ---------------------------------------------------------------------------
# Part A: content-based period detection (dry-run)
# ---------------------------------------------------------------------------

def _content_run(tmp_path, company, name, text, *extra):
    import re as _re
    text = "\f".join(
        _re.split(r"-----\s*PAGE\s+\d+\s*-----", text))
    inbox = tmp_path / "inbox"
    folder = inbox / company
    folder.mkdir(parents=True)
    (folder / name).write_bytes(_pdf(text))
    ledger = tmp_path / "ledger.jsonl"
    raw = tmp_path / "raw"
    matrix = tmp_path / "matrix.csv"
    _matrix(matrix)
    args = [
        "--inbox", str(inbox), "--ledger", str(ledger), "--raw", str(raw),
        "--coverage-matrix", str(matrix),
        "--company-registry", "data/acquisition/company_registry.csv",
        *extra,
    ]
    assert manual_ingest.main(args) == 0
    dry = {r["file"]: r for r in csv.DictReader((inbox / "_dry_run.csv").open())}
    unparsed = list(csv.DictReader((inbox / "_unparsed.csv").open()))
    return dry[name], unparsed


def test_content_results_statement_parsed(tmp_path):
    row, _ = _content_run(
        tmp_path, "infosys", "NSEUFR_something.pdf",
        "Infosys Limited: Statement of Unaudited Financial Results for the "
        "quarter ended 30th September 2024 (Standalone)")
    assert row["period"] == "Q2FY2025"
    assert row["doc_type"] == "quarterly"
    assert row["decision"] == "new"
    assert row["period_confidence"] == "high"
    assert "quarter ended" in row["snippet"].lower()
    assert row["variant"] == "standalone"


def test_content_outcome_letter_is_disclosure(tmp_path):
    row, _ = _content_run(
        tmp_path, "infosys", "Outcome.pdf",
        "Infosys Limited - Outcome of Board Meeting: approved the financial "
        "results for the quarter ended 30 June 2025")
    assert row["period"] == "Q1FY2026"
    assert row["doc_type"] == "disclosure"
    assert row["decision"] == "new"
    assert row["period_confidence"] == "high"


def test_content_half_year_file(tmp_path):
    row, _ = _content_run(
        tmp_path, "infosys", "HalfYearUpdate.pdf",
        "Infosys Limited Financial Results for the half year ended "
        "30th September 2024")
    assert row["period"] == "Q2FY2025"
    assert row["doc_type"] == "quarterly"
    assert row["decision"] == "new"


def test_content_letter_date_after_quarter_ignored(tmp_path):
    # letter dated 18 January 2025 must not override the quarter end
    row, _ = _content_run(
        tmp_path, "infosys", "intimation_letter.pdf",
        "Infosys Limited Outcome of Board Meeting: financial results for the "
        "quarter ended 31st December 2024. Letter dated January 18, 2025.")
    assert row["period"] == "Q3FY2025"
    assert row["decision"] == "new"


def test_content_two_conflicting_dates_stay_unparsed(tmp_path):
    row_, unparsed = _content_run(
        tmp_path, "infosys", "combined_period.pdf",
        "Infosys Limited financial results for the quarter ended 30 June "
        "2025 and financial results for the quarter ended 30 September 2025")
    assert row_["decision"] == "unparsed"
    assert row_["period"] == ""
    assert row_["period_confidence"] == "low"
    matches = [u for u in unparsed if u["filename"] == "combined_period.pdf"]
    assert matches and "tied top candidates" in matches[0]["reason"]
    assert matches[0]["suggested_period"] in ("Q1FY2026", "Q2FY2026")
    assert matches[0]["candidates"].count("=") >= 2
    assert matches[0]["dates_found"].count("2025") >= 1


def test_content_no_date_stays_unparsed(tmp_path):
    row_, unparsed = _content_run(
        tmp_path, "infosys", "weekend_note.pdf",
        "Infosys Limited Outcome of Board Meeting was recorded.")
    assert row_["decision"] == "unparsed"
    matches = [u for u in unparsed if u["filename"] == "weekend_note.pdf"]
    assert matches and matches[0]["reason"]


def test_content_variant_combined_requires_both_in_title(tmp_path):
    row, _ = _content_run(
        tmp_path, "infosys", "annualset_Standalone.pdf",
        "Infosys Limited Standalone and Consolidated Financial Results for the "
        "quarter ended 31st December 2024")
    assert row["variant"] == "combined"


def test_content_table_comparative_headers_ignored(tmp_path):
    # results table: previous-quarter and prior-year columns are comparatives
    row_, _ = _content_run(
        tmp_path, "infosys", "results_table.pdf",
        "Infosys Limited Statement of Unaudited Financial Results for the "
        "quarter ended 30th September 2024 (Standalone)\n"
        "Particulars  Quarter Ended 30-06-2024  Quarter Ended 30-09-2023  "
        "Quarter Ended 30-06-2023\n"
        "Revenue from operations  100  90  80")
    assert row_["period"] == "Q2FY2025"
    assert row_["period_confidence"] == "high"
    assert row_["decision"] == "new"
    assert "30th September 2024" in row_["snippet"]


def test_content_no_title_date_falls_back_to_latest_medium(tmp_path):
    # only table-column-header dates exist: no clean phrase match, the file
    # stays unparsed with the latest header date as a suggested_period
    text_page1 = ("Infosys Limited Statement of Financial Results "
                  "(Standalone)\nParticulars  Quarter Ended 30-06-2024  "
                  "Revenue from operations  100  90")
    text_page2 = "Particulars  Quarter Ended 30-09-2024  Revenue  110"
    row_, unparsed = _content_run(
        tmp_path, "infosys", "no_title_date.pdf",
        text_page1 + "\f" + text_page2)
    assert row_["decision"] == "unparsed"
    matches = [u for u in unparsed if u["filename"] == "no_title_date.pdf"]
    assert matches and matches[0]["suggested_period"] == "Q2FY2025"


def test_content_snapshot_lupin_q1fy2025(tmp_path):
    text = (Path("tests/fixtures/period_text")
            / "lupin__NSEQuickResults0682024.txt").read_text()
    row_, unparsed = _content_run(
        tmp_path, "lupin", "NSEQuickResults0682024.pdf", text)
    assert row_["period"] == "Q1FY2025"
    assert row_["doc_type"] == "quarterly"
    assert row_["variant"] == "combined"
    assert row_["decision"] == "new"
    assert row_["period_confidence"] == "high"
    assert row_["covers_full_year"] == "false"


def test_content_snapshot_tvs_q3fy2024(tmp_path):
    text = (Path("tests/fixtures/period_text")
            / "tvs_motor__TVSMNSEBSERESULTSQ3.txt").read_text()
    row_, _ = _content_run(
        tmp_path, "tvs_motor", "TVSMNSEBSERESULTSQ3.pdf", text)
    assert row_["period"] == "Q3FY2024"
    assert row_["doc_type"] == "quarterly"
    assert row_["variant"] == "combined"
    assert row_["decision"] == "new"
    assert row_["period_confidence"] == "high"


def test_content_snapshot_torrent_q4fy2024_annual(tmp_path):
    text = (Path("tests/fixtures/period_text")
            / "torrent_pharma__Financials.txt").read_text()
    row_, _ = _content_run(
        tmp_path, "torrent_pharma", "Financials.pdf", text)
    assert row_["period"] == "Q4FY2024"
    assert row_["doc_type"] == "annual"
    assert row_["variant"] == "combined"
    assert row_["decision"] == "new"
    assert row_["period_confidence"] == "high"
    assert row_["covers_full_year"] == "true"


def test_content_scoring_direct_unit():
    from src.ingestion.period_detection import detect_period_from_content
    # same-date reinforcement: three letter mentions of one quarter end
    letter = ("Delta Corp letter dated January 24, 2024. Sub: results for "
              "the quarter ended 31st December 2023. We enclose the results "
              "for the quarter ended 31st December 2023. Also, review of "
              "the quarter ended 31 December 2023 is enclosed.")
    det = detect_period_from_content(letter)
    assert det["period"] == "Q3FY2024"
    assert det["confidence"] == "high"  # 2x rule over weight-0 runner
    # tie: two equally mentioned quarter ends -> unparsed, not a guess
    tie = ("Epsilon Ltd letter dated 20 October 2025: results for the quarter "
           "ended 30 June 2025 and results for the quarter ended 30 "
           "September 2025 are enclosed.")
    det2 = detect_period_from_content(tie)
    assert det2["period"] is None and det2["suggested_period"] in ("Q1FY2026", "Q2FY2026")


def test_content_snapshot_coal_acct_q3fy2024(tmp_path):
    text = (Path("tests/fixtures/period_text")
            / "coal_india__acct.txt").read_text()
    row_, _ = _content_run(
        tmp_path, "coal_india", "acct.pdf", text)
    assert row_["period"] == "Q3FY2024"
    assert row_["doc_type"] == "quarterly"
    assert row_["variant"] == "combined"
    assert row_["decision"] == "new"
    assert row_["period_confidence"] == "high"


def test_content_snapshot_coal_signedaccts_q2fy2025(tmp_path):
    text = (Path("tests/fixtures/period_text")
            / "coal_india__signedaccts.txt").read_text()
    row_, _ = _content_run(
        tmp_path, "coal_india", "signedaccts.pdf", text)
    assert row_["period"] == "Q2FY2025"
    assert row_["doc_type"] == "quarterly"
    assert row_["variant"] == "combined"
    assert row_["decision"] == "new"
    assert row_["period_confidence"] == "high"


def test_content_snapshot_aurobindo_1002_q3fy2024(tmp_path):
    text = (Path("tests/fixtures/period_text")
            / "aurobindo_pharma__LtrToSEsOutcomeOfBM10022024.txt").read_text()
    row_, _ = _content_run(
        tmp_path, "aurobindo_pharma", "LtrToSEsOutcomeOfBM10022024.pdf", text)
    assert row_["period"] == "Q3FY2024"
    assert row_["variant"] == "combined"
    assert row_["decision"] == "new"
    assert row_["period_confidence"] == "high"


def test_content_two_title_dates_conflict(tmp_path):
    row_, _ = _content_run(
        tmp_path, "infosys", "two_titles.pdf",
        "Infosys Limited financial results for the quarter ended 30 June "
        "2025 and standalone financial results for the quarter ended "
        "30 September 2025")
    assert row_["decision"] == "unparsed"
