"""Extract pending manual-inbox PDFs into page/table stores; idempotent, resumable."""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from collections import Counter
from pathlib import Path

from src.ingestion.extractor import extract_pages
from src.ingestion.storage import LocalPageExtractStore, LocalTableRecordStore
from src.tables.extractor import extract_tables

LEDGER = Path("data/corpus/ledger.jsonl")
RAW = Path("data/corpus/raw")
EXTRACT = Path("data/corpus/extract")
FAILURE_REASONS = (
    "missing_raw", "unreadable_pdf", "corrupt_pdf", "password_protected", "parse_error",
)


def load_ledger_documents(ledger_path: Path) -> dict[str, dict]:
    """Latest row per checksum (ledger is append-only; last write wins)."""
    docs: dict[str, dict] = {}
    with ledger_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "document":
                data = record.get("data") or {}
                checksum = data.get("checksum")
                if checksum:
                    docs[checksum] = data
    return docs


def classify_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "password" in message or "encrypt" in message:
        return "password_protected"
    if "eof" in message or "truncated" in message or "corrupt" in message:
        return "corrupt_pdf"
    if "no such file" in message:
        return "missing_raw"
    return "parse_error"


def pending_manual_docs(docs: dict[str, dict], company: str | None) -> list[dict]:
    out = []
    for doc in docs.values():
        if doc.get("extraction_status") != "pending":
            continue
        if doc.get("source") != "manual":
            continue
        if company and doc.get("company_id") != company:
            continue
        out.append(doc)
    return sorted(out, key=lambda d: (d.get("company_id") or "", d.get("checksum") or ""))


def extract_one(doc: dict, page_store: LocalPageExtractStore,
                table_store: LocalTableRecordStore, freeze_version: str) -> tuple[str, int]:
    """Extract one doc; returns (status, page_count) or raises with reason."""
    checksum = doc["checksum"]
    pdf_path = RAW / f"{checksum}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("missing raw PDF")
    data = pdf_path.read_bytes()
    run_id = str(uuid.uuid4())
    pages = extract_pages(data, checksum, run_id)
    page_store.put(checksum, pages)
    tables = extract_tables(pages, checksum, run_id, doc.get("extraction_config_hash") or "cfg-0",
                            freeze_version=freeze_version)
    table_store.put(checksum, tables)
    return "extracted", len(pages)


def updated_row(doc: dict, page_count: int, run_id: str) -> dict:
    row = dict(doc)
    row["extraction_status"] = "extracted"
    row["extraction_run_id"] = run_id
    row["page_count"] = page_count
    return row


def coverage_table(inbox: Path, docs: dict[str, dict],
                   page_store: LocalPageExtractStore) -> Counter:
    rows: Counter = Counter()
    for company_dir in sorted(p for p in inbox.iterdir() if p.is_dir()):
        code = company_dir.name
        for pdf in sorted(company_dir.glob("*.pdf")):
            import hashlib
            checksum = hashlib.sha256(pdf.read_bytes()).hexdigest()
            doc = docs.get(checksum)
            if doc is None:
                rows[(code, "not_in_ledger")] += 1
                continue
            state = "extracted" if page_store._path(checksum).exists() else "pending"
            rows[(code, doc.get("doc_type") or "?")] += 0  # keep key types stable
            rows[(code, f"{doc.get('doc_type')}/{state}")] += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract pending manual-inbox PDFs")
    parser.add_argument("--company", default=None, help="restrict to one company code")
    parser.add_argument("--limit", type=int, default=None, help="max PDFs this run")
    parser.add_argument("--freeze-version", default="v2")
    parser.add_argument("--apply", action="store_true", help="write ledger status rows")
    parser.add_argument("--report", default="data/manual_inbox/_extract_report.csv")
    args = parser.parse_args()

    started = time.time()
    docs = load_ledger_documents(LEDGER)
    page_store = LocalPageExtractStore(EXTRACT / "page_extracts")
    table_store = LocalTableRecordStore(EXTRACT / "table_records")
    todo = pending_manual_docs(docs, args.company)
    if args.limit:
        todo = todo[: args.limit]
    print(f"pending manual docs: {len(todo)} (apply={args.apply})")

    report_rows: list[dict] = []
    done = failed = 0
    for index, doc in enumerate(todo, 1):
        checksum = doc["checksum"]
        if page_store._path(checksum).exists():
            existing_pages = sum(1 for _ in page_store._path(checksum).open(encoding="utf-8"))
            if args.apply:
                reconcile = updated_row(doc, existing_pages, "reconcile-existing")
                with LEDGER.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"kind": "document", "data": reconcile}) + "\n")
            done += 1
            continue
        try:
            status, page_count = extract_one(doc, page_store, table_store, args.freeze_version)
            if args.apply:
                run_id = str(uuid.uuid4())
                with LEDGER.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"kind": "document",
                                         "data": updated_row(doc, page_count, run_id)}) + "\n")
            done += 1
            print(f"[{index}/{len(todo)}] {doc.get('company_id')} "
                  f"{doc.get('doc_type')} {doc.get('fiscal_year')} -> {status} pages={page_count}")
        except Exception as exc:  # noqa: BLE001 — per-PDF failure must not stop the run
            reason = classify_failure(exc)
            failed += 1
            report_rows.append({"company_id": doc.get("company_id"), "checksum": checksum,
                                "doc_type": doc.get("doc_type"),
                                "fiscal_year": doc.get("fiscal_year"), "reason": reason,
                                "detail": str(exc)[:200]})
            print(f"[{index}/{len(todo)}] FAILED {doc.get('company_id')} {checksum[:12]}: {reason}")

    with Path(args.report).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["company_id", "checksum", "doc_type",
                                                "fiscal_year", "reason", "detail"])
        writer.writeheader()
        writer.writerows(report_rows)

    still_pending = sum(1 for d in pending_manual_docs(load_ledger_documents(LEDGER), args.company)
                        if not page_store._path(d["checksum"]).exists())
    print(f"done={done} failed={failed} still_pending={still_pending} "
          f"elapsed_s={int(time.time() - started)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
