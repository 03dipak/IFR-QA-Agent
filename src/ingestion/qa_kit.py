"""Generate the stratified QA kit CSV from persisted page extracts.

Uses the existing stratified sampler in src.ingestion.qa (seed 42, ~7%) and
strata of document x page class x statement-like. Writes the kit only; it does
not score pages.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .ledger import DEFAULT_LEDGER_PATH, Ledger
from .models import PageExtract
from .qa import DEFAULT_SAMPLE_RATE, build_qa_kit_rows

FIELDS = ["document_checksum", "company", "pdf_page_index",
          "printed_page_label", "page_class", "page_type", "statement_like",
          "stratum", "page_text"]


def _company_map(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            checksum = (row.get("checksum") or "").strip()
            company = (row.get("company") or "").strip()
            if checksum and company:
                out[checksum] = company
                out[checksum.removesuffix(".pdf")] = company
    return out


def _load_pages(extract_dir: Path, checksum: str) -> list[PageExtract]:
    path = extract_dir / "page_extracts" / f"{checksum}.pages.jsonl"
    if not path.is_file():
        return []
    pages: list[PageExtract] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pages.append(PageExtract.model_validate(json.loads(line)))
    return pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate data/qa/qa_kit.csv")
    parser.add_argument("--storage-dir", default="data/corpus")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--statement-map", default="data/qa/statement_map_proposed.csv")
    parser.add_argument("--out", default="data/qa/qa_kit.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-per-stratum", type=int, default=2)
    args = parser.parse_args(argv)

    extract_dir = Path(args.storage_dir) / "extract"
    ledger = Ledger(args.ledger)
    company_map = _company_map(Path(args.statement_map))
    pages: list[PageExtract] = []
    companies: dict[str, str] = {}
    for row in ledger.rows.values():
        doc_pages = _load_pages(extract_dir, row.checksum)
        if not doc_pages:
            continue
        pages.extend(doc_pages)
        companies[row.checksum] = company_map.get(row.checksum, row.company_id)

    kit, strata = build_qa_kit_rows(
        pages, companies, seed=args.seed, rate=args.rate,
        min_per_stratum=args.min_per_stratum)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kit)
    rate = (len(kit) / len(pages)) if pages else 0.0
    print(f"wrote {len(kit)} rows from {len(pages)} pages "
          f"({rate:.2%}) across {len(strata)} strata to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
