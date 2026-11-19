"""Propose balance-sheet / profit-and-loss / cash-flow spans from page text.

Implements the statement-map step of lld_tables.txt: statement starts are
chosen as the BS/PL/CF heading triple with the smallest page spread; the end
page extends only over statement-like continuation pages within the same
reporting basis. A span is flagged ``check`` when the start spread exceeds
six pages or a span exceeds eight pages.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from ..ingestion.extractor import is_statement_like
from ..ingestion.ledger import DEFAULT_LEDGER_PATH, Ledger

STATEMENT_ORDER = ("BS", "PL", "CF")
MAX_START_SPREAD = 6
MAX_SPAN = 8

_BASIS = {
    "consolidated": re.compile(r"\bconsolidated\b", re.I),
    "standalone": re.compile(r"\bstandalone\b", re.I),
}
_HEADINGS = {
    "BS": re.compile(r"\bbalance\s+sheet\b", re.I),
    "PL": re.compile(r"\bprofit\s*(?:and|&)\s*loss\b", re.I),
    "CF": re.compile(r"\b(?:cash\s*flow\s+statement|statement\s+of\s+cash\s*flows?)\b", re.I),
}
_OTHER_HEADING = re.compile(
    r"\bstatement\s+of\s+changes\s+in\s+equity\b|\bnotes\s+forming\s+part\b|"
    r"\bsignificant\s+accounting\s+polic",
    re.I,
)


@dataclass
class ProposedSpan:
    company: str
    checksum: str
    basis: str
    statement: str
    start_pdf_page: int
    end_pdf_page: int
    status: str

    def as_row(self) -> dict[str, str | int]:
        return {
            "company": self.company,
            "checksum": self.checksum,
            "basis": self.basis,
            "statement": self.statement,
            "start_pdf_page": self.start_pdf_page,
            "end_pdf_page": self.end_pdf_page,
            "status": self.status,
        }


def _basis_of(text: str) -> str | None:
    flat = " ".join(text.split())
    for window in (flat[:500], flat):
        low = window.lower()
        has_c = bool(_BASIS["consolidated"].search(low))
        has_s = bool(_BASIS["standalone"].search(low))
        if has_c and not has_s:
            return "consolidated"
        if has_s and not has_c:
            return "standalone"
    return None


def _headings_at(text: str, region: str) -> set[str]:
    flat = " ".join(text.split())
    if _OTHER_HEADING.search(flat[:500]) or _OTHER_HEADING.search(flat[-300:]):
        return set()
    window = flat[-300:] if region == "footer" else flat[:500]
    return {statement for statement in STATEMENT_ORDER
            if _HEADINGS[statement].search(window)}


def _table_signature(text: str) -> bool:
    head = " ".join(text.split())[:500].lower()
    return any(token in head for token in (
        "as at", "year ended", "particulars", "cash flow", "assets",
        "liabilities", "revenue", "income", "expenses"))


def _candidate_starts(pages: dict[int, str], basis: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {key: [] for key in STATEMENT_ORDER}
    previous_head: set[str] = set()
    previous_footer: set[str] = set()
    for index in sorted(pages):
        text = pages[index]
        if _basis_of(text) != basis or not is_statement_like(text):
            previous_head = set()
            previous_footer = set()
            continue
        flat = " ".join(text.split())
        auditor = "independent auditor" in flat[:500].lower()
        head = set() if auditor else _headings_at(text, "head")
        footer = (_headings_at(text, "footer")
                  if _table_signature(text) and not auditor else set())
        for statement in STATEMENT_ORDER:
            from_head = statement in head and statement not in previous_head
            from_footer = statement in footer and statement not in previous_footer
            if from_head or from_footer:
                out[statement].append(index)
        previous_head = head
        previous_footer = footer
    return out


def _choose_triple(starts: dict[str, list[int]]) -> dict[str, int] | None:
    if any(not starts[key] for key in STATEMENT_ORDER):
        return None
    best: tuple[int, int, tuple[int, int, int]] | None = None
    for bs in starts["BS"]:
        for pl in starts["PL"]:
            for cf in starts["CF"]:
                triple = (bs, pl, cf)
                spread = max(triple) - min(triple)
                key = (spread, sum(triple), triple)
                if best is None or key < best:
                    best = key
    if best is None:
        return None
    bs, pl, cf = best[2]
    return {"BS": bs, "PL": pl, "CF": cf}


def _is_continuation(text: str, statement: str, basis: str) -> bool:
    page_basis = _basis_of(text)
    if page_basis is not None and page_basis != basis:
        return False
    head = " ".join(text.split())[:500]
    if _OTHER_HEADING.search(head):
        return False
    for other in STATEMENT_ORDER:
        if other != statement and _HEADINGS[other].search(head):
            return False
    head = head[:400].lower()
    if statement == "CF" and "cash flow" in head:
        return True
    if "particulars" in head or "as at" in head:
        return True
    return is_statement_like(text)


def _end_page(pages: dict[int, str], start: int, statement: str, basis: str,
              next_start: int | None) -> int:
    end = start
    for index in range(start + 1, start + MAX_SPAN + 2):
        if next_start is not None and index >= next_start:
            break
        text = pages.get(index)
        if text is None or not _is_continuation(text, statement, basis):
            break
        end = index
    return end


def propose_spans(company: str, checksum: str,
                  pages: list[tuple[int, str]]) -> list[ProposedSpan]:
    page_map = {index: text for index, text in pages}
    spans: list[ProposedSpan] = []
    for basis in ("consolidated", "standalone"):
        starts = _candidate_starts(page_map, basis)
        triple = _choose_triple(starts)
        if triple is None:
            continue
        ordered = sorted(triple.values())
        spread = ordered[-1] - ordered[0]
        for statement in STATEMENT_ORDER:
            start = triple[statement]
            later = [value for value in ordered if value > start]
            next_start = later[0] if later else None
            end = _end_page(page_map, start, statement, basis, next_start)
            needs_check = spread > MAX_START_SPREAD or (end - start) > MAX_SPAN
            status = "check" if needs_check else "proposed"
            spans.append(ProposedSpan(
                company=company, checksum=checksum, basis=basis,
                statement=statement, start_pdf_page=start,
                end_pdf_page=end, status=status))
    if not spans:
        from .quarterly_map import quarterly_spans

        spans = quarterly_spans(company, checksum, pages)
    return spans


def load_pages(extract_dir: Path, checksum: str) -> list[tuple[int, str]]:
    key = checksum[:-4] if checksum.endswith(".pdf") else checksum
    path = extract_dir / "page_extracts" / f"{key}.pages.jsonl"
    if not path.is_file():
        return []
    import json

    out: list[tuple[int, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                payload = json.loads(line)
                out.append((int(payload["pdf_page_index"]), payload["page_text"]))
    return out


def _read_existing(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if not row.get("checksum") and row.get("document_checksum"):
            row["checksum"] = row["document_checksum"]
        row.pop("document_checksum", None)
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["company", "checksum", "basis", "statement",
              "start_pdf_page", "end_pdf_page", "status"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose BS/PL/CF statement spans")
    parser.add_argument("--storage-dir", default="data/corpus")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--candidates", default="data/candidates.csv")
    parser.add_argument("--out", default="data/qa/statement_map_proposed.csv")
    parser.add_argument("--annual-only", action="store_true", default=True)
    args = parser.parse_args(argv)

    annual_companies: set[str] = set()
    with open(args.candidates, newline="", encoding="utf-8") as fh:
        for csv_row in csv.DictReader(fh):
            if (csv_row.get("document_type") or "").strip() == "annual":
                annual_companies.add((csv_row.get("company_id") or "").strip())

    ledger = Ledger(args.ledger)
    out_path = Path(args.out)
    existing = _read_existing(out_path)
    seen = {(row["checksum"], row["basis"], row["statement"]) for row in existing
            if row.get("checksum")}
    new_rows: list[dict] = []
    for ledger_row in sorted(ledger.rows.values(),
                              key=lambda r: (r.company_id, r.checksum)):
        if (ledger_row.doc_type.value != "annual"
                or ledger_row.company_id not in annual_companies):
            continue
        name = f"{ledger_row.checksum}.pdf"
        if name in {seen_key[0] for seen_key in seen}:
            continue
        pages = load_pages(Path(args.storage_dir) / "extract", ledger_row.checksum)
        if not pages:
            continue
        for span in propose_spans(ledger_row.company_id, name, pages):
            new_rows.append(span.as_row())

    rows = existing + new_rows if existing else new_rows
    _write(out_path, rows)
    print(f"wrote {len(new_rows)} new spans for {len(annual_companies)} annual companies "
          f"to {out_path} ({len(rows)} rows total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
