"""Spot-check corrected quarterly docs against their printed pages.

Prints company, period, frozen variant label, the nature-of-report marker with
its page, and the printed revenue figure with its page.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

CORRECTIONS = Path("data/corpus/ledger_corrections_v3.jsonl")
FREEZE_V2 = Path("data/corpus/freeze_v2.json")
PAGES = Path("data/corpus/extract/page_extracts")

MARKER = re.compile(r"nature of report\s*(?:standalone or consolidated)?\s*"
                    r"(Standalone|Consolidated)", re.I)
REVENUE = re.compile(r"revenue from operations\s*[:\s]*([\d,.]+)", re.I)
REQUIRED = ("bhel", "upl", "pidilite")


def page_texts(doc_id: str) -> list[tuple[int, str]]:
    path = PAGES / f"{doc_id}.pages.jsonl"
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out.append((rec["pdf_page_index"], " ".join((rec.get("page_text") or "").split())))
    return out


def pick(overrides: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for company in REQUIRED:
        hit = next((o for o in overrides if o["company_id"] == company), None)
        if hit:
            chosen.append(hit)
    pool = [o for o in overrides if o not in chosen]
    chosen += random.Random(42).sample(pool, 10 - len(chosen))
    return chosen


def main() -> int:
    overrides = [json.loads(line) for line in CORRECTIONS.read_text(encoding="utf-8").splitlines()
                 if line.strip() and json.loads(line).get("kind") == "variant_override"]
    freeze = json.loads(FREEZE_V2.read_text(encoding="utf-8"))
    labels = {d["document_id"]: d.get("variant") for d in freeze["documents"]}
    for o in pick(overrides):
        doc_id = o["document_id"]
        marker_page = marker = revenue = rev_page = None
        for page, text in page_texts(doc_id):
            m = MARKER.search(text)
            if m and marker is None:
                marker_page, marker = page, m.group(1)
            r = REVENUE.search(text)
            if r and revenue is None:
                rev_page, revenue = page, r.group(1)
        print(f"{o['company_id']:<14} {o['fiscal_year']:<8} {doc_id[:12]} "
              f"label={labels.get(doc_id):<13} printed={marker} (p{marker_page}) "
              f"revenue={revenue} (p{rev_page})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
