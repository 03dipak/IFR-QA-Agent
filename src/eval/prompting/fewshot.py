"""Few-shot examples for S1, sourced only from non-gold corpus pages."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import GOLD_PATH

PATH = Path("data/eval/prompts/fewshot_v1.jsonl")
GOLD = Path(GOLD_PATH)


def _gold_rows() -> list[dict]:
    rows = []
    if not GOLD.is_file():
        return rows
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _gold_cited_pages() -> set[tuple[str, int]]:
    cited: set[tuple[str, int]] = set()
    for row in _gold_rows():
        for ev in row.get("evidence") or []:
            did = ev.get("document_id")
            pg = ev.get("pdf_page")
            if isinstance(did, str) and isinstance(pg, int):
                cited.add((did, pg))
    return cited


def _gold_doc_ids() -> set[str]:
    ids = set()
    for row in _gold_rows():
        did = row.get("document_id")
        if isinstance(did, str) and did:
            ids.add(did)
        for ev in row.get("evidence") or []:
            ed = ev.get("document_id") if isinstance(ev, dict) else None
            if isinstance(ed, str) and ed:
                ids.add(ed)
    return ids


def _row_valid(row: dict) -> bool:
    did = row.get("context_document_id")
    pg = row.get("context_page")
    if not isinstance(did, str) or not did.strip():
        return False
    if not isinstance(pg, int):
        return False
    return isinstance(row.get("answer"), dict)


def load_few_shot(exclude_doc_ids: set[str] | None = None) -> list[dict]:
    if not PATH.is_file():
        raise FileNotFoundError(PATH)
    rows = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 3:
        raise ValueError(f"expected 3 few-shot examples, got {len(rows)}")
    cited = _gold_cited_pages()
    gold_docs = _gold_doc_ids()
    ex = set(exclude_doc_ids or [])
    out = []
    for row in rows:
        if not _row_valid(row):
            raise AssertionError("few-shot example has ambiguous/invalid document identity")
        did = row["context_document_id"]
        key = (did, row["context_page"])
        if key in cited:
            raise AssertionError(f"few-shot overlaps gold evidence: {key}")
        if did in gold_docs:
            raise AssertionError(f"few-shot uses gold document id: {did}")
        if did in ex:
            continue
        out.append(row)
    return out
