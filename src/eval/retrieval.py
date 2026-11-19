"""Retrieval baselines: recall@k with the gold-evidence hit definition.

A hit = a retrieved chunk whose (document_id, page) equals a gold evidence
cite. For rows citing multiple (doc, page) pairs, ANY cited pair in the
top-k counts as a hit (document-level hit); this is the unit reported.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict


def gold_cites(rows: list[dict]) -> list[set]:
    """Per row, the set of cited (doc_id, page) pairs."""
    out = []
    for r in rows:
        ids = r["doc_id"].split("|")
        page_s = r["page"].split("|")
        cites = set()
        try:
            ps = [int(p) for p in page_s]
        except ValueError:
            ps = []
        for did, p in zip(ids, ps):
            cites.add((did, p))
        out.append(cites)
    return out


def hit_at_k(retrieved: list[dict], cites: set, k: int) -> bool:
    """True if any of the first k retrieved chunks matches any cited (doc,page)."""
    for chunk in retrieved[:k]:
        meta = chunk.get("meta", {})
        if (meta.get("document_id"), meta.get("page")) in cites:
            return True
    return False


def recall_at_k(per_row_hits: list[bool], k: int) -> float:
    if not per_row_hits:
        return 0.0
    return sum(per_row_hits) / len(per_row_hits)


def evaluate(rows: list[dict], retrieved_by_row: list[list[dict]],
             ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    cites = gold_cites(rows)
    per_k = {k: [] for k in ks}
    per_cat: dict[str, dict[int, list[bool]]] = defaultdict(lambda: {k: [] for k in ks})
    for row, retrieved, cit in zip(rows, retrieved_by_row, cites):
        cat = row.get("category", "?")
        for k in ks:
            h = hit_at_k(retrieved, cit, k)
            per_k[k].append(h)
            per_cat[cat][k].append(h)
    out = {f"recall@{k}": recall_at_k(per_k[k], k) for k in ks}
    out["hits"] = {str(k): int(sum(per_k[k])) for k in ks}
    out["n"] = len(rows)
    out["by_category"] = {
        cat: {f"recall@{k}": recall_at_k(cat_hits[k], k) for k in ks}
        for cat, cat_hits in sorted(per_cat.items())
    }
    return out


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
