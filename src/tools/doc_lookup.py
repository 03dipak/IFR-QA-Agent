"""Document lookup tool: filter the frozen 37-doc dev slice by company and year."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.resolver.query_resolver import ResolutionResult, resolve_query

FREEZE_PATH = Path("data/corpus/freeze_v1.json")
CHUNK_IDS = Path("results/week2_retrieval_cache/C2/chunk_ids.json")


def _slice_doc_ids() -> set[str]:
    ids = json.loads(CHUNK_IDS.read_text(encoding="utf-8"))
    return {item.split("::", 1)[0] for item in ids}


def load_slice() -> list[dict[str, Any]]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    slice_ids = _slice_doc_ids()
    return [doc for doc in freeze.get("documents", [])
            if doc.get("document_id") in slice_ids]


def lookup(
    question: str = "",
    *,
    query: str | None = None,
    company_ids: list[str] | None = None,
    years: list[int] | None = None,
    resolution: ResolutionResult | None = None,
) -> list[dict[str, Any]]:
    """Return freeze rows matching resolved company ids and fiscal years."""
    question = question or query or ""
    resolution = resolution or resolve_query(question)
    companies = company_ids if company_ids is not None else list(resolution.company_ids)
    wanted = set(companies)
    fiscal_years = years
    if fiscal_years is None:
        fiscal_years = [slot.fiscal_year for slot in resolution.periods]
    found = []
    for doc in load_slice():
        if wanted and doc.get("company_id") not in wanted:
            continue
        if fiscal_years:
            raw = str(doc.get("fiscal_year") or "")
            digits = "".join(c for c in raw if c.isdigit())
            if not any(str(year)[-2:] in digits or str(year) in raw for year in fiscal_years):
                continue
        found.append({
            "doc_id": doc.get("document_id"),
            "company_id": doc.get("company_id"),
            "fiscal_year": doc.get("fiscal_year"),
            "variant": doc.get("variant"),
            "doc_type": doc.get("doc_type"),
        })
    return found


TOOLS = {"doc_lookup": lookup}
