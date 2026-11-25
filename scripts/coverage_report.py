from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent.graph import AgentGraph
from src.agent.llm import StubLLM

INDEX = Path("results/full_index_v1/page_hash384/chunks.jsonl")
DB = Path("results/agent_state.sqlite")
COMPANIES = ["adani_green", "tcs"]
THREADS = ["t1d", "t2d", "t3d"]


def _periods(meta: dict, text: str) -> set[str]:
    found: set[str] = set()
    for key in ("period", "fiscal_year", "year", "fy", "report_period", "date"):
        value = meta.get(key)
        if value not in (None, "", [], {}):
            found.add(f"{key}={value}")
    hay = " ".join([
        str(meta.get("document_id", "")),
        str(meta.get("source", "")),
        str(meta.get("source_path", "")),
        str(meta.get("filename", "")),
        text[:2000],
    ])
    for pat in (
        r"FY\s*20\d{2}(?:-\d{2,4})?",
        r"20\d{2}-\d{2,4}",
        r"20\d{2}\s*-\s*20\d{2}",
    ):
        for m in re.finditer(pat, hay, flags=re.I):
            found.add(m.group(0).replace("  ", " ").strip())
    return found


def report_coverage() -> None:
    stats = {
        c: {"chunks": 0, "sources": set(), "periods": set(), "sample": []}
        for c in COMPANIES
    }
    with INDEX.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = chunk.get("meta") or {}
            company = meta.get("company_id")
            if company not in stats:
                continue
            s = stats[company]
            s["chunks"] += 1
            source = (
                meta.get("source") or meta.get("source_path")
                or meta.get("filename") or meta.get("document_id") or "unknown"
            )
            s["sources"].add(str(source))
            s["periods"].update(_periods(meta, str(chunk.get("text", ""))))
            if len(s["sample"]) < 3:
                s["sample"].append({
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": meta.get("document_id"),
                    "period": meta.get("period"),
                    "statement_kind": meta.get("statement_kind"),
                    "reporting_basis": meta.get("reporting_basis"),
                })
    for company in COMPANIES:
        s = stats[company]
        periods = sorted(s["periods"])
        print(f"== {company} ==")
        print(f"chunks={s['chunks']}")
        print(f"distinct_sources={len(s['sources'])}")
        for src in sorted(s["sources"])[:20]:
            print(f"  source: {src}")
        if len(s["sources"]) > 20:
            print(f"  ... {len(s['sources']) - 20} more")
        if periods:
            print("periods/years:")
            for p in periods[:60]:
                print(f"  - {p}")
            if len(periods) > 60:
                print(f"  ... {len(periods) - 60} more")
        else:
            print("periods/years: none found in meta or filename/text patterns")
        print("sample_meta:")
        print(json.dumps(s["sample"], indent=2, ensure_ascii=False))


def report_filters() -> None:
    targets = ["deepak_nitrite", "srf", "aurobindo_pharma", "dr_reddys"]
    company_ids: set[str] = set()
    statement_kinds: Counter[str] = Counter()
    company_stats = {
        c: {
            "chunks": 0,
            "periods": Counter(),
            "balance_sheet_fy2025": 0,
            "period_empty": 0,
        }
        for c in targets
    }
    with INDEX.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = chunk.get("meta") or {}
            cid = meta.get("company_id")
            if isinstance(cid, str) and cid:
                company_ids.add(cid)
            sk = meta.get("statement_kind")
            statement_kinds[str(sk) if sk not in (None, "") else ""] += 1
            if cid not in company_stats:
                continue
            s = company_stats[cid]
            s["chunks"] += 1
            period = str(meta.get("period") or "")
            s["periods"][period] += 1
            if not period:
                s["period_empty"] += 1
            if str(sk) == "balance_sheet" and period.upper().startswith("FY2025"):
                s["balance_sheet_fy2025"] += 1

    print("# all distinct company_ids")
    for cid in sorted(company_ids):
        print(cid)
    print(f"total_company_ids={len(company_ids)}")

    print("\n# target companies")
    for cid in targets:
        s = company_stats[cid]
        print(f"== {cid} ==")
        print(f"chunks={s['chunks']}")
        print("distinct_periods:")
        for period, count in s["periods"].most_common():
            label = period if period else "<EMPTY>"
            print(f"  {label}: {count}")
        print(f"balance_sheet_and_period_startswith_FY2025={s['balance_sheet_fy2025']}")
        print(f"period_empty={s['period_empty']}")

    print("\n# statement_kind counts")
    for kind, count in statement_kinds.most_common():
        label = kind if kind else "<EMPTY>"
        print(f"{label}: {count}")


def _figure_strings(text: str) -> list[str]:
    figs = []
    for m in re.finditer(r"(?<![A-Za-z_])[-+]?\d[\d,]*(?:\.\d+)?%?(?![A-Za-z_])", text):
        raw = m.group(0)
        normalized = raw.replace(",", "")
        if len(normalized.lstrip("-+%.")) < 2:
            continue
        figs.append(raw)
    return list(dict.fromkeys(figs))


def _retrieved_text(values: dict) -> str:
    parts = []
    for entry in values.get("tool_results") or []:
        result = entry.get("result")
        if isinstance(result, list):
            for hit in result:
                if isinstance(hit, dict):
                    parts.append(json.dumps(hit, ensure_ascii=False, default=str))
        elif isinstance(result, dict):
            parts.append(json.dumps(result, ensure_ascii=False, default=str))
    return "\n".join(parts)


def report_runs() -> None:
    con = sqlite3.connect(DB, check_same_thread=False)
    graph = AgentGraph(llm=StubLLM([]), checkpointer=SqliteSaver(con))
    for thread in THREADS:
        state = graph.graph.get_state({"configurable": {"thread_id": thread}}).values
        answer = str(state.get("draft_answer") or "")
        lower = answer.lower()
        insufficient = any(
            s in lower
            for s in [
                "insufficient", "not enough", "cannot find", "no evidence",
                "not available", "unable to", "no tool results",
            ]
        )
        retrieved = _retrieved_text(state)
        figures = _figure_strings(answer)
        print(f"== {thread} ==")
        print(f"query={state.get('query')}")
        print(f"stop_reason={state.get('stop_reason')}")
        print(f"insufficient_evidence_stated={insufficient}")
        print(f"answer={answer[:1200]}")
        if not figures:
            print("figures: none")
        else:
            for fig in figures:
                variants = {fig, fig.replace(",", ""), fig.replace(",", " ")}
                found = any(v in retrieved for v in variants if v)
                print(f"figure={fig!r} in_retrieved_chunks={found}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filters", action="store_true")
    args = parser.parse_args()
    if args.filters:
        report_filters()
        return
    print("# coverage")
    report_coverage()
    print("\n# runs")
    report_runs()


if __name__ == "__main__":
    main()
