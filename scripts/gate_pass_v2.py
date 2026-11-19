"""Gate pass v2: run analyze_pdf on gate-unset and gate-exclude docs; write decisions file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.ingestion.gate_report import analyze_pdf

LEDGER = Path("data/corpus/ledger.jsonl")
RAW = Path("data/corpus/raw")
OUT = Path("data/corpus/gate_pass_v2.json")


def load_docs() -> dict[str, dict]:
    docs: dict[str, dict] = {}
    with LEDGER.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") == "document" and record["data"].get("checksum"):
                docs[record["data"]["checksum"]] = record["data"]
    return docs


def selected(docs: dict[str, dict]) -> list[dict]:
    out = [d for d in docs.values()
           if d.get("extraction_status") == "extracted"
           and d.get("gate_decision") in (None, "exclude")]
    return sorted(out, key=lambda d: (d.get("company_id") or "", d.get("checksum")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate pass over unset/exclude docs")
    parser.add_argument("--apply", action="store_true", help="write gate_pass_v2.json")
    parser.add_argument("--limit", type=int, default=None, help="max docs this run")
    args = parser.parse_args()

    done: dict[str, dict] = {}
    if OUT.exists():
        for row in json.loads(OUT.read_text(encoding="utf-8")):
            done[row["checksum"]] = row
        print(f"resuming: {len(done)} already decided")

    docs = load_docs()
    todo = [d for d in selected(docs) if d["checksum"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    decisions = list(done.values())
    counts: Counter = Counter((d["prior_gate_decision"], d["decision"])
                              for d in decisions)
    for index, doc in enumerate(todo, 1):
        checksum = doc["checksum"]
        prior = doc.get("gate_decision")
        record = analyze_pdf(RAW / f"{checksum}.pdf", doc.get("doc_type"), None)
        decision = record["decision"]
        counts[(prior, decision)] += 1
        decisions.append({
            "checksum": checksum, "company_id": doc.get("company_id"),
            "doc_type": doc.get("doc_type"), "fiscal_year": doc.get("fiscal_year"),
            "prior_gate_decision": prior, "decision": decision,
            "fired_rules": record.get("fired_rules", []),
            "pages": record.get("pages"), "anomalies": record.get("anomalies"),
        })
        print(f"[{index}/{len(todo)}] {doc.get('company_id')} {doc.get('doc_type')} "
              f"{doc.get('fiscal_year')} {checksum[:12]} "
              f"prior={prior} -> {decision} rules={record.get('fired_rules', [])}",
              flush=True)
        if args.apply and index % 5 == 0:
            OUT.write_text(json.dumps(decisions, indent=1), encoding="utf-8")
    print("summary (prior -> new):", dict(counts))
    print("include count:", sum(1 for d in decisions if d["decision"] == "keep"))
    if args.apply:
        OUT.write_text(json.dumps(decisions, indent=1), encoding="utf-8")
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
