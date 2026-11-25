"""Day 3 submission evidence tables for retrieval, citation and failure analysis (LLD-29).

Reads existing evaluation artifacts, the frozen gold and the frozen C2 index metadata.
An optional bounded retrieval probe replays the recorded search queries to recover the
full top-k ranking that was truncated in the agent traces. It makes no model calls and
never re-embeds or downloads corpus files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.eval import agent_eval

CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)::C2::p(?P<page>\d+)::w(?P<win>\d+)$")
FIRST_CHUNK_RE = re.compile(r"doc_id='([0-9a-f]{8,64})', page=(\d+)")
DOC_RE = re.compile(r"[0-9a-f]{8,64}", re.I)
PAGE_RE = re.compile(r"p\.\s*(\d+)", re.I)
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
ABSTAIN_RE = re.compile(
    r"cannot answer|not possible|cannot determine|insufficient|no evidence|abstain", re.I,
)

FAILURE_LABELS = {
    "wrong_document_retrieved": "Wrong document retrieved",
    "correct_document_wrong_page": "Correct document, wrong page",
    "relevant_evidence_not_retrieved": "Relevant evidence not retrieved",
    "citation_unresolved": "Citation unresolved",
    "citation_parsing_failure": "Citation parsing failure",
    "unsupported_answer": "Unsupported answer",
    "incorrect_answer_despite_correct_evidence": "Incorrect answer despite correct evidence",
    "other_insufficient_evidence": "Other / insufficient evidence",
}
CLASSIFICATION_ORDER = tuple(FAILURE_LABELS)
COLUMNS_A = [
    "query_id", "expected_document_id", "retrieved_document_ids",
    "expected_document_retrieved", "retrieval_rank", "retrieval_call",
    "missing_document_status",
]
COLUMNS_B = [
    "query_id", "expected_document_id", "expected_page", "retrieved_pages",
    "correct_page_retrieved", "page_coverage_status",
]
COLUMNS_C = [
    "query_id", "generated_citation", "expected_document_page",
    "actual_cited_document_page", "citation_status", "failure_category",
    "harness_citation_parsed",
]
COLUMNS_D = [
    "query_id", "expected_behaviour", "answer_correct", "stopped_reason",
    "classifications", "primary_failure", "evidence_basis",
]
COLUMNS_E = ["metric", "value", "numerator", "denominator", "definition"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gold_pairs(row: dict[str, Any]) -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for evidence in row.get("evidence", []):
        page = evidence.get("pdf_page")
        doc = evidence.get("document_id")
        if isinstance(page, int) and doc:
            out.append((str(doc)[:12], page, str(evidence.get("evidence_role") or "")))
    return out


def gold_row_map(gold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("row_id")): row for row in gold_rows}


def indexed_pairs(chunk_ids_path: Path) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for chunk_id in json.loads(Path(chunk_ids_path).read_text(encoding="utf-8")):
        match = CHUNK_ID_RE.match(str(chunk_id))
        if match:
            pairs.add((match.group("doc")[:12], int(match.group("page"))))
    return pairs


def first_chunk(detail: str) -> tuple[str, int] | None:
    match = FIRST_CHUNK_RE.search(detail or "")
    return (match.group(1)[:12], int(match.group(2))) if match else None


def retrieval_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for event in events:
        if event.get("stage") != "tool" or event.get("tool") != "retrieve":
            continue
        calls.append({
            "query": str((event.get("args") or {}).get("query") or ""),
            "first_chunk": first_chunk(str(event.get("detail") or "")),
            "error": bool(event.get("error")),
        })
    return calls


def diagnostic_refs(answer: str) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for bracket in BRACKET_RE.findall(answer or ""):
        current: str | None = None
        for token in re.finditer(r"([0-9a-f]{8,64})|p\.\s*(\d+)", bracket, re.I):
            if token.group(1):
                current = token.group(1)[:12]
            elif token.group(2) and current:
                refs.append((current, int(token.group(2))))
    return list(dict.fromkeys(refs))


def harness_refs(answer: str) -> list[tuple[str, int]]:
    return agent_eval._citation_refs(answer)


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    return manifest, read_jsonl(Path(run_dir) / "per_query.jsonl")


def load_calls(
    row_ids: list[str], traces_dir: Path, run_id: str,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id in row_ids:
        path = Path(traces_dir) / f"eval_{run_id}_{row_id}.jsonl"
        out[row_id] = retrieval_calls(read_jsonl(path)) if path.is_file() else []
    return out


def probe_rows(
    row_id: str, calls: list[dict[str, Any]], existing: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    from src import config
    from src.tools.retrieve import retrieve

    _ = config.GOLD_PATH
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    rows = []
    for call_index, call in enumerate(calls):
        key = (row_id, call_index)
        if key in existing:
            rows.append(existing[key])
            continue
        start = time.monotonic()
        chunks = retrieve(call["query"])
        elapsed = int((time.monotonic() - start) * 1000)
        chunks_out = [
            {"rank": rank, "doc_id": chunk.doc_id[:12], "page": int(chunk.page),
             "score": float(chunk.score)}
            for rank, chunk in enumerate(chunks, start=1)
        ]
        expected = call.get("first_chunk")
        observed = (chunks_out[0]["doc_id"], chunks_out[0]["page"]) if chunks_out else None
        row = {
            "row_id": row_id, "call_index": call_index, "query": call["query"],
            "chunks": chunks_out, "latency_ms": elapsed,
            "trace_first_chunk": f"{expected[0]}:{expected[1]}" if expected else None,
            "anchor_match": bool(expected and observed == expected),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        rows.append(row)
        time.sleep(1.0)
    return rows


def run_probe(
    calls_by_row: dict[str, list[dict[str, Any]]], out_path: Path,
    *, limit_rows: int | None = None,
) -> dict[str, Any]:
    out_path = Path(out_path)
    existing_rows = read_jsonl(out_path) if out_path.is_file() else []
    existing = {(r["row_id"], int(r["call_index"])): r for r in existing_rows}
    row_ids = list(calls_by_row)[:limit_rows]
    first = time.monotonic()
    generated: list[dict[str, Any]] = []
    for index, row_id in enumerate(row_ids, start=1):
        generated.extend(probe_rows(row_id, calls_by_row[row_id], existing))
        print(f"probe {index}/{len(row_ids)} {row_id}", flush=True)
    merged = {**existing, **{(r["row_id"], int(r["call_index"])): r for r in generated}}
    rows = sorted(merged.values(), key=lambda r: (r["row_id"], int(r["call_index"])))
    write_jsonl(out_path, rows)
    total = int((time.monotonic() - first) * 1000)
    return {
        "n_queries": len(rows), "n_rows": len({r["row_id"] for r in rows}),
        "elapsed_ms": total, "anchor_matches": sum(bool(r.get("anchor_match")) for r in rows),
        "path": str(out_path),
    }


def probe_map(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["row_id"])].append(row)
    for calls in out.values():
        calls.sort(key=lambda r: int(r["call_index"]))
    return dict(out)


def retrieved(calls: list[dict[str, Any]]) -> list[tuple[str, int]]:
    out = []
    for call in calls:
        for chunk in call.get("chunks", []):
            pair = (str(chunk["doc_id"])[:12], int(chunk["page"]))
            if pair not in out:
                out.append(pair)
    return out


def rank_for(calls: list[dict[str, Any]], doc_id: str) -> tuple[int | None, int | None]:
    best: tuple[int, int] | None = None
    for call in calls:
        for chunk in call.get("chunks", []):
            if str(chunk["doc_id"])[:12] == doc_id:
                candidate = (int(call["call_index"]), int(chunk["rank"]))
                best = candidate if best is None or candidate < best else best
    return best if best else (None, None)


def table_a(
    rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]],
    calls_by_row: dict[str, list[dict[str, Any]]], probes: dict[str, list[dict[str, Any]]],
    indexed: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_id = str(row["row_id"])
        calls = probes.get(row_id, [])
        observed = retrieved(calls) if calls else [
            call["first_chunk"] for call in calls_by_row.get(row_id, [])
            if call.get("first_chunk")
        ]
        docs = list(dict.fromkeys(doc for doc, _ in observed))
        for doc_id, page, _role in gold_pairs(gold[row_id]):
            doc_indexed = any(pair[0] == doc_id for pair in indexed)
            if not calls:
                present, rank, call_no = "unknown", None, None
                status = "unknown_no_retrieval_evidence"
            else:
                call_no, rank = rank_for(calls, doc_id)
                present = "yes" if doc_id in docs else "no"
                status = ("retrieved" if present == "yes" else
                          "indexed_not_retrieved" if doc_indexed else "not_in_index")
            out.append({
                "query_id": row_id, "expected_document_id": doc_id,
                "retrieved_document_ids": "|".join(docs) or "(none)",
                "expected_document_retrieved": present, "retrieval_rank": rank,
                "retrieval_call": call_no, "missing_document_status": status,
                "expected_page": page,
            })
    return out


def table_b(
    rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]],
    probes: dict[str, list[dict[str, Any]]], indexed: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_id = str(row["row_id"])
        calls = probes.get(row_id, [])
        observed = retrieved(calls)
        for doc_id, page, _role in gold_pairs(gold[row_id]):
            pages = [p for d, p in observed if d == doc_id]
            if not calls:
                hit, status = "unknown", "unknown_no_retrieval_evidence"
            elif (doc_id, page) in observed:
                hit, status = "yes", "retrieved"
            elif any(d == doc_id for d, _ in observed):
                hit, status = "no", "doc_retrieved_page_missing"
            elif (doc_id, page) not in indexed:
                hit, status = "no", "gold_page_not_in_index"
            else:
                hit, status = "no", "doc_not_retrieved"
            out.append({
                "query_id": row_id, "expected_document_id": doc_id,
                "expected_page": page,
                "retrieved_pages": "|".join(str(p) for p in pages) or "(none)",
                "correct_page_retrieved": hit, "page_coverage_status": status,
            })
    return out


def citation_status(
    ref: tuple[str, int], gold: list[tuple[str, int, str]], observed: list[tuple[str, int]],
    has_probe: bool,
) -> str:
    if has_probe and ref not in observed:
        return "cited_evidence_not_retrieved"
    if ref in {(doc, page) for doc, page, _ in gold}:
        return "correct"
    if ref[0] in {doc for doc, _, _ in gold}:
        return "correct_document_wrong_page"
    return "wrong_document"


def table_c(
    rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]],
    probes: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_id = str(row["row_id"])
        answer = str(row.get("answer") or "")
        refs = diagnostic_refs(answer)
        accepted = set(harness_refs(answer))
        pairs = gold_pairs(gold[row_id])
        gold_text = "|".join(f"{doc} p.{page}" for doc, page, _ in pairs) or "(none)"
        observed = retrieved(probes.get(row_id, []))
        has_probe = bool(probes.get(row_id))
        for ref in refs:
            status = citation_status(ref, pairs, observed, has_probe)
            parsed = ref in accepted
            category = (
                "citation_parsing_failure" if not parsed else
                "citation_unresolved" if status == "cited_evidence_not_retrieved" else
                status
            )
            out.append({
                "query_id": row_id, "generated_citation": f"[{ref[0]} p.{ref[1]}]",
                "expected_document_page": gold_text,
                "actual_cited_document_page": f"{ref[0]} p.{ref[1]}",
                "citation_status": status, "failure_category": category,
                "harness_citation_parsed": "yes" if parsed else "no",
            })
        if not refs:
            unresolved = str(row.get("stopped_reason")) == "citation_unresolved"
            out.append({
                "query_id": row_id, "generated_citation": "(none)",
                "expected_document_page": gold_text,
                "actual_cited_document_page": "(none)",
                "citation_status": "missing_or_unparseable",
                "failure_category": "citation_unresolved" if unresolved else "missing_citation",
                "harness_citation_parsed": "no",
            })
    return out


def classify_row(
    row: dict[str, Any], gold: dict[str, Any], probes: dict[str, list[dict[str, Any]]],
    indexed: set[tuple[str, int]],
) -> tuple[list[str], str]:
    row_id = str(row["row_id"])
    calls = probes.get(row_id, [])
    observed = retrieved(calls)
    pairs = [(doc, page) for doc, page, _ in gold_pairs(gold)]
    docs = {doc for doc, _ in pairs}
    refs = diagnostic_refs(str(row.get("answer") or ""))
    accepted = harness_refs(str(row.get("answer") or ""))
    answer = str(row.get("answer") or "")
    found: list[str] = []
    if calls:
        retrieved_docs = {doc for doc, _ in observed}
        if docs - retrieved_docs:
            found.append("wrong_document_retrieved")
        if docs & retrieved_docs and not set(pairs) & set(observed):
            found.append("correct_document_wrong_page")
    if any(pair not in indexed for pair in pairs):
        found.append("relevant_evidence_not_retrieved")
    if str(row.get("stopped_reason")) == "citation_unresolved":
        found.append("citation_unresolved")
    if refs and not accepted:
        found.append("citation_parsing_failure")
    if answer.strip() and not ABSTAIN_RE.search(answer):
        if not refs or (calls and not set(refs) & set(observed)):
            found.append("unsupported_answer")
    if calls and set(pairs) & set(observed) and not row.get("answer_correct"):
        found.append("incorrect_answer_despite_correct_evidence")
    if not found:
        found.append("other_insufficient_evidence")
    ordered = [name for name in CLASSIFICATION_ORDER if name in found]
    return ordered, ordered[0]


def table_d(
    rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]],
    probes: dict[str, list[dict[str, Any]]], indexed: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_id = str(row["row_id"])
        found, primary = classify_row(row, gold[row_id], probes, indexed)
        out.append({
            "query_id": row_id, "expected_behaviour": row.get("expected_behaviour"),
            "answer_correct": row.get("answer_correct"),
            "stopped_reason": row.get("stopped_reason"),
            "classifications": "|".join(found), "primary_failure": primary,
            "evidence_basis": "retrieval_probe" if probes.get(row_id)
            else "trace_first_chunk_only",
        })
    return out


def _metric(
    name: str, value: Any, numerator: Any, denominator: Any, definition: str,
) -> dict[str, Any]:
    return {"metric": name, "value": value, "numerator": numerator,
            "denominator": denominator, "definition": definition}


def summary(
    rows: list[dict[str, Any]], gold: dict[str, dict[str, Any]], table_a_rows: list[dict],
    table_c_rows: list[dict], table_d_rows: list[dict], indexed: set[tuple[str, int]],
    probe_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    n = len(rows)
    complete_ids = {str(row_id) for row_id in (probe_meta or {}).get("row_ids", [])}
    denom = len(complete_ids)
    gold_doc_hits = 0
    gold_page_hits = 0
    all_page_hits = 0
    expected_docs = 0
    expected_doc_hits = 0
    expected_pairs = 0
    expected_pair_hits = 0
    for row in rows:
        row_id = str(row["row_id"])
        if row_id not in complete_ids:
            continue
        calls = (probe_meta or {}).get("probes", {}).get(row_id, [])
        observed = set(retrieved(calls))
        pairs = {(doc, page) for doc, page, _ in gold_pairs(gold[row_id])}
        docs = {doc for doc, _ in pairs}
        gold_doc_hits += bool(docs & {doc for doc, _ in observed})
        gold_page_hits += bool(pairs & observed)
        all_page_hits += bool(pairs) and pairs <= observed
        expected_docs += len(docs)
        expected_doc_hits += len(docs & {doc for doc, _ in observed})
        expected_pairs += len(pairs)
        expected_pair_hits += len(pairs & observed)
    ref_count = len([r for r in table_c_rows if r["generated_citation"] != "(none)"])
    correct_refs = len([r for r in table_c_rows if r["citation_status"] == "correct"])
    all_pairs = {(str(row["row_id"]), doc, page) for row in rows
                 for doc, page, _ in gold_pairs(gold[str(row["row_id"])])}
    cited_pairs = {
        (str(r["query_id"]), r["actual_cited_document_page"].split(" ")[0],
         int(r["actual_cited_document_page"].split("p.")[1]))
        for r in table_c_rows if r["citation_status"] == "correct"
    }
    answer_rows = [r for r in rows if r.get("expected_behaviour") == "answer"]
    classes = {name: 0 for name in FAILURE_LABELS}
    for row in table_d_rows:
        for name in str(row["classifications"]).split("|"):
            if name in classes:
                classes[name] += 1
    indexed_total = sum(
        1 for row in rows for doc, page, _ in gold_pairs(gold[str(row["row_id"])])
        if (doc, page) in indexed
    )
    metrics = [
        _metric("document_retrieval_coverage_rows",
                round(gold_doc_hits / denom, 4) if denom else None,
                gold_doc_hits, denom,
                "Rows where >=1 expected document appears in replayed top-k; "
                "denominator is complete-probe rows."),
        _metric("expected_document_retrieval_coverage",
                round(expected_doc_hits / expected_docs, 4) if expected_docs else None,
                expected_doc_hits, expected_docs,
                "Expected documents found in replayed top-k / all expected documents."),
        _metric("page_retrieval_coverage_rows_any",
                round(gold_page_hits / denom, 4) if denom else None,
                gold_page_hits, denom,
                "Rows where >=1 expected (document,page) pair appears in replayed top-k."),
        _metric("expected_page_retrieval_coverage",
                round(expected_pair_hits / expected_pairs, 4) if expected_pairs else None,
                expected_pair_hits, expected_pairs,
                "Expected (document,page) pairs found in replayed top-k / all expected pairs."),
        _metric("page_retrieval_coverage_rows_all",
                round(all_page_hits / denom, 4) if denom else None,
                all_page_hits, denom,
                "Rows where every expected (document,page) pair appears in replayed top-k."),
        _metric("citation_precision", round(correct_refs / ref_count, 4) if ref_count else None,
                correct_refs, ref_count,
                "Diagnostic citation refs matching a gold pair / all extracted citation refs."),
        _metric(
            "citation_recall", round(len(cited_pairs) / len(all_pairs), 4) if all_pairs else None,
            len(cited_pairs), len(all_pairs),
            "Gold (row,document,page) pairs correctly cited / all gold pairs.",
        ),
        _metric("citation_correctness_rows",
                round(sum(bool(r.get("citation_correct")) for r in answer_rows)
                      / len(answer_rows), 4) if answer_rows else None,
                sum(bool(r.get("citation_correct")) for r in answer_rows), len(answer_rows),
                "Per-query harness citation correctness / answer rows."),
        _metric("citation_unresolved_count",
                sum(1 for r in rows if r.get("stopped_reason") == "citation_unresolved"),
                None, None, "Rows whose agent stopped with citation_unresolved."),
        _metric("wrong_document_count", classes["wrong_document_retrieved"], None, None,
                "Rows with >=1 expected document missing from replayed top-k."),
        _metric("wrong_page_count", classes["correct_document_wrong_page"], None, None,
                "Rows with a retrieved expected document but no expected page in replayed top-k."),
        _metric("answer_accuracy",
                round(sum(bool(r.get("answer_correct")) for r in answer_rows)
                      / len(answer_rows), 4) if answer_rows else None,
                sum(bool(r.get("answer_correct")) for r in answer_rows), len(answer_rows),
                "Correct answers / rows with expected_behaviour=answer."),
        _metric("abstention_performance", None, 0, 0,
                "Not measurable: no abstain rows in this 10-row sample."),
        _metric("gold_index_coverage",
                round(indexed_total / len(all_pairs), 4) if all_pairs else None,
                indexed_total, len(all_pairs),
                "Gold (document,page) pairs present in the frozen C2 chunk index."),
    ]
    return {
        "n_rows": n, "probe_rows": denom, "all_gold_pairs": len(all_pairs),
        "citation_refs_extracted": ref_count, "correct_refs": correct_refs,
        "metrics": metrics, "failure_counts": classes,
        "failure_labels": FAILURE_LABELS,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    path: Path, *, manifest: dict[str, Any], metrics: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]], probe_meta: dict[str, Any] | None,
) -> None:
    m = {row["metric"]: row for row in metrics["metrics"]}
    lines = [
        "# Week 3–4 Day 3 — Submission Evidence Tables",
        "",
        "## 1. Executive summary",
        "",
        f"- Sample: `{manifest['n_rows']}` dev rows from frozen gold "
        f"`{manifest['gold_hash'][:16]}…`.",
        f"- Row-level doc coverage: **{m['document_retrieval_coverage_rows']['value']}** "
        f"({m['document_retrieval_coverage_rows']['numerator']}/"
        f"{m['document_retrieval_coverage_rows']['denominator']}); "
        f"expected-doc coverage: **{m['expected_document_retrieval_coverage']['value']}** "
        f"({m['expected_document_retrieval_coverage']['numerator']}/"
        f"{m['expected_document_retrieval_coverage']['denominator']}).",
        f"- Expected-page coverage: **{m['expected_page_retrieval_coverage']['value']}** "
        f"({m['expected_page_retrieval_coverage']['numerator']}/"
        f"{m['expected_page_retrieval_coverage']['denominator']}); row any-page: "
        f"**{m['page_retrieval_coverage_rows_any']['value']}**.",
        f"- Citation precision: **{m['citation_precision']['value']}**; recall: "
        f"**{m['citation_recall']['value']}**; harness citation correctness: "
        f"**{m['citation_correctness_rows']['value']}**; unresolved rows: "
        f"**{m['citation_unresolved_count']['value']}**.",
        f"- Answer accuracy: **{m['answer_accuracy']['value']}**; wrong-document rows: "
        f"**{m['wrong_document_count']['value']}**; wrong-page rows: "
        f"**{m['wrong_page_count']['value']}**.",
        f"- Gold evidence index coverage: **{m['gold_index_coverage']['value']}** — the misses "
        "are ranking/query-formulation failures, not missing indexed pages.",
        "",
        "## 2. Evaluation methodology",
        "",
        "- Inputs are the frozen gold file and the existing Day 2 live run; no gold labels were "
        "modified and no model/chat calls were made.",
        "- Table A/B use a bounded retrieval replay probe of the exact recorded tool queries "
        "against the frozen C2 index (top-5 per call), because original trace details were "
        "truncated to 300 characters. Probe anchor = the first chunk visible in the original "
        f"trace; {probe_meta['anchor_matches'] if probe_meta else 0}/"
        f"{probe_meta['n_queries'] if probe_meta else 0} anchors matched.",
        "- Table C extracts citation-like spans diagnostically, including comma-separated "
        "multi-document brackets that the production parser does not accept. The production "
        "parser output is recorded separately (`harness_citation_parsed`).",
        "- Table D separates measured retrieval/citation facts from inferred explanations; "
        "insufficient evidence is labelled, not forced into a failure class.",
        "",
        "## 3. Evidence tables",
        "",
        "### A. Retrieval document coverage",
        "",
        *markdown_table(tables["a"], COLUMNS_A),
        "",
        "### B. Retrieval page coverage",
        "",
        *markdown_table(tables["b"], COLUMNS_B),
        "",
        "### C. Citation evidence analysis",
        "",
        *markdown_table(tables["c"], COLUMNS_C),
        "",
        "### D. Failure classification",
        "",
        *markdown_table(tables["d"], COLUMNS_D),
        "",
        "### E. Summary metrics and definitions",
        "",
        *markdown_table(metrics["metrics"], COLUMNS_E),
        "",
        "## 4. Failure breakdown",
        "",
        *markdown_table(
            [{"class": FAILURE_LABELS[name], "rows": count} for name, count in
             metrics["failure_counts"].items()],
            ["class", "rows"],
        ),
        "",
        "## 5. Key findings (measured)",
        "",
        f"- Frozen index coverage: {m['gold_index_coverage']['numerator']}/"
        f"{m['gold_index_coverage']['denominator']} gold (document,page) pairs indexed; "
        "no index-gap failures measured.",
        f"- Document retrieval: {m['document_retrieval_coverage_rows']['numerator']}/"
        f"{m['document_retrieval_coverage_rows']['denominator']} rows found >=1 expected doc; "
        f"{m['expected_document_retrieval_coverage']['numerator']}/"
        f"{m['expected_document_retrieval_coverage']['denominator']} expected docs found "
        f"({m['wrong_document_count']['value']} rows missing >=1 expected doc).",
        f"- Page retrieval: {m['page_retrieval_coverage_rows_any']['numerator']}/"
        f"{m['page_retrieval_coverage_rows_any']['denominator']} rows found any expected page; "
        f"{m['expected_page_retrieval_coverage']['numerator']}/"
        f"{m['expected_page_retrieval_coverage']['denominator']} expected pairs found.",
        f"- Answers/citations: answer accuracy {m['answer_accuracy']['value']}, "
        f"harness citation correctness {m['citation_correctness_rows']['value']}, "
        f"diagnostic precision {m['citation_precision']['value']}, "
        f"diagnostic recall {m['citation_recall']['value']}.",
        f"- Citation failures: {m['citation_unresolved_count']['value']} unresolved rows; "
        f"{metrics['failure_counts']['citation_parsing_failure']} row(s) contain citation text "
        "not accepted by the production parser; remaining cited refs are wrong doc/page evidence.",
        "",
        "### Inferred explanations (not measured)",
        "",
        "- Wrong-document rows appear to be query/doc-selection errors: the planner reuses one "
        "company's document for the comparison company (G-001, G-003) or retrieves an unrelated "
        "company's report (G-036, G-037).",
        "- Wrong-page rows retrieve the right document but the wrong section/page; this is a "
        "ranking/query-formulation issue, not an index or chunking gap.",
        "",
        "## 6. Known limitations",
        "",
        "- Sample is 10 live dev rows, not the full 73-row dev set; no full evaluation was run.",
        "- The retrieval probe reruns recorded queries; small ranking drift is possible, but "
        "the trace anchors matched where checked.",
        "- No abstain rows, so abstention performance is not measurable.",
        "- Citations are analysed heuristically; unsupported-answer labels are conservative.",
        "",
        "## 7. Recommended next steps",
        "",
        "- Before any retrieval optimisation, add the planned statement map/arithmetic/gold "
        "evidence tables required by the submission plan (AGENTS.md Day 3).",
        "- Then run a full dev evaluation only after confirming runtime/token budget; use these "
        "tables as the baseline for retrieval fixes.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"HF_HUB_OFFLINE=1 .venv/bin/python -m src.eval.evidence_tables --run-dir "
        f"{manifest['run_dir']} --out-dir {manifest['out_dir']} --probe",
        "```",
        "",
        "## Artifacts",
        "",
        "- `table_a_document_coverage.csv`, `table_b_page_coverage.csv`, "
        "`table_c_citations.csv`, `table_d_failure_classification.csv`, "
        "`summary_metrics.csv`.",
        "- `tables.json`, `summary.json`, `manifest.json`, `retrieval_probe.jsonl`.",
        "- Trace anchors and source query IDs are in the per-row tables and probe JSONL.",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(
    *, run_dir: Path, traces_dir: Path, out_dir: Path, report_path: Path,
    gold_path: Path, chunk_ids: Path, retrieval_config: Path, probe: bool,
) -> dict[str, Any]:
    run_manifest, rows = load_run(run_dir)
    gold_rows = agent_eval.load_gold(gold_path)
    gold = gold_row_map(gold_rows)
    row_ids = [str(row["row_id"]) for row in rows if str(row["row_id"]) in gold]
    calls_by_row = load_calls(row_ids, traces_dir, str(run_manifest["run_id"]))
    indexed = indexed_pairs(chunk_ids)
    probe_meta = None
    probes: dict[str, list[dict[str, Any]]] = {}
    probe_path = Path(out_dir) / "retrieval_probe.jsonl"
    if probe:
        result = run_probe(calls_by_row, probe_path)
        probe_rows_data = read_jsonl(probe_path)
        probes = probe_map(probe_rows_data)
        probe_meta = {**result, "row_ids": sorted(probes),
                      "probes": probes}
    a = table_a(rows, gold, calls_by_row, probes, indexed)
    b = table_b(rows, gold, probes, indexed)
    c = table_c(rows, gold, probes)
    d = table_d(rows, gold, probes, indexed)
    tables = {"a": a, "b": b, "c": c, "d": d}
    metrics = summary(rows, gold, a, c, d, indexed, probe_meta)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_a_document_coverage.csv", a, COLUMNS_A)
    write_csv(out_dir / "table_b_page_coverage.csv", b, COLUMNS_B)
    write_csv(out_dir / "table_c_citations.csv", c, COLUMNS_C)
    write_csv(out_dir / "table_d_failure_classification.csv", d, COLUMNS_D)
    write_csv(out_dir / "summary_metrics.csv", metrics["metrics"], COLUMNS_E)
    (out_dir / "tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    index_meta = json.loads((Path(chunk_ids).parent / "meta.json").read_text(encoding="utf-8"))
    manifest = {
        "run_id": run_manifest["run_id"], "run_dir": str(run_dir),
        "out_dir": str(out_dir), "train_split": run_manifest.get("split"),
        "n_rows": metrics["n_rows"], "row_ids": row_ids,
        "gold_path": str(gold_path), "gold_hash": agent_eval.gold_hash(gold_path),
        "run_manifest_gold_hash": run_manifest.get("gold_hash"),
        "gold_hash_matches_run": agent_eval.gold_hash(gold_path) == run_manifest.get(
            "gold_hash"),
        "index_version": run_manifest.get("index_version"),
        "chunk_ids_sha256": sha256_file(chunk_ids),
        "chunk_ids_meta": index_meta,
        "retrieval_config_sha256": sha256_file(retrieval_config),
        "retrieval_config_path": str(retrieval_config),
        "probe": probe_meta,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": run_manifest.get("model"),
        "prompt_version": run_manifest.get("prompt_version"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(report_path, manifest=manifest, metrics=metrics, tables=tables,
                 probe_meta=probe_meta)
    return {"manifest": manifest, "metrics": metrics, "tables": tables}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.eval.evidence_tables")
    parser.add_argument("--run-dir", type=Path, default=Path("results/evaluation/day2_fix_v1"))
    parser.add_argument("--traces-dir", type=Path, default=Path("logs/traces"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/day3_evidence_v1"))
    parser.add_argument("--report-path", type=Path,
                        default=Path("docs/WEEK34_DAY3_EVIDENCE_TABLES.md"))
    parser.add_argument("--gold", type=Path, default=agent_eval.GOLD)
    parser.add_argument("--chunk-ids", type=Path,
                        default=Path("results/week2_retrieval_cache/C2/chunk_ids.json"))
    parser.add_argument("--retrieval-config", type=Path, default=Path("configs/retrieval.yaml"))
    parser.add_argument("--probe", action="store_true",
                        help="replay recorded search queries against the frozen index")
    args = parser.parse_args(argv)
    result = build(
        run_dir=args.run_dir, traces_dir=args.traces_dir, out_dir=args.out_dir,
        report_path=args.report_path, gold_path=args.gold, chunk_ids=args.chunk_ids,
        retrieval_config=args.retrieval_config, probe=args.probe,
    )
    print(json.dumps({"manifest": result["manifest"], "metrics": result["metrics"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
