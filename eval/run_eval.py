import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph import DEFAULT_TOOLS, AgentGraph
from src.agent.llm import EnvLLM

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

SAMPLE_SEED = 42
FIELDNAMES = [
    "id", "question", "page", "correct", "abstain_ok", "abstain", "winner_ok",
    "tokens", "seconds", "stop_reason", "errors", "got", "expected", "figures",
]


def _norm(value: object) -> str:
    text = str(value or "").replace(",", "").strip()
    try:
        return format(Decimal(text).normalize(), "f")
    except (InvalidOperation, ValueError):
        return text


def _as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def _norm_label(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _figure_values(final_answer: dict, requested: list[dict[str, str]]) -> set[str]:
    wanted = {
        (
            str(item.get("company_id") or "").lower(),
            _norm_label(item.get("label")),
        )
        for item in requested
        if isinstance(item, dict)
    }
    chosen: dict[tuple[str, str], str] = {}
    for fig in final_answer.get("figures", []):
        if not isinstance(fig, dict):
            continue
        key = (
            str(fig.get("company_id") or "").lower(),
            _norm_label(fig.get("label")),
        )
        if key in wanted and key not in chosen:
            chosen[key] = _norm(fig.get("value"))
    return {value for value in chosen.values() if value}


def _computed_values(final_answer: dict) -> set[str]:
    computed = final_answer.get("computed_values") or []
    values = computed if isinstance(computed, list) else [computed]
    return {norm for norm in (_norm(value) for value in values) if norm}


def _winner_ok(final_answer: dict, expected_winner: str) -> bool:
    if not expected_winner:
        return True
    if expected_winner in str(final_answer.get("answer", "")):
        return True
    return any(
        str(fig.get("company_id") or "").lower() == expected_winner.lower()
        for fig in final_answer.get("figures", [])
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _evaluate_row(graph: AgentGraph, row: dict, debug: bool = False) -> dict:
    start = time.monotonic()
    thread_id = f"eval-{time.time_ns()}-{row['id']}-{threading.get_ident()}"
    state = graph.run(row["question"], thread_id=thread_id)
    seconds = time.monotonic() - start
    token_count = int(state.get("token_count", 0) or 0)
    final = state.get("final_answer") or {}
    figures = final.get("figures") or []
    figures_json = json.dumps(figures, ensure_ascii=False, default=str)
    requested = state.get("requested") or []
    figure_values = _figure_values(final, requested)
    computed_values = _computed_values(final)
    got = figure_values | computed_values
    expected = {
        _norm(value)
        for value in row.get("expected_values", [])
        if _norm(value)
    }
    correct = expected <= got and figure_values <= expected
    winner_ok = _winner_ok(final, str(row.get("expected_winner") or "").strip())
    abstain = bool(row.get("abstain"))
    abstain_ok = bool(final.get("insufficient_evidence")) if abstain else not bool(
        final.get("insufficient_evidence")
    )
    row_correct = correct and winner_ok and abstain_ok
    errors = state.get("errors") or []
    if debug:
        print(
            f"[{row['id']}] final_answer: "
            f"{json.dumps(final, ensure_ascii=False, default=str)}"
        )
        print(f"[{row['id']}] figures: {figures_json}")
        print(f"[{row['id']}] stop_reason: {state.get('stop_reason', '')}")
        print(f"[{row['id']}] token_count: {token_count}")
        print(
            f"[{row['id']}] errors: "
            f"{json.dumps(errors, ensure_ascii=False, default=str)}"
        )
        print(f"[{row['id']}] got: {sorted(got)} expected: {sorted(expected)}")
    return {
        "id": row["id"],
        "question": str(row.get("question") or ""),
        "page": "" if row.get("page") is None else row["page"],
        "correct": row_correct,
        "abstain_ok": abstain_ok,
        "abstain": abstain,
        "winner_ok": winner_ok,
        "tokens": token_count,
        "seconds": round(seconds, 3),
        "stop_reason": str(state.get("stop_reason", "")),
        "errors": ";".join(str(error) for error in errors),
        "got": json.dumps(sorted(got), ensure_ascii=False),
        "expected": json.dumps(sorted(expected), ensure_ascii=False),
        "figures": figures_json,
    }


def _summary(path: Path) -> tuple[int, int, int, float]:
    rows, _ = _read_csv(path)
    total = len(rows)
    correct = sum(1 for row in rows if _as_bool(row.get("correct")))
    abstain_ok = sum(1 for row in rows if _as_bool(row.get("abstain_ok")))
    tokens = sum(int(float(row.get("tokens") or 0)) for row in rows)
    return total, correct, abstain_ok, tokens / max(1, total)


def run_eval(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the IFR-QA eval harness.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N rows")
    parser.add_argument("--debug", action="store_true", help="Print agent state and comparisons")
    parser.add_argument("--resume", action="store_true", help="Skip ids already in the CSV")
    parser.add_argument("--workers", type=int, default=4, help="Thread pool size")
    parser.add_argument("--sample", type=int, default=None, help="Random sample with fixed seed")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path")
    args = parser.parse_args(argv)

    index_path = os.environ.get("AGENT_INDEX_PATH", "")
    index_key = "".join(
        c if c.isalnum() else "_" for c in Path(index_path).name
    )[:60] if index_path else "index"
    gold_path = Path(__file__).resolve().parent / "gold.jsonl"
    rows = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.sample is not None:
        ids = [row["id"] for row in rows]
        chosen = set(random.Random(SAMPLE_SEED).sample(ids, min(args.sample, len(ids))))
        rows = [row for row in rows if row["id"] in chosen]

    if args.out is not None:
        out_path = args.out
    elif args.resume:
        candidates = sorted(
            Path("eval").glob(f"results_{index_key}_*.csv"),
            key=lambda path: path.stat().st_mtime,
        )
        out_path = candidates[-1] if candidates else Path("eval") / (
            f"results_{index_key}_{int(time.time())}.csv"
        )
    else:
        out_path = Path("eval") / f"results_{index_key}_{int(time.time())}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows, existing_fields = _read_csv(out_path) if args.resume else ([], [])
    existing_ids = {row["id"] for row in existing_rows}
    pending = [row for row in rows if row["id"] not in existing_ids]
    fieldnames = existing_fields or FIELDNAMES
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    graph = AgentGraph(llm=EnvLLM(), tools=DEFAULT_TOOLS, checkpointer=MemorySaver())
    write_lock = threading.Lock()
    print_lock = threading.Lock()

    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
            handle.flush()

        def _run_row(row: dict) -> dict:
            record = _evaluate_row(graph, row, args.debug)
            with write_lock:
                writer.writerow(record)
                handle.flush()
            if not _as_bool(record["correct"]):
                with print_lock:
                    print(
                        f"FAIL {record['id']}: got={record['got']} "
                        f"expected={record['expected']} figures={record['figures']}"
                    )
            return record

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(_run_row, row) for row in pending]
            for future in as_completed(futures):
                future.result()

    total, correct_rows, abstain_ok, mean_tokens = _summary(out_path)
    accuracy = correct_rows / max(1, total)
    print(
        f"Skipped {len(existing_ids)} ids; ran {len(pending)} rows with "
        f"{max(1, args.workers)} workers"
    )
    print(f"Accuracy: {correct_rows}/{total} ({accuracy:.1%}), mean tokens: {mean_tokens:.1f}")
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_eval())
