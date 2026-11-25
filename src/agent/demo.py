from __future__ import annotations

import argparse
import json
import sys

from .service import MissingEnvironmentError, run_question
from .tools import warm_index


def _print_json(label: str, value: object) -> None:
    print(f"{label}:")
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one IFR-QA agent turn.")
    parser.add_argument("query", help="Question to ask the agent")
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Persistent checkpoint thread id (default: default)",
    )
    args = parser.parse_args(argv)

    try:
        warm_index()
        result = run_question(args.query, session_id=args.thread_id)
    except MissingEnvironmentError as exc:
        print(
            "Missing required environment variables: " + ", ".join(exc.missing),
            file=sys.stderr,
        )
        print("Set them in .env or the shell; see README.md.", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    state = result.state
    _print_json("Plan", state.get("plan", []))
    _print_json("Tool calls", state.get("tool_results", []))
    for entry in state.get("tool_results", []):
        if entry.get("tool") != "search_index":
            continue
        res = entry.get("result")
        hits = res.get("hits", []) if isinstance(res, dict) else (res or [])
        modes = [hit.get("match_mode", "none") for hit in hits if isinstance(hit, dict)]
        info = res.get("match_info", {}) if isinstance(res, dict) else {}
        print(f"Match mode ({entry.get('args', {}).get('company')}): {modes}")
        if info:
            print(f"Match info: {json.dumps(info, ensure_ascii=False)}")
    print("Final answer:")
    print(result.answer)
    final = state.get("final_answer") or {}
    _print_json("Synthesis", final)
    _print_json("Citations", result.citations)
    print(f"Insufficient evidence: {final.get('insufficient_evidence')}")
    _print_json("Critique", state.get("critique", {}))
    print(f"Steps used: {state.get('steps_used', 0)}")
    print(f"Stop reason: {state.get('stop_reason', '')}")
    print(f"Token count: {result.token_count}")
    _print_json("Errors", state.get("errors", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
