"""CLI: python -m src.agents.run "question"."""

from __future__ import annotations

import argparse
import json

from src.agents.graph import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track B agent MVP")
    parser.add_argument("question", help="financial question to answer")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args(argv)
    result = run_agent(args.question, session_id=args.session_id)
    print("ANSWER:", result.answer)
    print("CITATIONS:", json.dumps(result.citations))
    print("STEPS:", json.dumps(result.steps, indent=2))
    print("TOKENS:", result.tokens)
    print("LATENCY_MS:", result.latency_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
