"""Run one query through the reflection loop and print the trace."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.llm_client import EndpointLLMClient
from src.agents.reflection_loop import run_reflection
from src.config import load_reflection_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflection loop demo")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_reflection_config()
    result = run_reflection(
        EndpointLLMClient(), args.query,
        max_iterations=args.max_iterations,
        max_total_tokens=config.max_total_tokens,
    )
    print(json.dumps({
        "query": args.query,
        "final_answer": result.final_answer,
        "passed": result.passed,
        "stopped_reason": result.stopped_reason,
        "iterations": result.iterations,
        "total_tokens": result.total_tokens,
        "trace": [vars(step) for step in result.trace],
    }, indent=2))


if __name__ == "__main__":
    main()
