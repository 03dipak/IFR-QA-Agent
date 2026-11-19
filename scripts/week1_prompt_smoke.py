"""Week-1 prompt baseline smoke (dev split only; SMOKE_NOT_SCORED)."""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.contexts import load_few_shot
from src.eval.prompting.inputs import load_dev_rows
from src.eval.prompting.runner import run_arm, rendered_prompt_for

LIMIT = 10
ARMS = ("B0", "B1")


def _first_record(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    raise RuntimeError(f"no records in {path}")


def main() -> int:
    rows = load_dev_rows(limit=LIMIT)
    few_shot = load_few_shot(n=3)
    manifests = []
    for arm in ARMS:
        manifest = run_arm(arm, rows=rows, smoke=True, live=True, limit=LIMIT,
                           few_shot=few_shot if arm == "B0" else None)
        manifests.append(manifest)
        prompt = rendered_prompt_for(arm, rows[0], few_shot=few_shot if arm == "B0" else None)
        rec = _first_record(Path(manifest["output_path"]))
        print("=" * 72)
        print("rendered prompt:")
        print(prompt[:2000])
        print(f"ARM {arm}  row={rec['row_id']}  status={rec['status']}  "
              f"latency_ms={rec['latency_ms']}")
        print("token usage:", rec.get("usage"))
        print("score:", rec.get("score"))
    print("SMOKE_NOT_SCORED")
    for m in manifests:
        print(m["arm"], m["run_id"], m["output_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
