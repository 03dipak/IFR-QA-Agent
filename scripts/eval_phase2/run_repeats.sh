#!/usr/bin/env bash
set -euo pipefail
cd /home/dipak/agentic/IFR-QA-Agent
LOG=/tmp/opencode/repeats_$(date -u +%Y%m%dT%H%M%SZ).log
ARMS=(B0 B1 S1 S2 S3 B2-filtered)
for arm in "${ARMS[@]}"; do
  for rep in 1 2 3; do
    RID="repeat_${arm}_r${rep}_$(date -u +%Y%m%dT%H%M%SZ)"
    echo "=== $RID" >> "$LOG"
    PYTHONPATH=. ARM="$arm" RUN_ID="$RID" .venv/bin/python - <<'PY' >> "$LOG" 2>&1
from src.eval.prompting.runner import run_arm
import os
m=run_arm(os.environ['ARM'], run_id=os.environ['RUN_ID'], live=True)
print(m)
PY
    echo "=== done $RID" >> "$LOG"
    sleep 1
  done
done
echo ALL_DONE >> "$LOG"
