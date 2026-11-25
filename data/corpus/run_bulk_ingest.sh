#!/usr/bin/env bash
# Resume-safe bulk ingestion: one company CSV at a time, sequential.
set -u
cd /home/dipak/agentic/IFR-QA-Agent
LOG=data/corpus/bulk_ingest_20261104.log
TIMING=data/corpus/bulk_ingest_timing.csv
echo "company,start_epoch,end_epoch,csv_rows,rc" > "$TIMING"
for f in data/corpus/pending/*.csv; do
  c=$(basename "$f" .csv)
  rows=$(($(wc -l < "$f") - 1))
  s=$(date +%s)
  echo "=== START $c rows=$rows $(date -u +%FT%TZ)" >> "$LOG"
  .venv/bin/python -m src.ingestion.pipeline bulk \
    --candidates "$f" --storage-dir data/corpus/raw >> "$LOG" 2>&1
  rc=$?
  e=$(date +%s)
  echo "$c,$s,$e,$rows,$rc" >> "$TIMING"
  echo "=== END $c rc=$rc elapsed=$((e-s))s $(date -u +%FT%TZ)" >> "$LOG"
done
echo "=== ALL_DONE $(date -u +%FT%TZ)" >> "$LOG"
