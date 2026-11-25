#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-results/full_index_v1/page_hash384}"
shift || true

for required in data/corpus/freeze_v1.json data/corpus/extract/page_extracts data/corpus/extract/table_records; do
  if [[ ! -e "$required" ]]; then
    echo "Missing corpus input: $required" >&2
    echo "Run ingestion/extraction first or mount an existing index." >&2
    exit 2
  fi
done

uv run python -m src.rag_ingestion.full_index build --out "$OUT" --embedder hash384 "$@"
