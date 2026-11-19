# src/rag_ingestion — contract consumer (boundary package)

`rag_ingestion` is the future home of chunking, embedding and indexing. It
exists as a **boundary package**: it consumes the data pipeline's published
artefacts and nothing else.

- `data_pipeline` = the existing `src/ingestion` package (acquisition,
  manual_ingest, gate, ledger, page/table extraction, freeze). It stays where
  it is; nothing was moved or renamed.
- `rag_ingestion` = this package. It reads ONLY the contract:

    data/corpus/ledger.jsonl                 (kept document rows)
    data/corpus/extract/page_extracts/*.pages.jsonl
    data/corpus/extract/table_records/*.tables.jsonl
    freeze manifest JSONL (e.g. .temp/session-*/freeze_manifest.jsonl)

- The one-way flow is enforced by `tests/unit/test_rag_ingestion_boundary.py`:
  no import of `src.ingestion` (or any `data_pipeline` module) is allowed here
  except the contract reader itself, which is this package's `contract.py`.
- The contract file formats and the freeze rule are documented in
  `docs/ingestion_contract.md`.

Rule of thumb: rag_ingestion code must never import from `src.ingestion*`;
if it needs more structure, the contract doc and `contract.py` grow together,
with the data_pipeline side publishing the additional artefact.
