"""RAG ingestion boundary (contract consumer side).

This package is the *rag_ingestion* side of the one-way module boundary:
it reads only the published contract artefacts (ledger kept rows, page and
table extract JSONL, freeze manifest) and imports nothing from the
data_pipeline package, which is the existing ``src/ingestion``.

See README.md (package scope) and docs/ingestion_contract.md (formats).
"""
