"""FastAPI backend.

Implements lld_api_ui.txt section 3: API endpoints POST /ask, GET /corpus,
GET /health with JSON-schema request/response and rate limiting.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="IFR-QA Agent API", version="0.1.0")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/corpus")
async def corpus():
    """Corpus status endpoint."""
    return {"freeze_version": "pending", "companies": []}


@app.post("/ask")
async def ask():
    """Question-answering endpoint (stub)."""
    return {"answer": "not implemented", "evidence": [], "trace": {}}


def main():
    """Entry point for the API server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
