import builtins
import json

import numpy as np

from src.rag_ingestion import chunking
from src.rag_ingestion import full_index as fi


def test_embed_text_is_deterministic_and_normalized():
    first = fi.embed_text("Revenue 1,23,456.78 and Q1FY2026")
    second = fi.embed_text("Revenue 1,23,456.78 and q1fy2026")
    assert np.array_equal(first, second)
    assert first.shape == (fi.DIM,)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-6


class _FakeResponse:
    def __init__(self, payload, error_text=None, status_code=None):
        self.payload = payload
        self.error_text = error_text
        self.text = error_text or ""
        self.status_code = status_code or (422 if error_text else 200)

    def raise_for_status(self):
        if self.error_text:
            import requests
            raise requests.HTTPError(self.error_text, response=self)

    def json(self):
        return self.payload


def test_bge_small_embedder_uses_endpoint_and_normalizes(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "bge-small-en-v1.5")
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "test-key")
    monkeypatch.setattr(fi, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fi.time, "sleep", lambda seconds: None)

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return _FakeResponse({"data": [
            {"index": 1, "embedding": [0.0, 3.0, 4.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
        ]})

    monkeypatch.setattr(fi.requests, "post", fake_post)
    vectors = fi.embed_texts(["a", "b"], dim=3, embedder="bge-small")

    assert captured["url"] == "https://example.invalid/v1/embeddings"
    assert captured["json"] == {"model": "bge-small-en-v1.5", "input": ["a", "b"]}
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [1.0, 0.0, 0.0])
    assert np.allclose(vectors[1], [0.0, 0.6, 0.8])


def test_bge_small_isolates_one_poisoned_text_in_256(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_EMBEDDINGS_URL", "https://example.invalid/v1")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "bge-small-en-v1.5")
    monkeypatch.setenv("EMBEDDINGS_API_KEY", "test-key")
    monkeypatch.setattr(fi, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(fi.time, "sleep", lambda seconds: None)

    poison = "POISON\x00INPUT"
    texts = [f"normal text {index}" for index in range(256)]
    texts[137] = poison
    chunk_ids = [f"chunk-{index}" for index in range(256)]
    failure_path = tmp_path / "failed_chunks.jsonl"
    requests_seen: list[list[str]] = []

    def fake_post(url, **kwargs):
        batch = kwargs["json"]["input"]
        requests_seen.append(batch)
        if any("POISON" in text for text in batch):
            return _FakeResponse({}, "poisoned embedding input", status_code=422)
        return _FakeResponse({"data": [
            {"index": index, "embedding": [1.0, 0.0, 0.0]}
            for index in range(len(batch))
        ]})

    monkeypatch.setattr(fi.requests, "post", fake_post)
    vectors = fi.embed_texts(
        texts, dim=3, embedder="bge-small",
        chunk_ids=chunk_ids, batch_index=7, failure_path=failure_path,
    )

    assert vectors.shape == (256, 3)
    assert np.allclose(vectors[137], np.zeros(3))
    assert np.allclose(np.delete(vectors, 137, axis=0), np.tile([1.0, 0.0, 0.0], (255, 1)))
    assert any(batch == ["POISON INPUT"] for batch in requests_seen)
    lines = failure_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    failure = json.loads(lines[0])
    assert failure["chunk_id"] == "chunk-137"
    assert failure["batch_index"] == 7
    assert "422" in failure["reason"]


def _setup_chunks(tmp_path, monkeypatch):
    doc = {"document_id": "a" * 64, "company_id": "bhel",
           "fiscal_year": "FY2026", "doc_type": "annual", "variant": "standalone"}
    missing = {"document_id": "b" * 64, "company_id": "srf",
               "fiscal_year": "FY2026", "doc_type": "annual", "variant": "standalone"}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"manifest_sha256": "m" * 64,
                                  "documents": [doc, missing]}))
    monkeypatch.setattr(chunking, "FREEZE", str(freeze))

    out = tmp_path / "index"
    out.mkdir()
    chunks = [
        {"chunk_id": "a-page", "id": "a-page", "text": "revenue 123456.78",
         "meta": {"document_id": doc["document_id"], "page": 1,
                  "chunk_type": "page"}},
        {"chunk_id": "a-table", "id": "a-table", "text": "profit 100",
         "meta": {"document_id": doc["document_id"], "page": 1,
                  "chunk_type": "table"}},
    ]
    (out / "chunks.jsonl").write_text(
        "\n".join(json.dumps(chunk, sort_keys=True) for chunk in chunks) + "\n",
        encoding="utf-8")
    return out


def test_full_index_build_and_search(tmp_path, monkeypatch):
    out = _setup_chunks(tmp_path, monkeypatch)
    meta = fi.build(out, batch_size=1)
    assert meta["documents_total"] == 2
    assert meta["documents_indexed"] == 1
    assert meta["documents_skipped"] == 1
    assert meta["chunks"] == 2

    index = fi.FullVectorIndex(out)
    results = index.search("revenue 100", k=2)
    assert len(results) == 2
    assert {r["meta"]["chunk_type"] for r in results} == {"page", "table"}


def test_full_index_resume_skips_completed_batches(tmp_path, monkeypatch):
    out = _setup_chunks(tmp_path, monkeypatch)
    first = fi.build(out, batch_size=1, checkpoint_batches=1)
    assert first["resumed_from_chunk"] == 0
    assert first["embedding_batches"] == 2

    progress = json.loads((out / fi.PROGRESS_NAME).read_text(encoding="utf-8"))
    assert progress["chunks_done"] == 2
    assert progress["status"] == "complete"

    def fail_embed(texts, dim=fi.DIM, embedder=fi.DEFAULT_EMBEDDER):
        raise AssertionError("completed batches were re-embedded")

    monkeypatch.setattr(fi, "embed_texts", fail_embed)
    resumed = fi.build(out, batch_size=1, checkpoint_batches=1)
    assert resumed["resumed_from_chunk"] == 2
    assert resumed["embedding_batches"] == 2
    assert resumed["chunks"] == 2


def test_append_missing_chunks_detects_and_skips_existing(tmp_path, monkeypatch):
    present = "a" * 64
    new = "c" * 64
    bad = "d" * 64
    docs = [
        {"document_id": present, "company_id": "bhel", "fiscal_year": "FY2026",
         "doc_type": "annual", "variant": "standalone"},
        {"document_id": new, "company_id": "srf", "fiscal_year": "FY2026",
         "doc_type": "annual", "variant": "standalone"},
        {"document_id": bad, "company_id": "dabur", "fiscal_year": "FY2026",
         "doc_type": "annual", "variant": "standalone"},
    ]
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"manifest_sha256": "m" * 64,
                                  "documents": docs}))
    monkeypatch.setattr(chunking, "FREEZE", str(freeze))

    page_dir = tmp_path / "page_extracts"
    page_dir.mkdir()
    (page_dir / f"{new}.pages.jsonl").write_text(
        json.dumps({"pdf_page_index": 1, "page_text": "new revenue"}) + "\n")
    table_dir = tmp_path / "table_records"
    table_dir.mkdir()
    monkeypatch.setattr(chunking, "PAGE_DIR", str(page_dir))
    monkeypatch.setattr(chunking, "TABLE_DIR", str(table_dir))

    out = tmp_path / "index"
    out.mkdir()
    (out / "chunks.jsonl").write_text(
        json.dumps({"chunk_id": f"{present}:p1", "id": f"{present}:p1",
                    "text": "old revenue",
                    "meta": {"document_id": present, "page": 1,
                             "chunk_type": "page"}}, sort_keys=True) + "\n")

    report = fi.append_missing_chunks(out)
    assert report["docs_total"] == 3
    assert report["docs_missing"] == 2
    assert report["docs_appended"] == 1
    assert report["docs_failed"] == 1
    assert report["docs_present_after"] == 2

    present_ids = fi.doc_ids_in_chunks(out / "chunks.jsonl")
    assert present in present_ids
    assert new in present_ids
    assert bad not in present_ids
    assert fi.read_failed_docs(fi.failed_docs_path(out)) == [bad]


def test_full_index_falls_back_to_numpy_when_faiss_blocked(tmp_path, monkeypatch):
    chunks = [
        {"chunk_id": "a", "company": "acme", "page": 1},
        {"chunk_id": "b", "company": "acme", "page": 2},
        {"chunk_id": "c", "company": "acme", "page": 3},
    ]
    offsets = []
    with (tmp_path / "chunks.jsonl").open("wb") as handle:
        for chunk in chunks:
            offset = handle.tell()
            encoded = json.dumps(chunk).encode("utf-8") + b"\n"
            handle.write(encoded)
            offsets.append([offset, len(encoded)])
    (tmp_path / "chunk_ids.json").write_text(json.dumps([c["chunk_id"] for c in chunks]))
    (tmp_path / "chunk_offsets.json").write_text(json.dumps(offsets))
    (tmp_path / "meta.json").write_text(json.dumps({"dim": 3, "embedder": "hash384"}))
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", vectors)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("faiss blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(fi, "embed_text", lambda text, dim, embedder: np.array([1.0, 0.0, 0.0]))

    index = fi.FullVectorIndex(tmp_path)
    hits = index.search("revenue", k=2)
    assert index.backend == "numpy"
    assert [hit["chunk_id"] for hit in hits] == ["a", "b"]
    assert hits[0]["score"] == 1.0
