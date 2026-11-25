from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests

from . import chunking
from .tokenizer import TOKENIZER_VERSION, tokenize

DEFAULT_OUT = Path("results/full_index_v1/page_hash384")
DIM = 384
MODEL = "local:hash384-v1"
BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
EMBEDDERS = ("hash384", "bge-small")
DEFAULT_EMBEDDER = "hash384"
CLI_EMBEDDER = "bge-small"
ETA_BATCHES = 20
EMBEDDING_MAX_INPUTS = 16
EMBEDDING_MAX_CHARS = 1000
SCOPE = "freeze_v1_all_documents"
PROGRESS_NAME = "embedding_progress.json"
FAILED_DOCS_NAME = "failed_docs.txt"
FAISS_BATCH = 4096
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_env() -> dict[str, str]:
    values = dict(os.environ)
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), value.strip().strip("\"'"))
    return values


def _model_for_embedder(embedder: str) -> str:
    if embedder == "hash384":
        return MODEL
    if embedder == "bge-small":
        return _load_env().get("EMBEDDINGS_MODEL", "bge-small-en-v1.5")
    raise ValueError(f"unsupported embedder: {embedder}")


def index_version(variant: str, dim: int = DIM,
                  embedder: str = DEFAULT_EMBEDDER) -> str:
    payload = {
        "scope": SCOPE,
        "manifest_sha256": chunking.freeze_manifest_sha256(),
        "variant": variant,
        "chunk_config": chunking.CHUNK_CONFIG,
        "embedder": embedder,
        "model": _model_for_embedder(embedder),
        "dim": dim,
        "tokenizer": TOKENIZER_VERSION,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _embed_hash(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        h = zlib.crc32(token.encode("utf-8"))
        vec[h % dim] += 1.0 if h & 0x10000 else -1.0
    norm = np.linalg.norm(vec)
    if norm:
        vec /= norm
    return vec.astype(np.float32)


_LAST_EMBED_CALL = 0.0
_BGE_MAX_RETRIES = 5
_FALLBACK_MAX_CHARS = 400


def _log_bge_error(status: Any, body: str, batch_index: int,
                   texts: list[str]) -> None:
    body_preview = str(body)[:500].replace("\n", "\\n")
    print(f"BGE HTTP {status} batch_index={batch_index} "
          f"response_body={body_preview}", flush=True)
    for offset, text in enumerate(texts):
        preview = text[:80].replace("\n", "\\n")
        print(f"BGE HTTP {status} batch_index={batch_index} input[{offset}] "
              f"len={len(text)} preview={preview!r}", flush=True)


def _sanitize_embedding_text(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:_FALLBACK_MAX_CHARS] or "empty"


def _record_failed_chunk(path: Path | None, chunk_id: str, reason: str,
                         batch_index: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({
        "batch_index": batch_index,
        "chunk_id": chunk_id,
        "reason": reason[:500],
    }, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _bge_request(
    texts: list[str],
    dim: int,
    env: dict[str, str],
    verify: str | bool,
    *,
    chunk_ids: list[str] | None = None,
    batch_index: int = 0,
    failure_path: Path | None = None,
    fallback_used: bool = False,
) -> np.ndarray:
    global _LAST_EMBED_CALL
    base = env.get("LLM_EMBEDDINGS_URL")
    if not base:
        raise RuntimeError("LLM_EMBEDDINGS_URL is not configured")
    api_key = env.get("EMBEDDINGS_API_KEY") or env.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("EMBEDDINGS_API_KEY is not configured")
    ids = chunk_ids or [f"batch-{batch_index}-input-{index}"
                        for index in range(len(texts))]
    wait = 1.0 - (time.time() - _LAST_EMBED_CALL)
    if wait > 0:
        time.sleep(wait)
    _LAST_EMBED_CALL = time.time()
    response = None
    for attempt in range(_BGE_MAX_RETRIES):
        try:
            response = requests.post(
                base.rstrip("/") + "/embeddings",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": env.get("EMBEDDINGS_MODEL", "bge-small-en-v1.5"),
                      "input": texts},
                timeout=120,
                verify=verify,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            error_response = getattr(exc, "response", None)
            status = getattr(error_response, "status_code", None)
            body = getattr(error_response, "text", None) or str(exc)
            _log_bge_error(status or "timeout", body, batch_index, texts)
            if status == 422:
                if len(texts) > 1:
                    middle = len(texts) // 2
                    print(f"BGE 422 splitting batch_index={batch_index} into "
                          f"{middle} + {len(texts) - middle} inputs", flush=True)
                    left = _bge_request(
                        texts[:middle], dim, env, verify,
                        chunk_ids=ids[:middle], batch_index=batch_index,
                        failure_path=failure_path,
                    )
                    right = _bge_request(
                        texts[middle:], dim, env, verify,
                        chunk_ids=ids[middle:], batch_index=batch_index,
                        failure_path=failure_path,
                    )
                    return np.vstack([left, right])
                if not fallback_used:
                    fallback = _sanitize_embedding_text(texts[0])
                    print(f"BGE 422 sanitizing chunk_id={ids[0]} "
                          f"len={len(fallback)}", flush=True)
                    return _bge_request(
                        [fallback], dim, env, verify,
                        chunk_ids=[ids[0]], batch_index=batch_index,
                        failure_path=failure_path, fallback_used=True,
                    )
                reason = f"422 after fallback: {body[:500]}"
                print(f"BGE 422 fallback failed chunk_id={ids[0]}", flush=True)
                _record_failed_chunk(failure_path, ids[0], reason, batch_index)
                return np.zeros((1, dim), dtype=np.float32)
            status_code = status if isinstance(status, int) else 0
            retryable = status_code == 429 or 500 <= status_code < 600 or status is None
            if attempt == _BGE_MAX_RETRIES - 1 or not retryable:
                raise
            time.sleep(2 ** attempt)
    assert response is not None
    rows = sorted(response.json()["data"], key=lambda row: row["index"])
    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    if vectors.shape != (len(texts), dim):
        raise ValueError(f"embedding shape {vectors.shape} != {(len(texts), dim)}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding endpoint returned a zero vector")
    return vectors / norms


def _embed_bge_batch(
    texts: list[str],
    dim: int,
    chunk_ids: list[str] | None = None,
    batch_index: int = 0,
    failure_path: Path | None = None,
) -> np.ndarray:
    env = _load_env()
    ca = env.get("EMBEDDINGS_CA_CERT")
    verify: str | bool = True
    if ca:
        ca_path = Path(ca)
        if not ca_path.is_absolute():
            ca_path = PROJECT_ROOT / ca_path
        if ca_path.is_file():
            verify = str(ca_path)
    blocks: list[np.ndarray] = []
    for start in range(0, len(texts), EMBEDDING_MAX_INPUTS):
        end = min(start + EMBEDDING_MAX_INPUTS, len(texts))
        batch = [text[:EMBEDDING_MAX_CHARS] for text in texts[start:end]]
        if chunk_ids is not None:
            batch_ids = list(chunk_ids[start:end])
        else:
            batch_ids = [f"batch-{batch_index}-input-{index}"
                         for index in range(start, end)]
        blocks.append(_bge_request(
            batch, dim, env, verify,
            chunk_ids=batch_ids, batch_index=batch_index,
            failure_path=failure_path,
        ))
    return np.vstack(blocks)


def embed_texts(
    texts: list[str],
    dim: int = DIM,
    embedder: str = DEFAULT_EMBEDDER,
    *,
    chunk_ids: list[str] | None = None,
    batch_index: int = 0,
    failure_path: Path | None = None,
) -> np.ndarray:
    if embedder not in EMBEDDERS:
        raise ValueError(f"unsupported embedder: {embedder}")
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    if embedder == "hash384":
        return np.vstack([_embed_hash(text, dim) for text in texts])
    return _embed_bge_batch(
        texts, dim, chunk_ids=chunk_ids,
        batch_index=batch_index, failure_path=failure_path,
    )


def embed_text(text: str, dim: int = DIM,
               embedder: str = DEFAULT_EMBEDDER) -> np.ndarray:
    return embed_texts([text], dim, embedder)[0]


def load_documents(freeze_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(freeze_path or chunking.FREEZE)
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data["documents"], key=lambda d: d["document_id"])


def doc_ids_in_chunks(chunks_path: Path | str) -> set[str]:
    doc_ids: set[str] = set()
    with Path(chunks_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = (record.get("meta") or {}).get("document_id")
            if doc_id:
                doc_ids.add(doc_id)
    return doc_ids


def failed_docs_path(out: Path | str) -> Path:
    return Path(out).parent / FAILED_DOCS_NAME


def read_failed_docs(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def append_missing_chunks(out: Path | str = DEFAULT_OUT, variant: str = "page"
                          ) -> dict[str, Any]:
    if variant != "page":
        raise ValueError(f"unsupported full-index variant: {variant}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chunks_path = out / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"existing chunks file required: {chunks_path}")

    failed_path = failed_docs_path(out)
    failed_path.write_text("", encoding="utf-8")
    docs = load_documents()
    present = doc_ids_in_chunks(chunks_path)
    missing = [doc for doc in docs if doc["document_id"] not in present]
    print(f"chunk completion docs_total={len(docs)} docs_present={len(present)} "
          f"docs_missing={len(missing)}", flush=True)

    appended_docs = 0
    failed_ids: list[str] = []
    processed = 0
    with chunks_path.open("ab") as fh:
        for doc in missing:
            doc_id = doc["document_id"]
            processed += 1
            try:
                chunks = chunking.chunks_for(doc, variant)
                if not chunks:
                    raise ValueError("no chunks produced")
                lines = [
                    json.dumps({"chunk_id": chunk["id"], **chunk}, sort_keys=True) + "\n"
                    for chunk in chunks
                ]
                fh.write("".join(lines).encode("utf-8"))
                fh.flush()
                os.fsync(fh.fileno())
                appended_docs += 1
            except Exception as exc:
                failed_ids.append(doc_id)
                failed_path.write_text("\n".join(failed_ids) + "\n", encoding="utf-8")
                print(f"failed doc {doc_id}: {type(exc).__name__}: {exc}", flush=True)
            if processed % 50 == 0:
                print(f"chunking progress {processed}/{len(missing)} "
                      f"appended={appended_docs} failed={len(failed_ids)}", flush=True)

    final_present = doc_ids_in_chunks(chunks_path)
    report = {
        "docs_total": len(docs),
        "docs_present_before": len(present),
        "docs_missing": len(missing),
        "docs_appended": appended_docs,
        "docs_failed": len(failed_ids),
        "docs_present_after": len(final_present),
        "expected_present": len(docs) - len(failed_ids),
        "failed_docs_file": str(failed_path),
    }
    print("chunk completion " + json.dumps(report, sort_keys=True), flush=True)
    if len(final_present) != report["expected_present"]:
        raise RuntimeError(
            f"chunk coverage mismatch: got {len(final_present)}, "
            f"expected {report['expected_present']}")
    return report


def verify_chunk_coverage(out: Path | str = DEFAULT_OUT) -> dict[str, Any]:
    out = Path(out)
    docs = load_documents()
    present = doc_ids_in_chunks(out / "chunks.jsonl")
    failed = read_failed_docs(failed_docs_path(out))
    expected = len(docs) - len(failed)
    report = {
        "freeze_docs": len(docs),
        "failed_docs": len(failed),
        "distinct_doc_ids": len(present),
        "expected_doc_ids": expected,
        "ok": len(present) == expected,
    }
    print("chunk coverage " + json.dumps(report, sort_keys=True), flush=True)
    if not report["ok"]:
        raise RuntimeError(f"chunk coverage mismatch: {report}")
    return report


def complete(out: Path | str = DEFAULT_OUT, variant: str = "page", dim: int = DIM,
             batch_size: int = 256, checkpoint_batches: int = 10,
             embedder: str = DEFAULT_EMBEDDER) -> dict[str, Any]:
    out = Path(out)
    chunk_report = append_missing_chunks(out, variant)
    coverage_report = verify_chunk_coverage(out)
    for name in (PROGRESS_NAME, "embeddings.npy", "index.faiss", "meta.json"):
        (out / name).unlink(missing_ok=True)
    print("forced full rebuild: removed prior embeddings, FAISS, progress and meta",
          flush=True)
    meta = build(out, variant, dim, batch_size, checkpoint_batches, embedder)
    meta["chunk_completion"] = chunk_report
    meta["chunk_coverage"] = coverage_report
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    return meta


def _scan_chunks(chunks_path: Path, ids_path: Path, offsets_path: Path) -> tuple[
    list[str], list[list[int]], list[str]
]:
    ids: list[str] = []
    offsets: list[list[int]] = []
    doc_ids: set[str] = set()
    seen: set[str] = set()
    with chunks_path.open("rb") as fh:
        while True:
            offset = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = record.get("chunk_id") or record["id"]
            if chunk_id in seen:
                raise ValueError(f"duplicate chunk id: {chunk_id}")
            seen.add(chunk_id)
            ids.append(chunk_id)
            offsets.append([offset, len(line)])
            doc_id = (record.get("meta") or {}).get("document_id")
            if doc_id:
                doc_ids.add(doc_id)
    ids_path.write_text(json.dumps(ids, sort_keys=True), encoding="utf-8")
    offsets_path.write_text(json.dumps(offsets, sort_keys=True), encoding="utf-8")
    return ids, offsets, sorted(doc_ids)


def _read_progress(progress_path: Path, embeddings_path: Path, *, chunks: int,
                   dim: int, batch_size: int, ids_sha256: str, version: str,
                   embedder: str) -> tuple[int, int]:
    if not progress_path.is_file() or not embeddings_path.is_file():
        return 0, 0
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if embeddings.shape != (chunks, dim):
            return 0, 0
    except Exception:
        return 0, 0
    expected = {"chunks": chunks, "dim": dim, "batch_size": batch_size,
                "embedder": embedder, "model": _model_for_embedder(embedder),
                "tokenizer": TOKENIZER_VERSION, "chunk_ids_sha256": ids_sha256,
                "index_version": version}
    if any(progress.get(key) != value for key, value in expected.items()):
        return 0, 0
    done = int(progress.get("chunks_done", 0))
    batches = int(progress.get("batches_done", 0))
    if done < 0 or done > chunks or batches < 0:
        return 0, 0
    if done != chunks and done % batch_size != 0:
        return 0, 0
    if batches * batch_size < done:
        return 0, 0
    return done, batches


def _write_progress(progress_path: Path, *, chunks: int, dim: int, batch_size: int,
                    ids_sha256: str, version: str, embedder: str, chunks_done: int,
                    batches_done: int, elapsed_seconds: float,
                    status: str) -> None:
    _write_json_atomic(progress_path, {
        "chunks": chunks,
        "dim": dim,
        "batch_size": batch_size,
        "embedder": embedder,
        "model": _model_for_embedder(embedder),
        "tokenizer": TOKENIZER_VERSION,
        "chunk_ids_sha256": ids_sha256,
        "index_version": version,
        "chunks_done": chunks_done,
        "batches_done": batches_done,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "status": status,
        "updated_at": time.time(),
    })


def build(out: Path | str = DEFAULT_OUT, variant: str = "page", dim: int = DIM,
          batch_size: int = 256, checkpoint_batches: int = 10,
          embedder: str = DEFAULT_EMBEDDER) -> dict[str, Any]:
    if embedder not in EMBEDDERS:
        raise ValueError(f"unsupported embedder: {embedder}")
    if variant != "page":
        raise ValueError(f"unsupported full-index variant: {variant}")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if checkpoint_batches <= 0:
        raise ValueError("checkpoint_batches must be positive")

    started = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chunks_path = out / "chunks.jsonl"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"existing chunks file required: {chunks_path}")

    ids_path = out / "chunk_ids.json"
    offsets_path = out / "chunk_offsets.json"
    vectors_path = out / "embeddings.npy"
    faiss_path = out / "index.faiss"
    progress_path = out / PROGRESS_NAME

    docs = load_documents()
    ids, offsets, indexed_docs = _scan_chunks(chunks_path, ids_path, offsets_path)
    ids_sha256 = _sha256_file(ids_path)
    version = index_version(variant, dim, embedder)
    done, batches_done = _read_progress(
        progress_path, vectors_path, chunks=len(ids), dim=dim,
        batch_size=batch_size, ids_sha256=ids_sha256, version=version,
        embedder=embedder)

    print(f"reusing chunks.jsonl chunks={len(ids)} documents={len(indexed_docs)} "
          f"embedder={embedder} model={_model_for_embedder(embedder)}", flush=True)
    mode = "r+" if done else "w+"
    embeddings = np.lib.format.open_memmap(
        vectors_path, mode=mode, dtype=np.float32, shape=(len(ids), dim))
    if done:
        print(f"resuming from chunk {done}/{len(ids)} batches_done={batches_done}",
              flush=True)

    embedding_started = time.time()
    processed = done
    total_batches = (len(ids) + batch_size - 1) // batch_size
    batches_this_run = 0
    eta_printed = False
    with chunks_path.open("rb") as fh:
        for start in range(done, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            fh.seek(offsets[start][0])
            chunks = [json.loads(fh.readline()) for _ in range(start, end)]
            texts = [chunk["text"] for chunk in chunks]
            vectors = embed_texts(
                texts, dim, embedder,
                chunk_ids=ids[start:end], batch_index=batches_done,
                failure_path=Path(out) / "failed_chunks.jsonl",
            )
            embeddings[start:end] = vectors.astype(np.float32)
            processed = end
            batches_done += 1
            batches_this_run += 1
            elapsed = time.time() - embedding_started
            rate = (processed - done) / elapsed if elapsed else 0.0
            eta = (len(ids) - processed) / rate if rate else 0.0
            print(f"batch {batches_done}/{total_batches} embedded "
                  f"{processed}/{len(ids)} elapsed={elapsed:.1f}s "
                  f"rate={rate:.0f}/s eta={eta:.0f}s", flush=True)
            if not eta_printed and batches_this_run == ETA_BATCHES:
                batch_rate = batches_this_run / elapsed if elapsed else 0.0
                estimated_total = total_batches / batch_rate if batch_rate else 0.0
                print(f"estimated_total_seconds={estimated_total:.0f} "
                      f"based_on_batches={batches_this_run} "
                      f"batch_rate={batch_rate:.3f}/s", flush=True)
                eta_printed = True
            if batches_done % checkpoint_batches == 0 or processed == len(ids):
                embeddings.flush()
                _write_progress(
                    progress_path, chunks=len(ids), dim=dim, batch_size=batch_size,
                    ids_sha256=ids_sha256, version=version, embedder=embedder,
                    chunks_done=processed, batches_done=batches_done,
                    elapsed_seconds=elapsed,
                    status="complete" if processed == len(ids) else "running")

    embeddings.flush()
    search_index = faiss.IndexFlatIP(dim)
    for start in range(0, len(ids), FAISS_BATCH):
        block = np.asarray(embeddings[start:start + FAISS_BATCH])
        if len(block):
            search_index.add(block)
    faiss.write_index(search_index, str(faiss_path))

    indexed_set = set(indexed_docs)
    skipped_docs = [d["document_id"] for d in docs
                    if d["document_id"] not in indexed_set]
    meta = {
        "scope": SCOPE,
        "variant": variant,
        "embedder": embedder,
        "model": _model_for_embedder(embedder),
        "embedding_kind": (
            "offline deterministic token feature hashing"
            if embedder == "hash384" else
            "sentence-transformers BAAI/bge-small-en-v1.5 via embeddings endpoint"
        ),
        "dim": dim,
        "tokenizer": TOKENIZER_VERSION,
        "index_version": version,
        "freeze_manifest_sha256": chunking.freeze_manifest_sha256(),
        "documents_total": len(docs),
        "documents_indexed": len(indexed_docs),
        "documents_skipped": len(skipped_docs),
        "skipped_document_ids": skipped_docs,
        "document_ids_sha256": hashlib.sha256(
            "\n".join(indexed_docs).encode("utf-8")).hexdigest(),
        "chunks": len(ids),
        "chunk_ids_sha256": ids_sha256,
        "embeddings_sha256": _sha256_file(vectors_path),
        "chunk_build_seconds": 0.0,
        "chunks_reused": True,
        "embedding_seconds": round(time.time() - embedding_started, 2),
        "embedding_batches": batches_done,
        "checkpoint_batches": checkpoint_batches,
        "resumed_from_chunk": done,
        "elapsed_seconds": round(time.time() - started, 2),
        "artifacts": ["chunks.jsonl", "chunk_ids.json", "embeddings.npy",
                      "index.faiss", "chunk_offsets.json", PROGRESS_NAME],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    print(f"wrote full index chunks={len(ids)} docs={len(indexed_docs)} "
          f"faiss={search_index.ntotal} elapsed={meta['elapsed_seconds']}s", flush=True)
    return meta


class FullVectorIndex:
    def __init__(self, cache: Path | str = DEFAULT_OUT):
        self.cache = Path(cache)
        self.meta = json.loads((self.cache / "meta.json").read_text(encoding="utf-8"))
        self.ids = json.loads((self.cache / "chunk_ids.json").read_text(encoding="utf-8"))
        self.embedder = self.meta.get("embedder", "hash384")
        self.embeddings = np.load(self.cache / "embeddings.npy", mmap_mode="r")
        self.index = faiss.read_index(str(self.cache / "index.faiss"))
        raw_offsets = json.loads(
            (self.cache / "chunk_offsets.json").read_text(encoding="utf-8"))
        if isinstance(raw_offsets, dict):
            self.offsets = [raw_offsets[chunk_id] for chunk_id in self.ids]
        else:
            self.offsets = raw_offsets
        if self.embeddings.shape[0] != len(self.ids):
            raise ValueError("full index embeddings and chunk ids disagree")
        if self.index.ntotal != len(self.ids):
            raise ValueError("FAISS index and chunk ids disagree")

    def _chunk_at(self, position: int) -> dict[str, Any]:
        offset, size = self.offsets[position]
        with (self.cache / "chunks.jsonl").open("rb") as fh:
            fh.seek(offset)
            chunk: dict[str, Any] = json.loads(fh.read(size).decode("utf-8"))
            return chunk

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        query_text = (BGE_QUERY_PREFIX + query) if self.embedder == "bge-small" else query
        vector = embed_text(query_text, int(self.meta["dim"]), self.embedder).reshape(1, -1)
        scores, positions = self.index.search(vector, min(k, len(self.ids)))
        return [{"chunk_id": self.ids[int(position)],
                 "score": float(score),
                 **self._chunk_at(int(position))}
                for score, position in zip(scores[0], positions[0]) if position >= 0]


def main() -> int:
    parser = argparse.ArgumentParser(prog="src.rag_ingestion.full_index")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    build_parser.add_argument("--variant", default="page", choices=["page"])
    build_parser.add_argument("--dim", type=int, default=DIM)
    build_parser.add_argument("--batch-size", type=int, default=256)
    build_parser.add_argument("--checkpoint-batches", type=int, default=10)
    build_parser.add_argument("--embedder", choices=EMBEDDERS,
                              default=CLI_EMBEDDER)
    complete_parser = sub.add_parser("complete")
    complete_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    complete_parser.add_argument("--variant", default="page", choices=["page"])
    complete_parser.add_argument("--dim", type=int, default=DIM)
    complete_parser.add_argument("--batch-size", type=int, default=256)
    complete_parser.add_argument("--checkpoint-batches", type=int, default=10)
    complete_parser.add_argument("--embedder", choices=EMBEDDERS,
                                 default=CLI_EMBEDDER)
    search_parser = sub.add_parser("search")
    search_parser.add_argument("--cache", type=Path, default=DEFAULT_OUT)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        meta = build(args.out, args.variant, args.dim, args.batch_size,
                     args.checkpoint_batches, args.embedder)
        print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
        return 0
    if args.command == "complete":
        meta = complete(args.out, args.variant, args.dim, args.batch_size,
                        args.checkpoint_batches, args.embedder)
        print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
        return 0
    index = FullVectorIndex(args.cache)
    print(json.dumps(index.search(args.query, args.k), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
