#!/usr/bin/env python3
"""Background rerun of the Week 2 decomposition (notebooks/04_retrieval_recall.ipynb, cell 29).

Same logic and same caches as the notebook: chunks, embeddings, BM25 and eval
rows are loaded the same way the notebook loads them (no re-embed, no rerank
of cached work). Rows already cached in
results/week2_retrieval_cache/C2/decomp_cache.json (merged_ranked with
decomp_rerank_pool_size) are skipped; caches are saved after every row.

Thread count from OMP_NUM_THREADS (default 4). Progress is logged to stdout
with flush=True.
"""

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import urllib3
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# --- Thread count from OMP_NUM_THREADS (default 4) ---
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "4"

# --- root + env (cell 02) ---
_root = Path.cwd().resolve()
if not (_root / "src").is_dir():
    _root = _root.parent
os.chdir(_root)
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

from src.rag_ingestion.tokenizer import tokenize  # noqa: E402 — path bootstrap above

# --- parameters (cell 01 subset) ---
FREEZE_PATH = "data/corpus/freeze_v1.json"
PAGE_DIR = "data/corpus/extract/page_extracts"
TABLE_DIR = "data/corpus/extract/table_records"
GOLD_PATH = "data/eval/datasets/gold_v2_eval_v8_public.jsonl"
CACHE_ROOT = "results/week2_retrieval_cache"
EMBED_MODEL = "bge-small-en-v1.5"
RRF_K = 60
POOL_K = 100
REPORT_KS = (1, 3, 5, 10, 20, 50, 100)

_omp = os.environ.get("OMP_NUM_THREADS")
if _omp:
    try:
        _n = max(1, int(_omp))
        os.environ.setdefault("ORT_INTRA_OP_NUM_THREADS", str(_n))
        try:
            import onnxruntime as ort

            ort.set_default_logger_severity(3)
            try:
                ort.intra_op_num_threads = _n
            except Exception:
                pass
        except Exception:
            pass
    except Exception:
        _n = None
else:
    _n = None

print(f"[decomp] start: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)

# --- cell 02: freeze ---
FREEZE = json.loads(Path(FREEZE_PATH).read_text(encoding="utf-8"))
FREEZE_SHA = FREEZE.get("manifest_sha256")
DOCS_BY_ID = {d["document_id"]: d for d in FREEZE.get("documents", [])}
print(f"[decomp] freeze docs={len(DOCS_BY_ID)} sha={FREEZE_SHA}", flush=True)

# --- cells 03 + 07: IO / eval set helpers + build dev eval set ---
def read_jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pages(doc_id):
    return read_jsonl(Path(PAGE_DIR) / f"{doc_id}.pages.jsonl")


def load_tables(doc_id):
    return read_jsonl(Path(TABLE_DIR) / f"{doc_id}.tables.jsonl")


def usable_pages(doc_id):
    return [
        r for r in load_pages(doc_id)
        if r.get("usable_as_evidence") and isinstance(r.get("pdf_page_index"), int)
    ]


def load_gold(path=GOLD_PATH):
    return read_jsonl(path)


def dev_answer_rows_with_evidence(gold_rows):
    rows = []
    for r in gold_rows:
        if r.get("split") != "dev" or r.get("expected_behaviour") != "answer":
            continue
        ev = [
            e for e in (r.get("evidence") or [])
            if isinstance(e, dict) and e.get("document_id")
            and isinstance(e.get("pdf_page"), int)
        ]
        if ev:
            rows.append(r)
    return rows


def gold_evidence_set(row):
    return {
        (e["document_id"], int(e["pdf_page"]))
        for e in (row.get("evidence") or [])
        if isinstance(e, dict) and e.get("document_id")
        and isinstance(e.get("pdf_page"), int)
    }


def dev_slice_doc_ids(eval_rows):
    ids = set()
    for r in eval_rows:
        for e in (r.get("evidence") or []):
            if isinstance(e, dict) and e.get("document_id"):
                ids.add(e["document_id"])
    return sorted(ids)


GOLD_ROWS = load_gold(GOLD_PATH)
EVAL_ROWS = dev_answer_rows_with_evidence(GOLD_ROWS)
SLICE_DOC_IDS = dev_slice_doc_ids(EVAL_ROWS)
EVAL_GOLD_SETS = [gold_evidence_set(r) for r in EVAL_ROWS]
EVAL_QUERIES = [(r.get("question_text") or r.get("question") or "") for r in EVAL_ROWS]
print(f"[decomp] eval_n={len(EVAL_ROWS)} slice_docs={len(SLICE_DOC_IDS)}", flush=True)

# --- cells 04 + 05: chunking primitives + build_chunks_for_slice ---
_TOK_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?|[a-z0-9]+", re.IGNORECASE)
HEADING_MAXLEN = 80
STMT_KEYS = (
    "balance sheet", "profit and loss", "income statement", "cash flow",
    "statement of", "notes to", "segment", "equity", "financial statements",
)


def tok_spans(text):
    return [m.span() for m in _TOK_RE.finditer(text or "")]


def is_heading(line):
    t = (line or "").strip()
    if not t or len(t) > HEADING_MAXLEN:
        return False
    if t.isupper() and any(c.isalpha() for c in t):
        return True
    low = t.lower()
    if any(k in low for k in STMT_KEYS) and len(t.split()) <= 8:
        return True
    if t.endswith(":") and len(t.split()) <= 10:
        return True
    words = t.split()
    if 1 <= len(words) <= 8 and not any(ch.isdigit() for ch in t):
        caps = sum(1 for w in words if w[:1].isupper())
        if caps >= max(1, len(words) - 1):
            return True
    return False


def detect_sections(text):
    text = text or ""
    lines = []
    i = 0
    for part in text.split("\n"):
        s, e = i, i + len(part)
        lines.append((part, s, e))
        i = e + 1
    blocks = []
    cur_head = None
    cur_body_s = 0
    cur_body_e = 0
    have = False

    def flush(h, bs, be):
        if bs is None or be is None or be < bs:
            bs, be = 0, 0
        blocks.append((h, text[bs:be], bs, be))

    for ln, s, e in lines:
        if is_heading(ln):
            if have:
                flush(cur_head, cur_body_s, cur_body_e)
            cur_head = ln.strip()
            cur_body_s = min(e + 1, len(text))
            cur_body_e = cur_body_s
            have = True
        else:
            if not have:
                cur_head = None
                cur_body_s = s
                cur_body_e = e
                have = True
            else:
                cur_body_e = e
    if have:
        flush(cur_head, cur_body_s, cur_body_e)
    elif text:
        blocks.append((None, text, 0, len(text)))
    return blocks


def windows_by_chars(text, size=800, overlap=100):
    step = max(1, size - overlap)
    out, i, n = [], 0, len(text)
    while i < n:
        s, e = i, min(n, i + size)
        if (text[s:e] or "").strip():
            out.append((s, e))
        if e >= n:
            break
        i += step
    return out


def windows_by_spans(spans, size, overlap_frac):
    if not spans:
        return []
    step = max(1, size - int(round(size * overlap_frac)))
    out, i, n = [], 0, len(spans)
    while i < n:
        j = min(n, i + size)
        s, e = spans[i][0], spans[j - 1][1]
        if e > s:
            out.append((s, e))
        if j >= n:
            break
        i += step
    return out


def render_table(rec):
    parts = []
    cap = (rec.get("caption") or "").strip()
    if cap:
        parts.append("caption: " + cap)
    parts.append("statement_kind: " + str(rec.get("statement_kind") or ""))
    uh = (rec.get("unit_header") or "").strip()
    if uh:
        parts.append("unit_header: " + uh)
    parts.append("reporting_basis: " + str(rec.get("reporting_basis") or ""))
    cols = rec.get("columns") or []
    if cols:
        col_lines = []
        for c in cols:
            col_lines.append(f"{c.get('column_id')}:{c.get('header_raw')}")
            if c.get("header_path"):
                col_lines[-1] += f" <{c.get('header_path')}>"
        parts.append("columns: " + " | ".join(col_lines))
    ordmap = {
        c.get("column_id"): (
            c.get("column_ordinal") if isinstance(c.get("column_ordinal"), int) else i
        )
        for i, c in enumerate(cols)
    }
    for row in rec.get("rows") or []:
        label = (row.get("row_label_raw") or "").strip()
        pairs = []
        for cell in sorted(
            row.get("cells") or [], key=lambda x: ordmap.get(x.get("column_id"), 0)
        ):
            raw = str(cell.get("raw") or "").strip()
            if raw:
                pairs.append(f"{cell.get('column_id')}={raw}")
        if label and pairs:
            parts.append(label + ": " + "; ".join(pairs))
        elif pairs:
            parts.append("; ".join(pairs))
    return "\n".join(p for p in parts if p.strip())


def make_chunk(chunk_id, text, document_id, page_start, page_end, chunker, meta_extra=None):
    m = {
        "document_id": document_id,
        "page_start": page_start,
        "page_end": page_end,
        "chunker": chunker,
    }
    if meta_extra:
        m.update(meta_extra)
    return {"chunk_id": chunk_id, "text": text, "meta": m}


def chunk_c2(doc_id, pages):
    out = []
    for rec in pages:
        text = rec.get("page_text") or ""
        page = rec.get("pdf_page_index")
        for j, (s, e) in enumerate(windows_by_chars(text, 800, 100)):
            cid = f"{doc_id}::C2::p{page}::w{j}"
            out.append(make_chunk(cid, text[s:e], doc_id, page, page, "C2", {"source": "page"}))
    return out


def chunk_section_tok512_ov10(doc_id, pages):
    out = []
    for rec in pages:
        text = rec.get("page_text") or ""
        page = rec.get("pdf_page_index")
        secs = detect_sections(text)
        k = 0
        for si, (head, body, bs, be) in enumerate(secs):
            if not (body or "").strip():
                continue
            b_spans = tok_spans(body)
            if len(b_spans) < 512:
                s, e = bs, be
                cid = f"{doc_id}::section_tok512_ov10::p{page}::s{si}::w{k}"
                out.append(make_chunk(
                    cid, text[s:e], doc_id, page, page,
                    "section_tok512_ov10", {"source": "section", "heading": head},
                ))
                k += 1
            else:
                for ls, le in windows_by_spans(b_spans, 512, 0.10):
                    s, e = bs + ls, bs + le
                    cid = f"{doc_id}::section_tok512_ov10::p{page}::s{si}::w{k}"
                    out.append(make_chunk(
                        cid, text[s:e], doc_id, page, page,
                        "section_tok512_ov10", {"source": "section", "heading": head},
                    ))
                    k += 1
    return out


def chunk_table_tok512_fixed(doc_id, pages, tables):
    """Fixed (in-notebook) table-aware chunker (see notebook 04)."""
    out = []
    covered = set()
    table_recs = []
    for rec in tables:
        ps, pe = rec.get("page_start"), rec.get("page_end")
        if isinstance(ps, int) and isinstance(pe, int):
            for p in range(ps, pe + 1):
                covered.add(p)
        ttext = render_table(rec)
        if not ttext.strip():
            continue
        table_recs.append((rec, ttext))
    for i, (rec, ttext) in enumerate(table_recs):
        ps = rec.get("page_start") if isinstance(rec.get("page_start"), int) else -1
        pe = rec.get("page_end") if isinstance(rec.get("page_end"), int) else ps
        out.append(make_chunk(
            f"{doc_id}::table_tok512_fixed::table::{i}", ttext, doc_id, ps, pe,
            "table_tok512_fixed",
            {
                "source": "table",
                "table_logical_key": rec.get("table_logical_key"),
                "table_extraction_id": rec.get("table_extraction_id"),
                "statement_kind": rec.get("statement_kind"),
            },
        ))
    for rec in pages:
        page = rec.get("pdf_page_index")
        if page in covered:
            continue
        text = rec.get("page_text") or ""
        spans = tok_spans(text)
        for j, (s, e) in enumerate(windows_by_spans(spans, 512, 0.10)):
            out.append(make_chunk(
                f"{doc_id}::table_tok512_fixed::p{page}::w{j}", text[s:e],
                doc_id, page, page, "table_tok512_fixed", {"source": "page_prose"},
            ))
    return out


def _fn_c2(doc_id, pages, tables):
    return chunk_c2(doc_id, pages)


def _fn_section_tok512_ov10(doc_id, pages, tables):
    return chunk_section_tok512_ov10(doc_id, pages)


CHUNK_FUNCS = {
    "C2": _fn_c2,
    "section_tok512_ov10": _fn_section_tok512_ov10,
    "table_tok512_fixed": chunk_table_tok512_fixed,
}


def build_chunks_for_slice(chunker_name, doc_ids):
    fn = CHUNK_FUNCS[chunker_name]
    chunks = []
    for doc_id in doc_ids:
        if doc_id not in DOCS_BY_ID:
            continue
        pages = usable_pages(doc_id)
        tables = load_tables(doc_id)
        chunks.extend(fn(doc_id, pages, tables))
    return chunks


# --- cell 08: retrieval + metrics (function-based) ---
def chunk_matches_gold(chunk_meta, gold_set):
    did = chunk_meta.get("document_id")
    ps = chunk_meta.get("page_start", chunk_meta.get("page"))
    pe = chunk_meta.get("page_end", ps)
    if not isinstance(ps, int):
        return False
    if not isinstance(pe, int):
        pe = ps
    for gd, gp in gold_set:
        if gd == did and ps <= gp <= pe:
            return True
    return False


def bm25_build(chunks):
    corpus_tokens = [tokenize(c.get("text") or "") for c in chunks]
    return BM25Okapi(corpus_tokens)


def bm25_search(bm25, chunks, query, k=10):
    scores = bm25.get_scores(tokenize(query or ""))
    if len(scores) == 0:
        return []
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def rrf_fuse(rank_lists, k=60, topn=10):
    agg = defaultdict(float)
    for rl in rank_lists:
        for rank, (idx, _s) in enumerate(rl, start=1):
            agg[idx] += 1.0 / (k + rank)
    ranked = sorted(agg.items(), key=lambda x: (-x[1], x[0]))
    return [(int(i), float(s)) for i, s in ranked[:topn]]


# --- cell 09: dense embeddings via VM02 (function-based) ---
def _safe_text(t, max_chars=3000):
    t = "" if t is None else str(t)
    t = t.replace("\x00", " ")
    t = t if len(t) <= max_chars else t[:max_chars]
    return t if t.strip() else " "


def _post_embeddings(batch_texts, base_url, api_key, model, ca_cert, max_retries=5):
    url = base_url.rstrip("/") + "/embeddings"
    verify = ca_cert if ca_cert and Path(ca_cert).exists() else False
    if verify is False:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": batch_texts}
    last_resp = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120, verify=verify)
            last_resp = r
            if r.status_code in (429, 503) or r.status_code >= 500:
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code == 422:
                return None, r
            r.raise_for_status()
            data = r.json()
            sorted_data = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            embs = [d["embedding"] for d in sorted_data]
            if len(embs) != len(batch_texts):
                raise RuntimeError(f"embedding count mismatch {len(embs)} vs {len(batch_texts)}")
            return embs, r
        except requests.HTTPError:
            raise
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** attempt, 20))
    return None, last_resp


def _embed_adaptive(batch_texts, base_url, api_key, model, ca_cert, max_retries=5, depth=0):
    embs, resp = _post_embeddings(batch_texts, base_url, api_key, model, ca_cert, max_retries)
    if embs is not None:
        return embs
    if len(batch_texts) > 1:
        mid = len(batch_texts) // 2
        left = _embed_adaptive(
            batch_texts[:mid], base_url, api_key, model, ca_cert, max_retries, depth + 1
        )
        right = _embed_adaptive(
            batch_texts[mid:], base_url, api_key, model, ca_cert, max_retries, depth + 1
        )
        return left + right
    text = batch_texts[0] or " "
    for max_chars in (1500, 800, 400, 200, 80, 20, 1):
        trial = text[:max_chars]
        if not trial.strip():
            trial = " "
        embs, resp = _post_embeddings(
            [trial], base_url, api_key, model, ca_cert, max_retries=max_retries
        )
        if embs is not None:
            return embs
    embs, resp = _post_embeddings(
        ["0"], base_url, api_key, model, ca_cert, max_retries=max_retries
    )
    if embs is not None:
        return embs
    if resp is not None:
        resp.raise_for_status()
    raise RuntimeError("embedding 422 unrecoverable for single item")


def embed_texts_vm02(
    texts, base_url, api_key, model, ca_cert,
    batch=16, max_retries=5, label="embed", resume_state=None, checkpoint_fn=None,
):
    texts = [_safe_text(t, 3000) for t in texts]
    resumed = [] if resume_state is None else resume_state.tolist()
    out = [np.asarray(x, dtype=np.float32) for x in resumed]
    start = len(out)
    total = len(texts)
    if start:
        print(f"{label}: resuming from {start}/{total}", flush=True)
    b = 0
    for i in range(start, total, batch):
        b += 1
        chunk = texts[i:i + batch]
        embs = _embed_adaptive(chunk, base_url, api_key, model, ca_cert, max_retries=max_retries)
        out.extend([np.asarray(e, dtype=np.float32) for e in embs])
        done = len(out)
        if b == 1 or b % 10 == 0 or done >= total:
            print(f"{label}: {done}/{total}", flush=True)
        if checkpoint_fn and (b % 5 == 0 or done >= total):
            checkpoint_fn(out)
    arr = np.asarray(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# --- cell 26: base (chunks, embeddings, BM25) ---
chunks_c2 = build_chunks_for_slice("C2", SLICE_DOC_IDS)
cache_dir = Path(CACHE_ROOT) / "C2"
embs_c2 = np.load(cache_dir / "embeddings.npy")
cache_ids = json.loads((cache_dir / "chunk_ids.json").read_text(encoding="utf-8"))
qembs = np.load(Path(CACHE_ROOT) / "eval_query_embeddings.npy")
chunk_ids_now = [c["chunk_id"] for c in chunks_c2]
assert cache_ids == chunk_ids_now, "C2 chunk_ids do not match cached embeddings"
assert qembs.shape[0] == len(EVAL_ROWS) == len(EVAL_QUERIES)
doc_ids_arr = np.array([c["meta"].get("document_id") for c in chunks_c2])
print(f"[decomp] C2 chunks={len(chunks_c2)} emb={embs_c2.shape} qemb={qembs.shape}", flush=True)

# --- cell 26: cleaned BM25 (all BM25 here) ---
STOPWORDS_SIMPLE = set("""
    a an and are as at be by for from has have he in is it its of on or that the to
    was were will with this these those but if into no not than then so such can may
    we our their they you your i me my
""".split())


def simple_tok(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = text.split()
    return [t for t in toks if t not in STOPWORDS_SIMPLE and len(t) > 1]


corpus_simple = [simple_tok(c.get("text") or "") for c in chunks_c2]
bm25_simple = BM25Okapi(corpus_simple)


def topk_from_scores(scores, k):
    k = min(k, len(scores))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


bm25s_scores_all = [bm25_simple.get_scores(simple_tok(q or "")) for q in EVAL_QUERIES]
dense_scores_all = [embs_c2 @ qembs[i] for i in range(len(EVAL_QUERIES))]


def score_page_level(rankings, chunks, ks):
    n = len(EVAL_ROWS)
    recalls = {k: 0 for k in ks}
    mrr_sum = 0.0
    for qi, gold in enumerate(EVAL_GOLD_SETS):
        ranked = rankings[qi]
        rr = 0.0
        for rank, (idx, _s) in enumerate(ranked, start=1):
            if chunk_matches_gold(chunks[idx]["meta"], gold):
                rr = 1.0 / rank
                break
        mrr_sum += rr
        for k in ks:
            if any(chunk_matches_gold(chunks[idx]["meta"], gold) for idx, _s in ranked[:k]):
                recalls[k] += 1
    hits = {f"hits@{k}": recalls[k] for k in ks}
    return {"n": n, **{f"recall@{k}": recalls[k] / n for k in ks}, **hits, "mrr": mrr_sum / n}


single_idx = [i for i, g in enumerate(EVAL_GOLD_SETS) if len(g) == 1]
multi_idx = [i for i, g in enumerate(EVAL_GOLD_SETS) if len(g) > 1]
print(
    f"[decomp] single={len(single_idx)} multi={len(multi_idx)} multi_idx={multi_idx}",
    flush=True,
)


def score_subset(rankings, chunks, idxs, ks):
    n = len(idxs)
    recalls = {k: 0 for k in ks}
    mrr_sum = 0.0
    for qi in idxs:
        gold = EVAL_GOLD_SETS[qi]
        ranked = rankings[qi]
        rr = 0.0
        for rank, (idx, _s) in enumerate(ranked, start=1):
            if chunk_matches_gold(chunks[idx]["meta"], gold):
                rr = 1.0 / rank
                break
        mrr_sum += rr
        for k in ks:
            if any(chunk_matches_gold(chunks[idx]["meta"], gold) for idx, _s in ranked[:k]):
                recalls[k] += 1
    hits = {f"hits@{k}": recalls[k] for k in ks}
    return {
        "n": n,
        **{f"recall@{k}": (recalls[k] / n if n else 0.0) for k in ks},
        **hits,
        "mrr": (mrr_sum / n if n else 0.0),
    }


def _fmt(m, ks):
    d = {f"R@{k}": round(m[f"recall@{k}"], 4) for k in ks}
    d["MRR"] = round(m["mrr"], 4)
    d["hits"] = {k: m[f"hits@{k}"] for k in ks}
    d["n"] = m["n"]
    return d


# --- cells 26 + 27: T4 inputs (hybrid_clean_rank100 from cached scores) ---
bm25_clean_rank100 = [
    topk_from_scores(bm25s_scores_all[i], POOL_K) for i in range(len(EVAL_QUERIES))
]
dense_rank100 = [topk_from_scores(dense_scores_all[i], POOL_K) for i in range(len(EVAL_QUERIES))]
hybrid_clean_rank100 = [
    rrf_fuse([bm25_clean_rank100[i], dense_rank100[i]], k=RRF_K, topn=POOL_K)
    for i in range(len(EVAL_QUERIES))
]

# --- cell 29 ---
DECOMP_CACHE = Path(CACHE_ROOT) / "C2" / "decomp_cache.json"
SUBQ_EMB_CACHE = Path(CACHE_ROOT) / "C2" / "subquery_embeddings.json"
SUBQ_RERANK_CACHE = Path(CACHE_ROOT) / "C2" / "subquery_rerank_cache.json"
RERANK_TOPK_SUBQ = 20  # per sub-query rerank only top-20 of hybrid pool

rerank_model_name = None
latencies = []


def _load_json_cache(p, default):
    p = Path(p)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _save_json_cache(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


decomp_cache = _load_json_cache(DECOMP_CACHE, {})
subq_emb_cache = _load_json_cache(SUBQ_EMB_CACHE, {})
subq_rerank_cache = _load_json_cache(SUBQ_RERANK_CACHE, {})

# T2 cache load for T4 (reranked_rank100 + latencies) — read-only
_rc = Path(CACHE_ROOT) / "C2" / "rerank_scores.npy"
_rm = Path(CACHE_ROOT) / "C2" / "rerank_scores.meta.json"
_rl = Path(CACHE_ROOT) / "C2" / "rerank_latency.json"
reranked_rank100 = None
if _rc.exists() and _rm.exists() and hybrid_clean_rank100 is not None:
    _arr = np.load(_rc)
    if _arr.shape[0] == len(EVAL_QUERIES):
        rerank_model_name = (
            globals().get("rerank_model_name")
            or json.loads(_rm.read_text(encoding="utf-8")).get("model")
        )
        if not latencies and _rl.exists():
            latencies = json.loads(_rl.read_text(encoding="utf-8")).get("per_query_seconds", [])
        reranked_rank100 = []
        for _qi in range(len(EVAL_QUERIES)):
            _pool = hybrid_clean_rank100[_qi]
            _order = np.argsort(-_arr[_qi])
            reranked_rank100.append(
                [(_pool[_j][0], float(_arr[_qi, _j])) for _j in _order]
            )
        print(
            f"[decomp] T2 rerank cache loaded: model={rerank_model_name} shape={_arr.shape}",
            flush=True,
        )


def _row_key(question, n_parts):
    return hashlib.sha256((question + "||" + str(n_parts)).encode("utf-8")).hexdigest()


def llm_decompose(question, n_parts, cache, key):
    if key in cache and isinstance(cache[key], dict) and cache[key].get("sub_queries"):
        return cache[key]["sub_queries"], "cache"
    try:
        from src.llm import client as llm_client
    except Exception as e:
        return None, f"import_error:{e!r}"
    prompt = (
        "Split the following financial question into exactly " + str(n_parts)
        + " focused sub-questions, "
        "one per company/document being compared or referenced. Do not mention document IDs. "
        "Return ONLY a JSON array of strings.\nQuestion: " + question
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = llm_client.chat(messages, max_tokens=300, temperature=0.0, cache=True, timeout=60.0)
        txt = resp.text
        m = re.search(r"\[.*\]", txt, re.DOTALL)
        arr = json.loads(m.group(0) if m else txt)
        subs = [str(x).strip() for x in arr if str(x).strip()]
        subs = subs[:n_parts]
        if len(subs) == n_parts:
            cache[key] = {**(cache.get(key) or {}), "question": question,
                          "n_parts": n_parts, "sub_queries": subs}
            _save_json_cache(DECOMP_CACHE, cache)
            return subs, "llm"
        return None, f"bad_count:{len(subs)}"
    except Exception as e:
        return None, f"error:{e!r}"


def embed_subq(subq, cache):
    skey = hashlib.sha256(("dense:" + subq).encode("utf-8")).hexdigest()
    if skey in cache:
        return np.asarray(cache[skey], dtype=np.float32), "cache"
    try:
        base_url = os.environ.get("LLM_EMBEDDINGS_URL") or os.environ.get("EMBEDDINGS_URL")
        api_key = (
            os.environ.get("EMBEDDINGS_API_KEY")
            or os.environ.get("LLM_EMBEDDINGS_API_KEY") or ""
        )
        ca_cert = os.environ.get("EMBEDDINGS_CA_CERT") or ""
        if "embed_texts_vm02" in globals():
            arr = embed_texts_vm02(
                [subq or ""], base_url, api_key, EMBED_MODEL, ca_cert,
                batch=1, max_retries=5, label="subq",
            )[0]
        else:
            raise RuntimeError("embed_texts_vm02 not available")
        cache[skey] = [float(x) for x in arr]
        _save_json_cache(SUBQ_EMB_CACHE, cache)
        return np.asarray(arr, dtype=np.float32), "embed"
    except Exception as e:
        return None, f"error:{e!r}"


_subq_rerank_model = None


def _get_subq_rerank_model():
    global _subq_rerank_model
    if _subq_rerank_model is not None:
        return _subq_rerank_model
    model_name = globals().get("rerank_model_name")
    if model_name is None:
        # try load from T2 cache meta
        meta_p = Path(CACHE_ROOT) / "C2" / "rerank_scores.meta.json"
        if meta_p.exists():
            try:
                model_name = json.loads(meta_p.read_text(encoding="utf-8")).get("model")
            except Exception:
                model_name = None
    if model_name is None:
        return None
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        if _n is not None:
            try:
                _subq_rerank_model = TextCrossEncoder(model_name=model_name, threads=_n)
            except TypeError:
                _subq_rerank_model = TextCrossEncoder(model_name=model_name)
        else:
            _subq_rerank_model = TextCrossEncoder(model_name=model_name)
        globals()["_subq_rerank_model"] = _subq_rerank_model
        return _subq_rerank_model
    except Exception as e:
        print("subq rerank model load failed:", e, flush=True)
        return None


def _subq_rerank_scores(subq, pool_rerank):
    # pool_rerank: list[(idx, score)] length<=RERANK_TOPK_SUBQ; cache by (subq, pool_size, pool ids)
    pool_idx = [int(i) for i, _s in pool_rerank]
    rkey_src = subq + "||ps=" + str(RERANK_TOPK_SUBQ) + "||pool=" + ",".join(map(str, pool_idx))
    rkey = hashlib.sha256(rkey_src.encode("utf-8")).hexdigest()
    if rkey in subq_rerank_cache:
        ent = subq_rerank_cache[rkey]
        if ent.get("pool_idx") == pool_idx and len(ent.get("scores", [])) == len(pool_idx):
            return [float(x) for x in ent["scores"]], "cache"
    model = _get_subq_rerank_model()
    if model is None:
        return None, "no_model"
    try:
        texts = [(chunks_c2[idx].get("text") or "") for idx in pool_idx]
        sc = [float(x) for x in list(model.rerank(subq, texts))]
        subq_rerank_cache[rkey] = {
            "subq": subq, "pool_size": RERANK_TOPK_SUBQ,
            "pool_idx": pool_idx, "scores": sc,
        }
        _save_json_cache(SUBQ_RERANK_CACHE, subq_rerank_cache)
        return sc, "rerank"
    except Exception as e:
        return None, f"error:{e!r}"


def decomp_retrieve_merge(subqs):
    merged = {}
    per_sub = []
    for sq in subqs:
        bs = bm25_simple.get_scores(simple_tok(sq or ""))
        b_rank = topk_from_scores(bs, POOL_K)
        qv, qv_src = embed_subq(sq, subq_emb_cache)
        if qv is not None:
            ds = embs_c2 @ qv
            d_rank = topk_from_scores(ds, POOL_K)
            pool_full = rrf_fuse([b_rank, d_rank], k=RRF_K, topn=POOL_K)
        else:
            pool_full = b_rank
        pool_rerank = pool_full[:RERANK_TOPK_SUBQ]
        sc, sc_src = _subq_rerank_scores(sq, pool_rerank)
        if sc is not None:
            for j, (idx, _s) in enumerate(pool_rerank):
                score = float(sc[j])
                if (idx not in merged) or (score > merged[idx]):
                    merged[idx] = score
        else:
            # fallback: RRF scores for top-20 only
            for idx, s in pool_rerank:
                score = float(s)
                if (idx not in merged) or (score > merged[idx]):
                    merged[idx] = score
        per_sub.append({
            "subq": sq, "dense": qv_src, "rerank": sc_src,
            "pool_n": len(pool_full), "rerank_n": len(pool_rerank),
        })
    merged_ranked = sorted(merged.items(), key=lambda x: -x[1])
    return [(int(i), float(s)) for i, s in merged_ranked], per_sub


print(f"[decomp] start multi_evidence loop: rows={len(multi_idx)}", flush=True)
multi_results = {}
decomp_errors = []

if not multi_idx:
    print("no multi-evidence rows", flush=True)
else:
    for qi in multi_idx:
        row = EVAL_ROWS[qi]
        q = row.get("question_text") or row.get("question") or ""
        n_parts = len(EVAL_GOLD_SETS[qi])
        key = _row_key(q, n_parts)
        ent = decomp_cache.get(key) or {}
        # skip row if fully done in cache with current params
        if ent.get("merged_ranked") and ent.get("decomp_rerank_pool_size") == RERANK_TOPK_SUBQ:
            merged_ranked = [(int(i), float(s)) for i, s in ent["merged_ranked"]]
            multi_results[qi] = {
                "sub_queries": ent.get("sub_queries", []),
                "merged_ranked": merged_ranked,
                "status": "cache_row",
            }
            print(f"row {qi}: skipped (cached merged, ps={RERANK_TOPK_SUBQ})", flush=True)
            continue
        subs = ent.get("sub_queries") if ent.get("sub_queries") else None
        if subs is not None:
            status = "cache_subq"
        else:
            print(f"row {qi}: decomposing (n_parts={n_parts})", flush=True)
            subs, status = llm_decompose(q, n_parts, decomp_cache, key)
        if subs is None:
            decomp_errors.append((qi, status))
            print(f"  row {qi} decomp fail: {status}", flush=True)
            continue
        print(
            f"row {qi}: merge+rerank top-{RERANK_TOPK_SUBQ} per subq "
            f"(n_subq={len(subs)}, src={status})", flush=True,
        )
        merged_ranked, per_sub = decomp_retrieve_merge(subs)
        # save after each row
        ent = decomp_cache.get(key) or {}
        ent.update({
            "question": q,
            "n_parts": n_parts,
            "sub_queries": subs,
            "merged_ranked": [[int(i), float(s)] for i, s in merged_ranked],
            "decomp_rerank_pool_size": RERANK_TOPK_SUBQ,
            "per_sub": per_sub,
        })
        decomp_cache[key] = ent
        _save_json_cache(DECOMP_CACHE, decomp_cache)
        multi_results[qi] = {
            "sub_queries": subs, "merged_ranked": merged_ranked, "status": status,
        }
        print(f"  row {qi}: saved (merged_n={len(merged_ranked)})", flush=True)

    if decomp_errors and not multi_results:
        print(
            "decomposition unavailable/failed for all multi-evidence rows; "
            "coverage@10 not computed.",
            flush=True,
        )
        for qi, st in decomp_errors:
            print(f"  row idx {qi}: {st}", flush=True)
    else:
        if decomp_errors:
            print("decomposition errors (skipped):", decomp_errors, flush=True)
        done_idxs = [qi for qi in multi_idx if qi in multi_results]
        per_row_cov = []
        for qi in done_idxs:
            gold = EVAL_GOLD_SETS[qi]
            ranked = multi_results[qi]["merged_ranked"][:10]
            if not gold:
                per_row_cov.append(0.0)
                continue
            hit_pairs = sum(
                1 for g in gold
                if any(chunk_matches_gold(chunks_c2[idx]["meta"], {g}) for idx, _s in ranked)
            )
            per_row_cov.append(hit_pairs / len(gold))
        avg_cov = sum(per_row_cov) / len(per_row_cov) if per_row_cov else 0.0
        print(f"multi-evidence rows total={len(multi_idx)} completed={len(done_idxs)}", flush=True)
        print(
            "evidence-level coverage@10 (avg pair fraction hit, completed rows): "
            f"{round(avg_cov, 4)}",
            flush=True,
        )
        for qi in done_idxs:
            cov = per_row_cov[done_idxs.index(qi)]
            print(
                f"  row {qi} n_gold={len(EVAL_GOLD_SETS[qi])} cov@10={round(cov, 4)} "
                f"subq_src={multi_results[qi]['status']}",
                flush=True,
            )

# --- cell 29 T4: Tables 1-4 (read-only metrics from cached scores) ---
print("\n=== 4. Split every table by single vs multi-evidence, with counts ===", flush=True)
counts_line = (
    f"counts: total={len(EVAL_ROWS)} single_evidence_rows={len(single_idx)} "
    f"multi_evidence_rows={len(multi_idx)}"
)
print(counts_line, flush=True)

print("\n[Table 1] Hybrid RRF (cleaned BM25 + dense), pool top-100 (page-level)", flush=True)
all_idx = list(range(len(EVAL_ROWS)))
split_sets = [("single_evidence", single_idx), ("multi_evidence", multi_idx), ("all", all_idx)]
for split_name, idxs in split_sets:
    ms = score_subset(hybrid_clean_rank100, chunks_c2, idxs, ks=REPORT_KS)
    print(f"  {split_name} n={ms['n']}", _fmt(ms, REPORT_KS), flush=True)

print("\n[Table 2] Cross-encoder rerank of top-100 pool (page-level)", flush=True)
if reranked_rank100 is not None:
    print("  model:", rerank_model_name, flush=True)
    for split_name, idxs in split_sets:
        ms = score_subset(reranked_rank100, chunks_c2, idxs, ks=(1, 3, 5, 10))
        print(f"  {split_name} n={ms['n']}", _fmt(ms, (1, 3, 5, 10)), flush=True)
    if latencies:
        mean_lat = round(sum(latencies) / len(latencies), 4)
        min_lat = round(min(latencies), 4)
        max_lat = round(max(latencies), 4)
        print(f"  latency/query sec mean/min/max: {mean_lat} {min_lat} {max_lat}", flush=True)
else:
    print("  SKIPPED/unavailable", flush=True)

print("\n[Table 3] Decomposition multi-evidence evidence-level coverage@10", flush=True)
if not multi_idx:
    print("  no multi-evidence rows", flush=True)
elif multi_results:
    done_idxs = [qi for qi in multi_idx if qi in multi_results]
    per_row_cov = []
    for qi in done_idxs:
        gold = EVAL_GOLD_SETS[qi]
        ranked = multi_results[qi]["merged_ranked"][:10]
        hit_pairs = sum(
            1 for g in gold
            if any(chunk_matches_gold(chunks_c2[idx]["meta"], {g}) for idx, _s in ranked)
        ) if gold else 0
        per_row_cov.append((hit_pairs / len(gold)) if gold else 0.0)
    avg = round(sum(per_row_cov) / len(per_row_cov), 4) if per_row_cov else "NA"
    print(
        f"  multi_evidence n={len(multi_idx)} completed={len(done_idxs)} "
        f"avg_pair_fraction_hit@10={avg}",
        flush=True,
    )
    print(f"  single_evidence n={len(single_idx)} (decomposition N/A by design)", flush=True)
else:
    detail = "errors: " + str(decomp_errors) if decomp_errors else "unavailable"
    print(f"  multi_evidence n={len(multi_idx)} completed=0 coverage@10=NA ({detail})", flush=True)
    print(f"  single_evidence n={len(single_idx)} (decomposition N/A)", flush=True)

print(f"\n[decomp] end: completed={len(multi_results)}/{len(multi_idx)}", flush=True)
