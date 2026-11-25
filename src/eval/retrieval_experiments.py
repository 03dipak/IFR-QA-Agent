"""Day 5 controlled retrieval experiments (LLD-29). Diagnosis-only, no production changes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.eval import agent_eval
from src.eval.retrieval_diagnosis import rrf_scores

CHUNK_RE = __import__('re').compile(r"^(?P<doc>.+)::C2::p(?P<page>\d+)::w(?P<win>\d+)$")


def write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def doc_chunk_indices(ids: list[str], doc12: str) -> list[int]:
    out = []
    for i, cid in enumerate(ids):
        m = CHUNK_RE.match(cid)
        if m and m.group('doc')[:12] == doc12:
            out.append(i)
    return out


def mrr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


def within_doc_ranking(
    retriever, query: str, expected_doc: str, expected_page: int,
) -> dict[str, Any]:
    from src.tools.retrieve import cleaned_tokens, embed_query
    ids = retriever.ids
    tokens = cleaned_tokens(query)
    bm25 = np.asarray(retriever._bm25().get_scores(tokens), dtype=np.float32)
    q = embed_query(query, config=retriever.config).astype(np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    dense = (retriever.embeddings @ q).astype(np.float32)
    idxs = doc_chunk_indices(ids, expected_doc)
    if not idxs:
        return {'error': 'doc_not_in_index'}
    rows = []
    for i in idxs:
        m = CHUNK_RE.match(ids[i])
        rows.append({
            'idx': i, 'page': int(m.group('page')),
            'bm25': float(bm25[i]), 'dense': float(dense[i]),
        })
    # page-level aggregate: best chunk score per page
    by_page: dict[int, dict[str, float]] = {}
    for r in rows:
        p = r['page']
        cur = by_page.setdefault(p, {'bm25': -1e9, 'dense': -1e9})
        cur['bm25'] = max(cur['bm25'], r['bm25'])
        cur['dense'] = max(cur['dense'], r['dense'])
    pages = list(by_page)
    bm25_order = sorted(pages, key=lambda p: -by_page[p]['bm25'])
    dense_order = sorted(pages, key=lambda p: -by_page[p]['dense'])
    def norm_scores(vals):
        arr = np.array(vals, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo + 1e-9)
    b = norm_scores([by_page[p]['bm25'] for p in pages])
    d = norm_scores([by_page[p]['dense'] for p in pages])
    hybrid = {p: float((b[i]+d[i])/2) for i, p in enumerate(pages)}
    hybrid_order = sorted(pages, key=lambda p: -hybrid[p])
    def rank(order):
        return order.index(expected_page) + 1 if expected_page in order else None
    return {
        'n_pages_in_doc': len(pages),
        'bm25_rank': rank(bm25_order), 'dense_rank': rank(dense_order),
        'hybrid_rank': rank(hybrid_order),
        'bm25_top5': bm25_order[:5], 'dense_top5': dense_order[:5],
        'hybrid_top5': hybrid_order[:5],
    }


def simulate_reranker(retriever, query: str, rrf: np.ndarray, pool_n: int, topk: int):
    pool = np.argsort(-rrf, kind='mergesort')[:pool_n]
    reranked = retriever._rerank(query, [(int(i), float(rrf[i])) for i in pool])
    return reranked[:topk]


def run_day5(run_dir: Path, day3_dir: Path, gold_path: Path, out_dir: Path) -> dict[str, Any]:
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    from src import config  # noqa: F401
    from src.tools.retrieve import Retriever, cleaned_tokens, embed_query
    retriever = Retriever()
    ids = retriever.ids
    fail = list(csv.DictReader(
        (Path('results/day4_retrieval_diagnosis/failure_analysis.csv')).open()))

    exp_a = []
    exp_c = []
    seen_queries = {}
    for row in fail:
        q = row['query']
        if q not in seen_queries:
            tokens = cleaned_tokens(q)
            bm25 = np.asarray(retriever._bm25().get_scores(tokens), dtype=np.float32)
            qv = embed_query(q, config=retriever.config).astype(np.float32)
            qv = qv / (np.linalg.norm(qv) or 1.0)
            dense = (retriever.embeddings @ qv).astype(np.float32)
            rrf = rrf_scores(retriever, bm25, dense)
            seen_queries[q] = {'bm25': bm25, 'dense': dense, 'rrf': rrf}
        st = seen_queries[q]
        edoc, epage = row['expected_doc'], int(row['expected_page'])
        # Exp A: within-doc page ranking (diagnostic upper bound given correct doc)
        w = within_doc_ranking_cached(retriever, q, edoc, epage, st)
        w.update({'row_id': row['row_id'], 'call_index': row['call_index'],
                  'expected_doc': edoc, 'expected_page': epage,
                  'baseline_root_cause': row['root_cause'],
                  'baseline_final_rank': row['final_best_rank'] or None})
        exp_a.append(w)
        # Exp C: RRF/reranker simulations
        variants = [(30,5,'baseline'), (60,5,'pool60_top5'), (30,10,'pool30_top10'),
                    (60,10,'pool60_top10'), (30,5,'no_rerank_rrf')]
        for pool_n, topk, label in variants:
            if label == 'no_rerank_rrf':
                order = np.argsort(-st['rrf'], kind='mergesort')[:topk]
                final = [(int(i), float(st['rrf'][i])) for i in order]
            else:
                final = simulate_reranker(retriever, q, st['rrf'], pool_n, topk)
            hit = None
            for rank, (idx, score) in enumerate(final, start=1):
                m = CHUNK_RE.match(ids[idx])
                if m and m.group('doc')[:12]==edoc and int(m.group('page'))==epage:
                    hit = rank
                    break
            exp_c.append({'row_id': row['row_id'], 'call_index': row['call_index'],
                          'expected_doc': edoc, 'expected_page': epage,
                          'variant': label, 'pool_n': pool_n, 'topk': topk,
                          'final_rank': hit, 'baseline_root_cause': row['root_cause']})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a_cols = ['row_id','call_index','expected_doc','expected_page','baseline_root_cause',
              'baseline_final_rank','n_pages_in_doc','bm25_rank','dense_rank','hybrid_rank',
              'bm25_top5','dense_top5','hybrid_top5']
    c_cols = ['row_id','call_index','expected_doc','expected_page','baseline_root_cause',
              'variant','pool_n','topk','final_rank']
    write_csv(out_dir/'exp_a_within_doc.csv', exp_a, a_cols)
    write_csv(out_dir/'exp_c_rrf_reranker.csv', exp_c, c_cols)
    summary = summarize(exp_a, exp_c)
    (out_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def within_doc_ranking_cached(retriever, query, expected_doc, expected_page, st):
    ids=retriever.ids
    idxs=doc_chunk_indices(ids, expected_doc)
    if not idxs:
        return {'error': 'doc_not_in_index'}
    by_page = {}
    for i in idxs:
        m=CHUNK_RE.match(ids[i])
        p=int(m.group('page'))
        cur=by_page.setdefault(p,{'bm25':-1e9,'dense':-1e9})
        cur['bm25']=max(cur['bm25'], float(st['bm25'][i]))
        cur['dense']=max(cur['dense'], float(st['dense'][i]))
    pages = list(by_page)
    def norm_scores(vals):
        arr = np.array(vals, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo + 1e-9)
    b = norm_scores([by_page[p]['bm25'] for p in pages])
    d = norm_scores([by_page[p]['dense'] for p in pages])
    hybrid = {p: float((b[i] + d[i]) / 2) for i, p in enumerate(pages)}
    def rank(order):
        return order.index(expected_page) + 1 if expected_page in order else None
    bm25_order = sorted(pages, key=lambda p: -by_page[p]['bm25'])
    dense_order = sorted(pages, key=lambda p: -by_page[p]['dense'])
    hybrid_order = sorted(pages, key=lambda p: -hybrid[p])
    return {
        'n_pages_in_doc': len(pages),
        'bm25_rank': rank(bm25_order),
        'dense_rank': rank(dense_order),
        'hybrid_rank': rank(hybrid_order),
        'bm25_top5': bm25_order[:5],
        'dense_top5': dense_order[:5],
        'hybrid_top5': hybrid_order[:5],
    }


def summarize(exp_a, exp_c):
    def agg(rows, key):
        ranks = [r[key] for r in rows if isinstance(r.get(key), int)]
        n = len(rows)
        return {'n': n, 'found': len(ranks), 'top1': sum(r == 1 for r in ranks),
                'top3': sum(r <= 3 for r in ranks), 'top5': sum(r <= 5 for r in ranks),
                'mrr': round(sum(1 / r for r in ranks) / n, 4) if n else None}
    a={'bm25':agg(exp_a,'bm25_rank'),'dense':agg(exp_a,'dense_rank'),'hybrid':agg(exp_a,'hybrid_rank')}
    c={}
    for v in sorted({r['variant'] for r in exp_c}):
        rows = [r for r in exp_c if r['variant'] == v]
        c[v] = agg(rows, 'final_rank')
    return {'exp_a_within_doc':a,'exp_c_rrf_reranker':c,'n_rows':len(exp_a)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,default=Path('results/evaluation/day2_fix_v1'))
    ap.add_argument('--day3-dir',type=Path,default=Path('results/day3_evidence_v1'))
    ap.add_argument('--gold',type=Path,default=agent_eval.GOLD)
    ap.add_argument('--out-dir',type=Path,default=Path('results/day5_retrieval_experiments'))
    args=ap.parse_args()
    s=run_day5(args.run_dir,args.day3_dir,args.gold,args.out_dir)
    print(json.dumps(s,indent=2))

if __name__=='__main__':
    main()
