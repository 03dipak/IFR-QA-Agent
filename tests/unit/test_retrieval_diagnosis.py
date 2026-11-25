"""Unit tests for Day 4 retrieval diagnosis helpers (LLD-29)."""
from __future__ import annotations

import numpy as np

from src.eval import retrieval_diagnosis as rd


def _ranked(n, target_rank, target_idx=10, doc_idx=11, doc_rank=5):
    scores = np.linspace(1.0, 0.0, n, dtype=np.float32)
    # place target at target_rank (1-based)
    scores[target_idx] = scores[target_rank - 1] - 1e-6
    scores[doc_idx] = scores[doc_rank - 1] - 5e-7
    return scores


def _stage(bm25_rank, dense_rank, rrf_rank, final=False, doc_bm25=5, doc_dense=5, doc_rrf=5):
    n = 200
    ids = [f"{i:012x}" + "0" * 52 + f"::C2::p{i}::w0" for i in range(n)]
    ids[10] = "aaaaaaaaaaaa" + "0" * 52 + "::C2::p99::w0"
    ids[11] = "aaaaaaaaaaaa" + "0" * 52 + "::C2::p50::w0"
    bm25 = _ranked(n, bm25_rank, doc_rank=doc_bm25)
    dense = _ranked(n, dense_rank, doc_rank=doc_dense)
    rrf = _ranked(n, rrf_rank, doc_rank=doc_rrf)
    return {
        "ids": ids,
        "_bm25": bm25, "_dense": dense, "_rrf": rrf,
        "_bm25_r": rd.full_ranks(bm25), "_dense_r": rd.full_ranks(dense),
        "_rrf_r": rd.full_ranks(rrf),
        "final_top": [{"doc_id": "aaaaaaaaaaaa", "page": 99, "rank": 1}] if final else [],
    }


def test_classify_reranker_drop():
    st = _stage(10, 10, 10, final=False)
    cause, _ = rd.classify("aaaaaaaaaaaa", 99, st,
                           {"page_exists": True, "excerpt_in_page": True,
                            "excerpt_found_pages": [99]},
                           pool=30, topk=5)
    assert cause == "reranker_dropped_expected_page"


def test_classify_rrf_loss():
    st = _stage(10, 10, 80, final=False)
    cause, _ = rd.classify("aaaaaaaaaaaa", 99, st,
                           {"page_exists": True, "excerpt_in_page": True,
                            "excerpt_found_pages": [99]},
                           pool=30, topk=5)
    assert cause == "rrf_fusion_loss"


def test_classify_correct_doc_wrong_page():
    st = _stage(80, 80, 80, final=False, doc_rrf=5)
    cause, _ = rd.classify("aaaaaaaaaaaa", 99, st,
                           {"page_exists": True, "excerpt_in_page": True,
                            "excerpt_found_pages": [99]},
                           pool=30, topk=5)
    assert cause == "correct_doc_wrong_page_ranking"


def test_classify_page_offset():
    st = _stage(10, 10, 10, final=False)
    cause, _ = rd.classify("aaaaaaaaaaaa", 99, st,
                           {"page_exists": True, "excerpt_in_page": False,
                            "excerpt_found_pages": [100]},
                           pool=30, topk=5)
    assert cause == "pdf_page_number_offset"
