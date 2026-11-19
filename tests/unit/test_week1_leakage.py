import json
from pathlib import Path

from src.config import GOLD_PATH
from src.eval.contexts import load_few_shot as load_dev_few_shot
from src.eval.prompting.inputs import load_dev_rows, load_holdout_ids
from src.eval.prompting.fewshot import load_few_shot as load_s1_few_shot


def _all_rows():
    rows = []
    for line in Path(GOLD_PATH).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _gold_evidence_pairs(rows):
    out = set()
    for r in rows:
        for ev in r.get("evidence") or []:
            if ev.get("document_id") and isinstance(ev.get("pdf_page"), int):
                out.add((ev["document_id"], ev["pdf_page"]))
    return out


def _gold_doc_ids(rows):
    out = set()
    for r in rows:
        if r.get("document_id"):
            out.add(r["document_id"])
        for ev in r.get("evidence") or []:
            if ev.get("document_id"):
                out.add(ev["document_id"])
    return out


def test_dev_excludes_holdout_rows():
    dev = load_dev_rows()
    hold = load_holdout_ids()
    assert hold
    assert all(str(r.get("id")) not in hold for r in dev)
    assert all(r.get("split") == "dev" for r in dev)


def test_b0_few_shot_excludes_eval_holdout_and_same_doc():
    rows = _all_rows()
    hold = {str(r["id"]) for r in rows if r.get("split") == "holdout"}
    dev = load_dev_rows()
    for row in dev[:20]:
        rid = str(row["id"])
        rdocs = {row.get("document_id")} | {e.get("document_id") for e in (row.get("evidence") or [])}
        rdocs = {d for d in rdocs if isinstance(d, str) and d}
        fs = load_dev_few_shot(n=3, exclude_id=rid, exclude_doc_ids=rdocs)
        fids = {str(r["id"]) for r in fs}
        assert rid not in fids
        assert not (fids & hold)
        for ex in fs:
            edocs = {ex.get("document_id")} | {e.get("document_id") for e in (ex.get("evidence") or [])}
            edocs = {d for d in edocs if isinstance(d, str) and d}
            assert edocs
            assert not (edocs & rdocs)
            assert ex.get("split") == "dev"


def test_s1_static_few_shot_no_gold_overlap_or_ambiguous_identity():
    rows = _all_rows()
    cited = _gold_evidence_pairs(rows)
    docs = _gold_doc_ids(rows)
    fs = load_s1_few_shot()
    assert 1 <= len(fs) <= 3
    for ex in fs:
        did = ex.get("context_document_id")
        pg = ex.get("context_page")
        assert isinstance(did, str) and did.strip()
        assert isinstance(pg, int)
        assert (did, pg) not in cited
        assert did not in docs
        assert isinstance(ex.get("answer"), dict)


def test_tech_mahindra_few_shot_examples_audited():
    fs_path = Path("data/eval/prompts/fewshot_v1.jsonl")
    raw = [json.loads(l) for l in fs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(raw) == 3
    tm = [r for r in raw if "tech mahindra" in (r.get("question") or "").lower()]
    assert len(tm) == 3
    rows = _all_rows()
    docs = _gold_doc_ids(rows)
    for ex in tm:
        assert ex["context_document_id"] == "007171d132177317b6e3d4874a5b231f14440c19a7c5ef6fbdebef836270edfa"
        assert ex["context_document_id"] not in docs
