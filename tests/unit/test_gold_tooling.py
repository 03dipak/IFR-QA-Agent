import json
import sys
from pathlib import Path

import pytest

import src.eval.gold_schema as gs
import src.eval.gold_rows as grow
import src.eval.gold_review as gres
import src.eval.gold_check as gchk
import src.eval.gold_freeze as gfrz
import src.eval.gold_author as gauth


def _row(rid="G-001", company="testco", metric="total assets",
         category="single_fact_lookup", answer="12,345",
         doc="a" * 64, page=3):
    return {
        "row_id": rid, "question_id": None,
        "category": category, "company_id": company,
        "period": "FY2025", "fiscal_year": "FY2025", "doc_type": "annual",
        "variant": "consolidated", "metric": metric,
        "question_text": f"Q{rid} assets?", "question": f"Q{rid} assets?",
        "expected_answer": answer, "answer_text": answer,
        "expected_value": 12345.0, "answer_value": 12345.0,
        "expected_unit": "money",
        "expected_unit_raw": "Rs. in crore",
        "expected_formula": None,
        "standalone_question": None,
        "evidence": [{"document_id": doc, "pdf_page": page,
                      "evidence_role": "primary", "excerpt": "Total assets 12,345",
                      "snippet_hash": "sha256:x"}],
        "evidence_span_text": "Total assets 12,345",
        "document_id": doc,
        "pdf_page_index": page,
        "printed_page_label": None,
        "table_refs": None,
        "extraction_checksum": None,
        "must_include": [], "must_not_include": [],
        "abstention_subtype": None, "depends_on": None,
        "relevance_rule": "primary",
        "expected_behaviour": "answer",
        "owner": None,
        "annotated_at": None, "re_annotated_at": None,
        "review_date": None,
        "review_status": "unreviewed", "lifecycle_status": "active",
        "reviewer": None, "reviewers": [], "row_revision": 0,
        "notes": "", "split": "dev",
        "annotator": None,
        "gold_schema_version": "2.0", "gold_set_version": "gold_v1",
        "sub_answers": None,
        "status": "active",
        "version": None,
        "id": rid,
        "signed_sha256": None,
    }


def test_sign_requires_reviewer_id():
    row = _row()
    with pytest.raises(ValueError):
        gres.sign(row, "")


def test_metric_not_applicable_subtype_is_accepted():
    row = _row("G-104", category="unanswerable_abstain")
    row["evidence"] = []
    row["evidence_span_text"] = ""
    row["abstention_subtype"] = "metric_not_applicable"
    row["expected_behaviour"] = "abstain"
    assert gs.validate_row(row) == []
    assert "metric_not_applicable" in gs.ABSTAIN_SUBTYPES


def test_sign_and_unsign_on_change(tmp_path, monkeypatch, capsys):
    row = _row()
    gold = tmp_path / "gold.jsonl"
    monkeypatch.setattr(gauth, "GOLD", str(gold))
    rows = [row]
    gres.sign(row, "reviewer-a")
    assert row["review_status"] == "self_reviewed"
    assert row["reviewers"] == ["reviewer-a"]
    row["expected_answer"] = "99,999"   # content changed after signing
    rc = gauth.cmd_check(rows, "reviewer-a")
    assert rc == 0                      # no structural failure
    assert row["review_status"] == "unreviewed"
    assert row["reviewers"] == []
    assert any("content changed" in e for e in gres.review_events(row))


def _seed_freeze(tmp_path, monkeypatch, companies):
    docs = []
    for i, co in enumerate(companies):
        docs.append({"document_id": f"{i:064x}", "sha256": f"{i:064x}",
                     "company_id": co, "doc_type": "annual", "fiscal_year": "FY2025",
                     "variant": "consolidated", "period_type": "annual",
                     "extraction_config_hash": "cfg-0",
                     "extraction_status": "extracted", "source": "manual"})
    fr = tmp_path / "freeze_v1.json"
    fr.write_text(json.dumps({"kept_ids": [d["document_id"] for d in docs],
                              "documents": docs,
                              "manifest_sha256": "cafebabe" * 8}))
    monkeypatch.setattr(gchk, "FREEZE", str(fr))
    monkeypatch.setattr(gfrz, "FREEZE", str(fr))
    pagedir = tmp_path / "pe"
    pagedir.mkdir()
    for d in docs:
        (pagedir / f"{d['document_id']}.pages.jsonl").write_text(
            json.dumps({"pdf_page_index": 3,
                        "page_text": "Revenue from operations 100\nTotal assets 12,345"}) + "\n")
    monkeypatch.setattr(gchk, "PAGE_DIR", str(pagedir))
    return docs


def test_abstain_coherence_report(tmp_path, monkeypatch):
    docs = _seed_freeze(tmp_path, monkeypatch, ["co_a", "co_b"])
    row_yes = _row("G-101", company="co_a", metric="total assets",
                   category="unanswerable_abstain",
                   doc=docs[0]["document_id"], page=3)
    row_yes["question"] = "what was total assets for co_a in FY2025? Abstain if absent."
    row_yes["evidence"] = [{"document_id": docs[0]["document_id"], "pdf_page": 3,
                            "evidence_role": "primary",
                            "excerpt": "Total assets 12,345",
                            "snippet_hash": "sha256:x"}]
    row_no = _row("G-102", company="co_b", metric="revenue from operations",
                  category="unanswerable_abstain",
                  doc=docs[1]["document_id"], page=3)
    row_no["question"] = "what was revenue from operations for co_b in FY2025? Abstain if absent."
    report = {r["id"]: r for r in gchk.abstain_keyword_report([row_yes, row_no])}
    assert report["G-101"]["cited_doc_hits"] >= 1
    assert report["G-101"]["hits"] >= 1
    assert report["G-102"]["cited_doc_hits"] >= 1   # page text has 'revenue from operations' too
    row_none = _row("G-103", company="co_b", metric="profit before tax",
                    category="unanswerable_abstain",
                    doc=docs[1]["document_id"], page=3)
    row_none["question"] = "what was profit before tax for co_b in FY2025? Abstain if absent."
    report = {r["id"]: r for r in gchk.abstain_keyword_report([row_none])}
    assert report["G-103"]["hits"] == 0 and report["G-103"]["cited_doc_hits"] == 0


def test_last_wins_persist(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"id": "G-1", "v": 1}) + "\n"
                    + json.dumps({"id": "G-1", "v": 2}) + "\n"
                    + json.dumps({"id": "G-2", "v": 3}) + "\n")
    rows = grow.load_rows(str(path))
    assert [r["v"] for r in rows] == [2, 3]


def _reviewed_pair(doc_a, doc_b):
    a = _row("G-001", company="co_a", doc=doc_a, page=3)
    a.update(review_status="self_reviewed", reviewers=["r1", "r2"], row_revision=2)
    b = _row("G-002", company="co_b", category="cross_company_comparison",
             doc=doc_b, page=3)
    b.update(review_status="peer_reviewed", reviewers=["r1"], row_revision=1)
    return [a, b]


def test_freeze_determinism(tmp_path, monkeypatch):
    docs = _seed_freeze(tmp_path, monkeypatch, ["co_a", "co_b"])
    rows = _reviewed_pair(docs[0]["document_id"], docs[1]["document_id"])
    for rep in range(2):
        m = gfrz.freeze(rows, seed=42, splitter_version="gold_freeze.v1")
        assert m["assignments"] == gfrz.freeze(rows, seed=42, splitter_version="gold_freeze.v1")["assignments"]
    assert gfrz.freeze(rows, seed=42, splitter_version="gold_freeze.v1")["assignments"]["G-001"] in ("dev", "holdout")


def test_freeze_records_corpus_sha_and_split(tmp_path, monkeypatch):
    docs = _seed_freeze(tmp_path, monkeypatch, ["co_a", "co_b"])
    gold = tmp_path / "g.jsonl"
    manifest = tmp_path / "g.split_manifest.json"
    monkeypatch.setattr(gfrz, "GOLD", str(gold))
    monkeypatch.setattr(gfrz, "MANIFEST", str(manifest))
    monkeypatch.setattr(gfrz, "MIN_REVIEWED", 2)
    rows = _reviewed_pair(docs[0]["document_id"], docs[1]["document_id"])
    grow.persist(rows, gold)
    sys.argv = ["gold_freeze"]
    assert gfrz.main() == 0
    m = json.loads(manifest.read_text())
    assert m["corpus_freeze_manifest_sha256"] == "cafebabe" * 8
    assert m["n_reviewed"] == 2
    assert m["split_manifest_hash"]
    cats = set(m["counts_by_category"])
    assert cats == {"single_fact_lookup", "cross_company_comparison"}
    # second run without --force refuses
    assert gfrz.main() == 2


def test_freeze_below_floor_blocked(tmp_path, monkeypatch):
    docs = _seed_freeze(tmp_path, monkeypatch, ["co_a", "co_b"])
    gold = tmp_path / "g.jsonl"
    manifest = tmp_path / "g.split_manifest.json"
    monkeypatch.setattr(gfrz, "GOLD", str(gold))
    monkeypatch.setattr(gfrz, "MANIFEST", str(manifest))
    rows = _reviewed_pair(docs[0]["document_id"], docs[1]["document_id"])[:1]
    grow.persist(rows, gold)
    sys.argv = ["gold_freeze"]
    assert gfrz.main() == 2
    assert not manifest.exists()

# ---------------------------------------------------------------------------
# gold_merge / --verify --ids
# ---------------------------------------------------------------------------

import src.eval.gold_merge as gmerge


def _write_gold(path, rows):
    path.write_text("".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows))


def _signed_row(rid, answer="12,345", reviewers=("reviewer-a",)):
    r = _row(rid, answer=answer)
    r.update(review_status="self_reviewed", reviewer=reviewers[0],
             reviewers=list(reviewers), row_revision=1,
             signed_sha256="sha256:x", annotated_at="2026-11-08T00:00:00Z")
    return r


def test_disjoint_merge_combines_signatures(tmp_path, capsys):
    keep = tmp_path / "keep.jsonl"
    a = _signed_row("G-001", answer="12,345", reviewers=("reviewer-a",))
    b = _signed_row("G-001", answer="12,345", reviewers=("reviewer-b",))
    c = _signed_row("G-002", answer="9,876", reviewers=("reviewer-a",))
    d = _signed_row("G-003", answer="1,000", reviewers=("reviewer-b",))
    _write_gold(tmp_path / "a.jsonl", [a, c])
    _write_gold(tmp_path / "b.jsonl", [b, d])
    out = tmp_path / "merged.jsonl"
    import sys
    argv_saved = sys.argv
    try:
        sys.argv = ["gold_merge", "--inputs", str(tmp_path / "a.jsonl"),
                    str(tmp_path / "b.jsonl"), "--output", str(out)]
        rc = gmerge.main()
    finally:
        sys.argv = argv_saved
    assert rc == 0
    rows = {r["id"]: r for r in grow.load_rows(str(out))}
    assert rows["G-001"]["reviewers"] == ["reviewer-a", "reviewer-b"]
    assert rows["G-001"]["review_status"] == "peer_reviewed"
    assert rows["G-002"]["reviewers"] == ["reviewer-a"]
    assert rows["G-003"]["reviewers"] == ["reviewer-b"]
    outtxt = capsys.readouterr().out
    assert "sha256:" in outtxt


def test_conflicting_merge_refused(tmp_path, capsys):
    a = _signed_row("G-001", answer="12,345", reviewers=("reviewer-a",))
    b = _signed_row("G-001", answer="99,999", reviewers=("reviewer-b",))
    _write_gold(tmp_path / "a.jsonl", [a])
    _write_gold(tmp_path / "b.jsonl", [b])
    out = tmp_path / "merged.jsonl"
    import sys
    argv_saved = sys.argv
    try:
        sys.argv = ["gold_merge", "--inputs", str(tmp_path / "a.jsonl"),
                    str(tmp_path / "b.jsonl"), "--output", str(out)]
        rc = gmerge.main()
    finally:
        sys.argv = argv_saved
    assert rc == 1
    err = capsys.readouterr().err
    assert "MERGE REFUSED" in err and "G-001" in err
    assert not out.exists()


def test_parse_ids():
    assert gauth._parse_ids("G-001..G-003") == {"G-001", "G-002", "G-003"}
    assert gauth._parse_ids("G-003,G-001") == {"G-001", "G-003"}
    assert gauth._parse_ids("G-002") == {"G-002"}
    with pytest.raises(ValueError):
        gauth._parse_ids("X..Y")


def test_verify_ids_filter(tmp_path, monkeypatch, capsys):
    rows = [_row(f"G-{n:03d}") for n in range(1, 6)]
    _write_gold(tmp_path / "g.jsonl", rows)
    monkeypatch.setattr(gauth, "GOLD", str(tmp_path / "g.jsonl"))
    monkeypatch.setattr(gauth, "REVIEW_DIR", tmp_path / "review")
    inputs = iter(["rv1", "enter", "enter", "enter"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    class A: pass
    a = A()
    a.reviewer = ""
    a.ids = "G-001..G-003"
    a.fast = True
    from src.eval import gold_author as ga
    rc = ga.cmd_verify(rows, a)
    assert rc == 0
    assert rows[0]["review_status"] == "self_reviewed"
    assert rows[1]["review_status"] == "self_reviewed"
    assert rows[2]["review_status"] == "self_reviewed"
    assert rows[3]["review_status"] == "unreviewed"
    assert rows[4]["review_status"] == "unreviewed"


def test_verify_ids_unknown_signer_flag(tmp_path, monkeypatch):
    rows = [_row("G-001")]
    monkeypatch.setattr(gauth, "GOLD", str(tmp_path / "g.jsonl"))
    monkeypatch.setattr(gauth, "REVIEW_DIR", tmp_path / "review")
    inputs = iter(["rv2", "enter"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    class A: pass
    a = A(); a.reviewer = ""; a.ids = "G-001"; a.fast = True
    assert gauth.cmd_verify(rows, a) == 0
    assert rows[0]["review_status"] == "self_reviewed"
    assert rows[0]["reviewers"] == ["rv2"]

# ---------------------------------------------------------------------------
# LLD-aligned v2 schema: migration, drift, scoring
# ---------------------------------------------------------------------------

import src.eval.gold_migrate as gmig


def test_migration_preserves_signatures():
    rows = _reviewed_pair("a" * 64, "b" * 64)
    rows = [dict(r, annotated_at="2026-11-08T00:00:00Z",
                 re_annotated_at="2026-11-08T00:00:00Z") for r in rows]
    for r in rows:
        r["review_status"] = "self_reviewed"
        r["reviewers"] = ["reviewer-a"]
        r.update({"id": r.get("id", r["row_id"])})
    rows = [gmig.migrate_row(r) for r in rows]
    # after migration the signature is re-stamped w/ v2 fields but rows stay signed
    for r in rows:
        assert r["review_status"] in ("self_reviewed", "peer_reviewed")
        assert r["signed_sha256"]
    assert all(r["gold_schema_version"] == "2.0" for r in rows)
    assert all("question_text" in r and "expected_value" in r for r in rows)


def test_drift_check_flags_changed_page(tmp_path, monkeypatch):
    docs = _seed_freeze(tmp_path, monkeypatch, ["co_a", "co_b"])
    monkeypatch.setattr(gmig, "PAGE_DIR", str(tmp_path / "pe"))
    row = _row("G-101", doc=docs[0]["document_id"], page=3)
    row["extraction_checksum"] = gmig.page_checksum(docs[0]["document_id"], 3)
    row["evidence_span_text"] = "Total assets 12,345"
    # page unchanged -> no drift
    assert gmig.page_checksum(docs[0]["document_id"], 3) == row["extraction_checksum"]
    # now corrupt the page text
    path = tmp_path / "pe" / f"{docs[0]['document_id']}.pages.jsonl"
    path.write_text('{"pdf_page_index": 3, "page_text": "WRONG PAGE"}\n')
    assert gmig.page_checksum(docs[0]["document_id"], 3) != row["extraction_checksum"]


def test_score_run_denominator_and_errors(tmp_path, monkeypatch):
    import src.eval.prompt_baseline as pb
    gold_path = tmp_path / "gold.jsonl"
    bands_path = tmp_path / "scoring_bands.json"
    rows = [
        {"row_id": "G-001", "category": "single_fact_lookup", "expected_answer": "12,345"},
        {"row_id": "G-002", "category": "single_fact_lookup", "expected_answer": "6,789"},
    ]
    gold_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    bands_path.write_text(json.dumps({"normalisation_rules_version": "norm-2", "registry_version": "v1"}))
    monkeypatch.setattr(pb, "GOLD", gold_path)
    monkeypatch.setattr(pb, "BANDS", bands_path)
    monkeypatch.setattr(pb, "SPLIT_MANIFEST", tmp_path / "gold_data.split_manifest.json")
    gold0, bands0 = pb._gold_tuple()
    out_dir = tmp_path / "run1"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({
        "n_rows": 2, "gold_hash": gold0, "scoring_bands_hash": bands0,
        "split_manifest_hash": "x"}))
    (out_dir / "outputs.jsonl").write_text(
        json.dumps({"row_id": rows[0]["row_id"], "category": "single_fact_lookup",
                    "expected_answer": rows[0]["expected_answer"], "pdf_page": 1,
                    "live": False, "status": "not_live", "output": ""}) + "\n"
        + json.dumps({"row_id": rows[1]["row_id"], "category": "single_fact_lookup",
                      "expected_answer": rows[1]["expected_answer"], "pdf_page": 1,
                      "live": False, "status": "answered",
                      "output": f"answer: {rows[1]['expected_answer']} | page: 1"}) + "\n")
    r = pb.score_run(out_dir)
    assert r["n"] == 2
    assert r["errors"] == 1
    assert r["error_rate"] == 0.5
    assert r["unreportable"] is True  # >5%


def test_holdout_guard_blocks_second_system(tmp_path, monkeypatch):
    import src.eval.prompt_baseline as pb
    monkeypatch.setattr(pb, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(pb, "holdout_guard", pb.holdout_guard)
    assert pb.holdout_guard("B0") is True
    assert pb.holdout_guard("B0") is False  # second refused
    assert pb.holdout_guard("B0", force=True) is True


def test_scoring_determinism(tmp_path):
    import src.eval.prompt_baseline as pb
    rows = pb.load_jsonl("data/eval/datasets/gold_data.jsonl")[:2]
    gold0, bands0 = pb._gold_tuple()
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "manifest.json").write_text(json.dumps({
        "n_rows": 2, "gold_hash": gold0, "scoring_bands_hash": bands0,
        "split_manifest_hash": "x"}))
    (out_dir / "outputs.jsonl").write_text(json.dumps({
        "row_id": rows[0]["row_id"], "category": "single_fact_lookup",
        "expected_answer": rows[0]["expected_answer"], "pdf_page": 1,
        "live": False, "status": "answered",
        "output": f"answer: {rows[0]['expected_answer']} | page: 1"}) + "\n")
    a = pb.score_run(out_dir)
    b = pb.score_run(out_dir)
    assert a == b


def test_stated_precision_bands():
    from src.eval.numeric_scoring import numeric_match, stated_precision_tolerance
    assert not numeric_match("0.30", "0.80")    # EPS vs EPS
    assert numeric_match("19.5", "19.51")       # same quantum neighbour
    assert not numeric_match("19.51", "19.5")   # expected printed to 2dp
    assert not numeric_match("0.002", "0.05")   # ratio
    assert stated_precision_tolerance("0.30") == 0.005
    assert stated_precision_tolerance("19.5") == 0.05
    assert stated_precision_tolerance("29,440") == 0.5


def test_run_all_skips_superseded_rows(monkeypatch):
    """Superseded rows must not produce fails or warns (lld_eval.txt amendment policy)."""
    row_bad = _row(rid="G-002", category="not-a-category", answer="banana")
    row_bad["status"] = "superseded"
    row_bad["lifecycle_status"] = "superseded"
    row_bad["evidence"] = [{"document_id": "f" * 64, "pdf_page": 99,
                            "evidence_role": "primary", "excerpt": "not in corpus",
                            "snippet_hash": "sha256:x"}]
    # unknown document/page both produced fails before; must produce none now
    monkeypatch.setattr(gchk, "freeze_ids", lambda: set())
    monkeypatch.setattr(gchk, "_page_text", lambda did, pg: None)
    failing, warns, summary = gchk.run_all([row_bad], allocation=None)
    assert failing == []
    assert summary["n_superseded"] == 1


def test_depends_on_must_resolve_to_live_row():
    r1 = _row("G-123")
    r2 = _row("G-124")
    r2["depends_on"] = "G-123"
    assert gchk.check_depends_on_resolves([r1, r2]) == []
    r2["depends_on"] = "D-223"  # stale draft id: must FAIL
    fails = gchk.check_depends_on_resolves([r1, r2])
    assert fails and fails[0][0] == "G-124" and "D-223" in fails[0][1]
    r2["depends_on"] = "G-123"
    r1["status"] = "superseded"  # superseded partner: pair not whole
    fails = gchk.check_depends_on_resolves([r1, r2])
    assert fails and fails[0][0] == "G-124" and "does not resolve" in fails[0][1]


def test_load_rows_active_only_excludes_superseded(tmp_path):
    path = tmp_path / "gold.jsonl"
    r1 = _row("G-001"); r2 = _row("G-002"); r2["status"] = "superseded"
    path.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")
    assert [r["id"] for r in grow.load_rows(str(path))] == ["G-001", "G-002"]
    assert [r["id"] for r in grow.load_rows(str(path), active_only=True)] == ["G-001"]
