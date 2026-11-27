import json
import os
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.agent.extract import _valid_value, build_answer, extract_figures, normalize_label
from src.agent.graph import AgentGraph
from src.agent.llm import StubLLM
from src.agent.tools import _meta_rows_resolved

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "agent_figures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("fixture", "expected", "difference"),
    [
        (
            "t1_total_assets",
            {
                "srf|total assets": "21,557.12",
                "deepak_nitrite|total assets": "7,717.67",
            },
            "13,839.45",
        ),
        (
            "t2_total_equity",
            {
                "dr_reddys|total equity": "339,274",
                "aurobindo_pharma|total equity": "326,469.3",
            },
            "12,804.7",
        ),
        ("t3_revenue", {"coforge|revenue from operations": "91,790"}, None),
        ("t4_profit", {"adani_green|profit for the year": "604"}, None),
        (
            "t5_total_assets",
            {
                "maruti_suzuki|total assets": "1,319,718",
                "bosch|total assets": "202,411",
            },
            "1,117,307",
        ),
        (
            "t6_total_assets",
            {
                "aurobindo_pharma|total assets": "497,849.7",
                "dr_reddys|total assets": "494,266",
            },
            "3,583.7",
        ),
    ],
)
def test_fixture_figures(fixture: str, expected: dict[str, str], difference: str | None):
    data = _load(fixture)
    extracted = extract_figures(data["requested"], data["chunks"], basis="consolidated")
    answer = build_answer(data["requested"], extracted, data["query"])

    assert extracted["insufficient_evidence"] is False
    actual = {
        f"{fig['company_id']}|{normalize_label(fig['label'])}": fig["value"]
        for fig in answer["figures"]
    }
    assert actual == expected
    assert all(fig["period_column"] == "current" for fig in answer["figures"])
    assert all(fig["chunk_id"] and fig["page"] for fig in answer["figures"])
    if difference:
        assert difference in answer["answer"]
        assert answer["computed_values"] == [difference]


def test_conflict_reports_values_but_returns_one_figure():
    data = _load("conflict_total_assets")
    extracted = extract_figures(data["requested"], data["chunks"])
    answer = build_answer(data["requested"], extracted, data["query"])

    assert answer["insufficient_evidence"] is True
    assert answer["conflict"] is True
    assert len(answer["figures"]) == 1
    assert answer["figures"][0]["value"] in {"100", "200"}
    assert "100" in answer["answer"] and "200" in answer["answer"]


def test_total_equity_does_not_match_total_equity_and_liabilities():
    data = _load("equity_not_liabilities")
    extracted = extract_figures(data["requested"], data["chunks"])
    answer = build_answer(data["requested"], extracted, data["query"])

    assert answer["insufficient_evidence"] is False
    assert answer["figures"][0]["value"] == "50"
    assert answer["figures"][0]["label"] == "Total equity"


def test_total_assets_uses_equity_and_liabilities_only_when_no_total_assets_row():
    data = _load("total_assets_equivalence")
    extracted = extract_figures(data["requested"], data["chunks"])
    answer = build_answer(data["requested"], extracted, data["query"])

    assert answer["insufficient_evidence"] is False
    assert answer["figures"][0]["value"] == "100"
    assert answer["figures"][0]["label_equivalence"] is True


def test_t4_prefers_exact_label_chunk_and_uses_last_number():
    data = _load("t4_profit_both_formats")
    extracted = extract_figures(
        data["requested"], data["chunks"], basis="consolidated",
        statement_kind="income_statement",
    )

    assert extracted["insufficient_evidence"] is False
    assert extracted["figures"][0]["value"] == "604"
    assert extracted["figures"][0]["chunk_id"].endswith(":t263")


def test_graph_uses_deterministic_answer_without_answer_llm(monkeypatch):
    data = _load("t6_total_assets")
    monkeypatch.setattr(
        "src.agent.graph.detect_companies",
        lambda query: ["aurobindo_pharma", "dr_reddys"],
    )
    planner = json.dumps({
        "steps": [{"tool": "search_index", "args": {"query": "total assets", "k": 8}}],
        "requested": data["requested"],
    })
    llm = StubLLM([planner, json.dumps({
        "verdict": "accept", "feedback": "ok", "next_steps": [],
    })])

    def fake_search(query, k=5, **kwargs):
        return {"hits": data["chunks"], "match_info": {}}

    graph = AgentGraph(
        llm=llm,
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run(data["query"], thread_id="t-extract")

    final = result["final_answer"]
    assert "3,583.7" in final["answer"]
    assert final["insufficient_evidence"] is False
    assert final["company_mismatches"] == []
    assert len(llm.calls) == 1


def test_graph_runs_one_fallback_search_when_label_is_missing():
    planner = json.dumps({
        "steps": [{
            "tool": "search_index",
            "args": {"query": "alpha total assets", "k": 5, "company": "alpha"},
        }],
        "requested": [{"company_id": "alpha", "label": "Total assets"}],
    })
    calls: list[dict] = []

    def fake_search(query, k=5, **kwargs):
        calls.append({"query": query, "k": k, "company": kwargs.get("company")})
        if k == 60:
            return {"hits": [{
                "chunk_id": "f1",
                "text": "Total assets 10",
                "meta": {"company_id": "alpha", "statement_kind": "balance_sheet"},
            }]}
        return {"hits": []}

    graph = AgentGraph(
        llm=StubLLM([planner]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("What is total assets for alpha?", thread_id="t-fallback")

    assert [call["k"] for call in calls] == [20, 60]
    assert all(call["query"] == "Total assets" for call in calls)
    assert result["final_answer"]["figures"][0]["value"] == "10"
    assert result["final_answer"]["fallback_used"] is True
    assert result["final_answer"]["insufficient_evidence"] is False


def test_multi_year_t47_row_is_excluded_and_flagged():
    chunks = [{
        "chunk_id": "t47",
        "text": (
            "Total Assets 492,989 387,518 321,854 296,654 265,491\n"
            "Total current assets 250,181 248,053"
        ),
        "meta": {"company_id": "dr_reddys"},
    }]
    extracted = extract_figures(
        [{"company_id": "dr_reddys", "label": "Total assets"}], chunks
    )
    assert extracted["conflicts"] == []
    assert extracted["figures"][0]["multi_year_row"] is True


def test_t435_and_t6_total_assets_expected_values():
    data = _load("t6_total_assets")
    extracted = extract_figures(data["requested"], data["chunks"], basis="consolidated")
    actual = {
        f"{fig['company_id']}|{normalize_label(fig['label'])}": fig["value"]
        for fig in extracted["figures"]
    }
    assert actual["aurobindo_pharma|total assets"] == "497,849.7"
    assert actual["dr_reddys|total assets"] == "494,266"


def test_missing_label_is_reported_only_for_that_company():
    planner = json.dumps({
        "steps": [
            {"tool": "search_index", "args": {"query": "total assets", "k": 5, "company": "alpha"}},
            {"tool": "search_index", "args": {"query": "total assets", "k": 5, "company": "beta"}},
        ],
        "requested": [
            {"company_id": "alpha", "label": "Total assets"},
            {"company_id": "beta", "label": "Total assets"},
        ],
    })

    def fake_search(query, k=5, **kwargs):
        if k == 60:
            return {"hits": []}
        if kwargs.get("company") == "alpha":
            return {"hits": [{
                "chunk_id": "a1",
                "text": "Total assets 10",
                "meta": {"company_id": "alpha", "statement_kind": "balance_sheet"},
            }]}
        return {"hits": []}

    graph = AgentGraph(
        llm=StubLLM([planner]),
        tools={"search_index": fake_search},
        checkpointer=MemorySaver(),
    )
    result = graph.run("Compare total assets for alpha and beta", thread_id="t-missing")

    answer = result["final_answer"]
    assert "alpha Total assets: 10" in answer["answer"]
    assert "beta: label not found (Total assets)" in answer["answer"]
    assert answer["unresolved"] == [{"company_id": "beta", "label": "Total assets"}]
    assert answer["insufficient_evidence"] is True


def test_t263_equity_profit_uses_last_number_after_prior_balance():
    data = _load("t4_profit_both_formats")
    chunk = next(c for c in data["chunks"] if c["chunk_id"].endswith(":t263"))
    extracted = extract_figures(
        data["requested"],
        [chunk],
        basis="standalone",
        statement_kind="income_statement",
    )

    assert extracted["missing"] == []
    assert extracted["conflicts"] == []
    assert extracted["figures"][0]["value"] == "604"
    assert extracted["figures"][0]["equity_statement"] is True


def _coforge_chunk(suffix: str, page: int) -> dict:
    return {
        "chunk_id": f"coforge:{suffix}",
        "text": (
            "statement_kind: income_statement\n"
            "unit: \n"
            "reporting_basis: consolidated\n"
            "Revenue from operations 91,790 80,146: Revenue | from | operations"
        ),
        "meta": {
            "company_id": "coforge",
            "page": page,
            "period": "FY2024",
            "reporting_basis": "consolidated",
            "statement_kind": "income_statement",
        },
    }


def test_coforge_prefers_frequent_period_matched_value_over_noise():
    chunks = [
        _coforge_chunk("t390", 250),
        _coforge_chunk("t392", 250),
        _coforge_chunk("t391", 250),
        _coforge_chunk("t308", 210),
        {
            "chunk_id": "coforge:noise",
            "text": (
                "Revenue from operations 1,64,027 1,20,733: Revenue | from | operations\n"
                "Revenue from operations 48,489 42,305: Revenue | from | operations"
            ),
            "meta": {
                "company_id": "coforge",
                "page": 156,
                "period": "FY2025",
                "reporting_basis": "standalone",
                "statement_kind": "income_statement",
            },
        },
    ]
    extracted = extract_figures(
        [{"company_id": "coforge", "label": "Revenue from operations"}],
        chunks,
        basis="consolidated",
        statement_kind="income_statement",
        period="FY2024",
    )

    assert extracted["missing"] == []
    assert extracted["conflicts"] == []
    assert len(extracted["figures"]) == 1
    assert extracted["figures"][0]["value"] == "91,790"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("17,55,986", True),
        ("1,04,727", True),
        ("1,23,456", True),
        ("1,234,567", True),
        ("12,34", False),
        ("1,2,3", False),
    ],
)
def test_valid_value_rejects_malformed_indian_grouping(value: str, expected: bool):
    assert _valid_value(value) is expected


def test_meta_rows_cache_invalidates_on_chunks_change(tmp_path, monkeypatch):
    from src.agent import tools

    root = tmp_path / "index"
    root.mkdir()
    (root / "chunk_ids.json").write_text(json.dumps(["c1"]), encoding="utf-8")
    (root / "meta.json").write_text(
        json.dumps({"chunk_ids_sha256": "abc123", "chunks": 1}), encoding="utf-8"
    )
    chunks = root / "chunks.jsonl"
    chunks.write_text(
        json.dumps({"chunk_id": "c1", "meta": {"company_id": "old"}}) + "\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(tools, "_META_CACHE_DIR", cache_dir)
    _meta_rows_resolved.cache_clear()
    first = _meta_rows_resolved(str(root))
    old_names = {path.name for path in cache_dir.glob("*.pkl")}
    assert len(old_names) == 1

    stat = chunks.stat()
    chunks.write_text(
        json.dumps({"chunk_id": "c1", "meta": {"company_id": "new"}}) + "\n",
        encoding="utf-8",
    )
    os.utime(chunks, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    _meta_rows_resolved.cache_clear()
    second = _meta_rows_resolved(str(root))
    new_names = {path.name for path in cache_dir.glob("*.pkl")}
    assert first[0][0] == "old"
    assert second[0][0] == "new"
    assert old_names < new_names


@pytest.mark.skipif(not os.environ.get("AGENT_INDEX_PATH"), reason="real index not configured")
def test_t3_revenue_real_index_regression(tmp_path, monkeypatch):
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_SESSION_LOG_DIR", str(tmp_path / "sessions"))
    from src.agent.service import run_question

    result = run_question(
        "What is Coforge's revenue from operations for FY2025 consolidated?",
        session_id="test-t3-revenue-real-index",
    )
    assert result.answer == (
        "coforge Revenue from operations (FY2025): 91,790 (Rs. in million), prior 80,146"
    )
    assert "[15b2d66be472 p.250]" in result.citations


def test_conflict_detection_groups_by_period_scope_and_unit():
    def chunk(chunk_id: str, period: str, value: str) -> dict:
        return {
            "chunk_id": chunk_id,
            "text": f"Revenue from operations {value} 80,146: Revenue | from | operations",
            "meta": {
                "company_id": "coforge",
                "period": period,
                "reporting_basis": "consolidated",
                "statement_kind": "income_statement",
                "page": 250 if period == "FY2024" else 156,
            },
        }

    chunks = [
        chunk("fca9:t527", "FY2026", "1,64,027"),
        chunk("15b2:t390", "FY2024", "91,790"),
    ]
    extracted = extract_figures(
        [{"company_id": "coforge", "label": "Revenue from operations"}],
        chunks,
        basis="consolidated",
        statement_kind="income_statement",
        period="FY2025",
    )
    assert extracted["conflicts"] == []
    assert extracted["figures"][0]["value"] == "91,790"


def test_extract_cipla_p353_selects_fy2025_column_by_header():
    from src.agent.tools import get_chunk
    chunk = get_chunk("ac7e314c36eb9a51b806663ac56a49b06009ba3c14f05dc019b30d0694fd6d9b:p353")
    extracted = extract_figures(
        [{"company_id": "cipla", "label": "Revenue from operations", "period": "FY2025"}],
        [chunk],
        period="FY2025",
    )
    assert extracted["missing"] == []
    fig = extracted["figures"][0]
    assert fig["value"] == "27,547.62"
    assert fig["reporting_basis"] == "consolidated"
    assert fig["column_year"] == 2025
    assert fig["comparative_column"] is True


def test_extract_aurobindo_p205_selects_header_year_not_note():
    from src.agent.tools import get_chunk
    chunk = get_chunk("e97121aee57558adf2dc27d6770bec497df8a9cb4deeb37d67186abef5484ed2:p205")
    for period, value in [("FY2024", "290,018.7"), ("FY2025", "317,237.3")]:
        extracted = extract_figures(
            [{"company_id": "aurobindo_pharma", "label": "Revenue from operations", "period": period}],
            [chunk],
            period=period,
        )
        assert extracted["missing"] == []
        fig = extracted["figures"][0]
        assert fig["value"] == value
        assert fig["reporting_basis"] == "consolidated"
        assert fig["column_year"] == int(period[2:])


def test_extract_biocon_p349_consolidated_face_statement():
    from src.agent.tools import get_chunk
    chunk = get_chunk("4d1c338c892adf88bcfa40799c3d6c77dfc4e44c640cea76e1ca7de7b7bdc6c6:p349")
    extracted = extract_figures(
        [{"company_id": "biocon", "label": "Revenue from operations", "period": "FY2025"}],
        [chunk],
        period="FY2025",
    )
    assert extracted["missing"] == []
    fig = extracted["figures"][0]
    assert fig["value"] == "152,617"
    assert fig["reporting_basis"] == "consolidated"


def test_extract_rejects_standalone_or_note_rows_without_consolidated_header_match():
    from src.agent.tools import get_chunk
    standalone = get_chunk("9e092bb4d317944f9b65229c7d6cd2efe8032505deb88b934aa3a34f810884e6:p137")
    note = get_chunk("4d1c338c892adf88bcfa40799c3d6c77dfc4e44c640cea76e1ca7de7b7bdc6c6:p414")
    r1 = extract_figures(
        [{"company_id": "aurobindo_pharma", "label": "Revenue from operations", "period": "FY2024"}],
        [standalone],
        period="FY2024",
    )
    assert not any(str(f.get("reporting_basis")).lower() == "consolidated" for f in r1["figures"])
    r2 = extract_figures(
        [{"company_id": "biocon", "label": "Revenue from operations", "period": "FY2025"}],
        [note],
        period="FY2025",
    )
    assert r2["figures"] == [] or all(f.get("value") != "47" for f in r2["figures"])


def test_extract_rejects_summarised_note_and_selects_face_pl_comparative():
    from src.agent.tools import get_chunk
    summary = get_chunk("e57b7fc675de5a90df876e355e2b4c92b0d6928e6e8a67020496f9fc6700cb53:p367")
    face = get_chunk("4d1c338c892adf88bcfa40799c3d6c77dfc4e44c640cea76e1ca7de7b7bdc6c6:p349")
    r_summary = extract_figures(
        [{"company_id": "biocon", "label": "Revenue from operations", "period": "FY2024"}],
        [summary],
        period="FY2024",
    )
    assert not any(f.get("value") == "88,242" for f in r_summary["figures"])
    r_face = extract_figures(
        [{"company_id": "biocon", "label": "Revenue from operations", "period": "FY2024"}],
        [face],
        period="FY2024",
    )
    assert r_face["missing"] == []
    fig = r_face["figures"][0]
    assert fig["value"] == "147,557"
    assert fig.get("comparative_column") is True
    assert str(fig.get("reporting_basis")).lower() == "consolidated"


def test_face_pl_table_without_anchor_rows_is_not_rejected():
    chunk = {
        "chunk_id": "coforge:t999",
        "text": (
            "statement_kind: income_statement\n"
            "reporting_basis: consolidated\n"
            "Revenue from operations 91,790 80,146: Revenue | from | operations"
        ),
        "meta": {
            "company_id": "coforge", "page": 250, "period": "FY2024",
            "reporting_basis": "consolidated", "statement_kind": "income_statement",
            "chunk_type": "table",
        },
    }
    extracted = extract_figures(
        [{"company_id": "coforge", "label": "Revenue from operations"}],
        [chunk], basis="consolidated", statement_kind="income_statement",
    )
    assert extracted["missing"] == []
    assert extracted["figures"][0]["value"] == "91,790"
