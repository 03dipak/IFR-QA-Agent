"""Day 2 regression tests for token exhaustion and citation parsing (LLD-29)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agents import graph
from src.agents.executor import Observation
from src.agents.graph import run_agent
from src.eval import agent_eval as ae
from src.llmops.tracing import Tracer
from src.tools.retrieve import RetrievedChunk

PLAN_JSON = '{"steps":[{"sub_question":"q1","tool":"retrieve","args":{"query":"q1"}}]}'
CRITIC_PASS = '{"score":5,"issues":[],"verdict":"pass"}'
DOC = "a" * 12


class Chunk:
    def __init__(self, doc_id: str, page: int, text: str) -> None:
        self.doc_id = doc_id
        self.page = page
        self.text = text


def test_evidence_text_is_bounded() -> None:
    observations = [Observation(
        step=1, sub_question="q", tool="retrieve", args={},
        text="ignored", latency_ms=0,
        chunks=[Chunk(DOC, 3, "x" * 5000), Chunk("b" * 12, 4, "y" * 5000)],
    )]
    text = graph._build_evidence_text(observations)
    assert len(text) <= graph.MAX_EVIDENCE_CHARS
    assert f"[{DOC} p.3]" in text


def test_citation_refs_accept_quoted_ids_and_multipage() -> None:
    answer = f"value [doc_id '{DOC}' p.7] and [bbbbbbbbbbbb p.213, p.217]"
    assert graph._citation_refs(answer) == [
        (DOC[:12], 7), ("bbbbbbbbbbbb", 213), ("bbbbbbbbbbbb", 217),
    ]


def test_eval_citation_refs_accept_quoted_ids_and_multipage() -> None:
    answer = f"value [doc_id '{DOC}' p.7] and [bbbbbbbbbbbb p.213, p.217]"
    assert ae._citation_refs(answer) == [
        (DOC[:12], 7), ("bbbbbbbbbbbb", 213), ("bbbbbbbbbbbb", 217),
    ]


@dataclass
class FakeCompletion:
    text: str
    prompt_tokens: int = 10
    completion_tokens: int = 5
    latency_ms: int = 1


class FakeClient:
    def __init__(self, script: list[str]) -> None:
        self.script = list(script)

    def chat(self, system: str, user: str, max_tokens: int) -> FakeCompletion:
        return FakeCompletion(self.script.pop(0))


def test_run_agent_counts_critique_tokens(monkeypatch, tmp_path: Path) -> None:
    from src.agents.executor import Executor

    monkeypatch.setattr(graph, "resolve_query", lambda q: type(
        "R", (), {"as_dict": lambda self: {}})())
    client = FakeClient([
        PLAN_JSON, f"Total assets are 1,234 [{DOC[:12]} p.3].", CRITIC_PASS,
    ])
    result = run_agent(
        "What were total assets?", client=client,
        executor=Executor(tools={"retrieve": lambda **_: [
            RetrievedChunk(DOC, 3, "Total assets 1,234", 1.0),
        ]}),
        tracer=Tracer("tokens", root=tmp_path / "traces"),
    )
    assert result.tokens == 45


def test_citation_check_accepts_quoted_id(monkeypatch, tmp_path: Path) -> None:
    from src.citations.models import VerificationOutcome

    monkeypatch.setattr(graph, "direct_verify", lambda *args: VerificationOutcome.VERIFIED)
    observations = [Observation(
        step=1, sub_question="q", tool="retrieve", args={},
        text="", latency_ms=0, chunks=[Chunk(DOC, 7, "Total assets 1,234")],
    )]
    answer = f"Total assets are 1,234 [doc_id '{DOC}' p.7]."
    citations, issues = graph._citation_check(answer, observations)
    assert citations == [f"[{DOC[:12]} p.7]"]
    assert issues == []
