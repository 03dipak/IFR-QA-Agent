from src.api import _build_response, _refusal_response
from src.safety.refusal import (
    INVESTMENT_ADVICE_REPLY,
    OUT_OF_SCOPE_REPLY,
    check_intent,
)
from src.ui.app import render_status, is_conversational


class _FakeResult:
    def __init__(self, answer, state, citations=None):
        self.answer = answer
        self.state = state
        self.citations = citations or []
        self.raw_citations = []
        self.session_id = "s1"
        self.latency_s = 0.01
        self.steps = []
        self.events = []
        self.token_count = 1


def test_investment_advice_refused():
    for q in [
        "Recommend three mutual funds",
        "Should I buy shares of Maruti Suzuki now?",
        "Which stock is good to buy in terms of profitability?",
    ]:
        r = check_intent(q)
        assert r.refused is True
        assert r.category == "investment_advice"
        assert r.reason == INVESTMENT_ADVICE_REPLY


def test_out_of_scope_refused():
    for q in [
        "What is the difference between a bear and a bull market?",
        "Who won the cricket world cup?",
    ]:
        r = check_intent(q)
        assert r.refused is True
        assert r.category == "out_of_scope"
        assert r.reason == OUT_OF_SCOPE_REPLY


def test_factual_report_questions_not_refused():
    for q in [
        "What was the share buyback amount reported by Infosys in FY2025?",
        "What mutual fund investments were held by Reliance in FY2024 consolidated?",
        "What were the inventories of SRF as at 31 March 2025 consolidated?",
        "How much did the company buy back shares for in FY2025?",
    ]:
        r = check_intent(q)
        assert r.refused is False, q


def test_refusal_api_response_shape():
    resp = _refusal_response(INVESTMENT_ADVICE_REPLY, "req1", 5, "investment_advice")
    assert resp.stopped_reason == "refused"
    assert resp.refusal.refused is True
    assert resp.answer == INVESTMENT_ADVICE_REPLY
    assert resp.citations == []
    assert resp.evidence == []
    assert resp.trace == []


def test_in_scope_missing_evidence_plain_wording():
    result = _FakeResult(
        "I could not find Total assets for Alpha in the available annual reports.",
        {"stop_reason": "done", "final_answer": {"insufficient_evidence": True, "figures": []}, "tool_results": []},
    )
    resp = _build_response(result)
    assert resp.stopped_reason == "done"
    assert "chunks" not in resp.answer.lower()
    assert "could not find" in resp.answer.lower()


def test_refusal_is_conversational_for_ui():
    assert is_conversational({"response_type": "refusal"}) is True
