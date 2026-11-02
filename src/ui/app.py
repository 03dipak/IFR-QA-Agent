"""Streamlit UI entry point.

Implements lld_api_ui.txt section 6: four-page UI — question input,
answer display, trace dashboard, corpus status.
"""

import streamlit as st


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="IFR-QA Agent", page_icon="📊")
    st.title("Indian Financial-Report QA Agent")

    page = st.sidebar.selectbox(
        "Page",
        ["Question Input", "Answer Display", "Trace Dashboard", "Corpus Status"],
    )

    if page == "Question Input":
        _render_question_input()
    elif page == "Answer Display":
        _render_answer_display()
    elif page == "Trace Dashboard":
        _render_trace_dashboard()
    elif page == "Corpus Status":
        _render_corpus_status()


def _render_question_input():
    """Page 1: Question input with session toggle."""
    st.header("Ask a Question")
    question = st.text_area(
        "Question",
        placeholder="What was Tata Steel's revenue in FY2023?",
    )
    session_mode = st.radio("Session", ["New session", "Continue session"])
    if st.button("Submit"):
        st.info("API integration pending — see lld_api_ui.txt")


def _render_answer_display():
    """Page 2: Answer with evidence panel and trace summary."""
    st.header("Answer")
    st.info("Answer display pending — implements lld_api_ui.txt section 6")


def _render_trace_dashboard():
    """Page 3: Trace dashboard with per-turn timeline."""
    st.header("Trace Dashboard")
    st.info("Trace dashboard pending — implements lld_api_ui.txt section 6")


def _render_corpus_status():
    """Page 4: Corpus status with exclusion ledger."""
    st.header("Corpus Status")
    st.info("Corpus status pending — implements lld_api_ui.txt section 6")


if __name__ == "__main__":
    main()