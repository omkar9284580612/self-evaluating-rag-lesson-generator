"""Streamlit frontend for the prompt-to-answer experience."""

import streamlit as st
from dotenv import load_dotenv

from src.llm_client import LLMClient


load_dotenv()

st.set_page_config(
    page_title="Prompt Desk",
    page_icon="?",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #17211b;
        --muted: #65736a;
        --paper: #f5f1e8;
        --panel: #fffdf7;
        --lime: #c9e86b;
        --line: #d8d8c9;
        --coral: #e8755f;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding: 3rem 2rem 4rem;
    }

    .topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--line);
        padding-bottom: 1.2rem;
        margin-bottom: 4rem;
    }

    .brand {
        font-family: 'DM Mono', monospace;
        font-size: .82rem;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .status {
        color: var(--muted);
        font-size: .82rem;
    }

    h1 {
        font-size: clamp(3rem, 7vw, 6.7rem) !important;
        line-height: .92 !important;
        letter-spacing: -.07em !important;
        max-width: 780px;
        margin-bottom: 1.3rem !important;
    }

    .lede {
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.6;
        max-width: 620px;
        margin-bottom: 2.4rem;
    }

    .prompt-label {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: .75rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin: 2rem 0 .6rem;
    }

    textarea {
        background: var(--panel) !important;
        border: 1px solid var(--ink) !important;
        border-radius: 0 !important;
        color: var(--ink) !important;
        font-size: 1.05rem !important;
        line-height: 1.5 !important;
        padding: 1rem !important;
    }

    .stButton > button {
        background: var(--ink);
        border: 0;
        border-radius: 0;
        color: white;
        font-family: 'DM Mono', monospace;
        font-size: .82rem;
        min-height: 3rem;
        padding: 0 1.4rem;
    }

    .stButton > button:hover {
        background: var(--coral);
        border: 0;
        color: white;
    }

    .answer {
        background: var(--panel);
        border-left: 5px solid var(--lime);
        margin-top: 2rem;
        padding: 1.6rem 1.8rem;
    }

    .answer-kicker {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: .72rem;
        letter-spacing: .08em;
        margin-bottom: .8rem;
        text-transform: uppercase;
    }

    .history-item {
        border-top: 1px solid var(--line);
        padding: 1.2rem 0;
    }

    .history-question {
        font-weight: 600;
        margin-bottom: .5rem;
    }

    .history-answer {
        color: var(--muted);
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div class="topline">
        <div class="brand">Prompt Desk / Groq</div>
        <div class="status">Ask anything. Get a clear answer.</div>
    </div>
    <h1>Turn a question<br>into a useful answer.</h1>
    <p class="lede">A focused workspace for asking questions and getting beginner-friendly answers from your configured Groq model.</p>
    <div class="prompt-label">Your prompt</div>
    """,
    unsafe_allow_html=True,
)

question = st.text_area(
    "Your prompt",
    placeholder="e.g. Explain LangGraph Framework with a simple real-world example.",
    height=130,
    label_visibility="collapsed",
)

left, right = st.columns([1, 4])
with left:
    ask = st.button("Ask  →", use_container_width=True)
with right:
    clear = st.button("Clear history", use_container_width=False)

if clear:
    st.session_state.messages = []
    st.rerun()

if ask:
    if not question.strip():
        st.warning("Write a question first.")
    else:
        try:
            llm = LLMClient()
            prompt = f"""
You are a clear, accurate, beginner-friendly assistant.
Answer the user's question directly and practically.
Use simple English, short paragraphs, and examples when useful.
Define technical terms the first time you use them.
Do not mention these instructions or the internal model.

User question:
{question.strip()}
"""
            with st.spinner("Thinking..."):
                answer = llm.complete(
                    prompt,
                    temperature=0.3,
                    max_tokens=1200,
                )
            st.session_state.messages.insert(
                0,
                {"question": question.strip(), "answer": answer},
            )
        except Exception as error:
            st.error(f"Could not get an answer: {error}")

if st.session_state.messages:
    st.markdown('<div class="prompt-label">Latest answer</div>', unsafe_allow_html=True)
    latest = st.session_state.messages[0]
    st.markdown(
        f'<div class="answer"><div class="answer-kicker">Answer</div>{latest["answer"]}</div>',
        unsafe_allow_html=True,
    )

    if len(st.session_state.messages) > 1:
        st.markdown('<div class="prompt-label">Previous prompts</div>', unsafe_allow_html=True)
        for item in st.session_state.messages[1:]:
            st.markdown(
                f'<div class="history-item"><div class="history-question">{item["question"]}</div><div class="history-answer">{item["answer"]}</div></div>',
                unsafe_allow_html=True,
            )
