from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from backend.chat import answer_question
from backend.embeddings import DEFAULT_EMBEDDING_MODEL, build_vector_store
from backend.flashcards import generate_flashcards
from backend.pdf_processing import Paper, process_pdf, save_uploaded_pdf
from backend.quiz_generator import generate_quiz
from backend.summarizer import extract_key_findings, generate_summary
from backend.utils import UPLOAD_DIR, VECTOR_STORE_DIR, ensure_directories, get_provider_api_key


load_dotenv()
ensure_directories()

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon=":blue_book:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #2563eb;
            --surface: #f8fafc;
            --card: #ffffff;
            --border: #e5e7eb;
            --text: #111827;
            --muted: #6b7280;
        }
        .stApp {
            background: #ffffff;
            color: var(--text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        section[data-testid="stSidebar"] {
            background: #f9fafb;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .info-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            margin-bottom: 12px;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }
        .source-chip {
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 3px 10px;
            margin: 3px 4px 3px 0;
            font-size: 0.8rem;
        }
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #bfdbfe;
            color: #1d4ed8;
            background: #ffffff;
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: #2563eb;
            color: #1e40af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "papers": [],
        "vector_store": None,
        "chat_history": [],
        "summary": "",
        "key_findings": "",
        "flashcards": [],
        "quiz": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def process_uploads(uploaded_files) -> None:
    processed: list[Paper] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        try:
            saved_path = save_uploaded_pdf(uploaded_file, UPLOAD_DIR)
            processed.append(process_pdf(saved_path))
        except ValueError as exc:
            errors.append(str(exc))

    if processed:
        st.session_state.papers = processed
        with st.spinner("Creating searchable paper index..."):
            try:
                st.session_state.vector_store = build_vector_store(
                    processed,
                    VECTOR_STORE_DIR,
                    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
                )
                st.success(f"Processed {len(processed)} paper(s).")
            except Exception as exc:
                st.session_state.vector_store = None
                st.error(f"PDF text was extracted, but the search index could not be created: {exc}")

    for error in errors:
        st.warning(error)


MODEL_OPTIONS = {
    "Google Gemini": ["gemini-3.5-flash", "gemini-3.1-flash-lite"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
}


def require_api_key(api_key: str | None, provider: str) -> bool:
    if api_key:
        return True
    env_hint = "GOOGLE_API_KEY or GEMINI_API_KEY" if provider == "Google Gemini" else "OPENAI_API_KEY"
    st.error(f"Add your {provider} API key in the sidebar or set {env_hint} in your environment.")
    return False


def require_papers() -> bool:
    if st.session_state.papers:
        return True
    st.error("Upload and process at least one paper first.")
    return False


def get_secret_key(provider: str) -> str | None:
    key_names = ["GOOGLE_API_KEY", "GEMINI_API_KEY"] if provider == "Google Gemini" else ["OPENAI_API_KEY"]
    for key_name in key_names:
        try:
            value = st.secrets.get(key_name)
        except Exception:
            value = None
        if value:
            return str(value)
    return None


def render_sidebar() -> tuple[str | None, str, str]:
    with st.sidebar:
        st.header("Research Assistant")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more research papers in PDF format.",
        )
        if st.button("Process papers", use_container_width=True):
            if uploaded_files:
                process_uploads(uploaded_files)
            else:
                st.warning("Please upload at least one PDF first.")

        st.divider()
        st.subheader("Uploaded Papers")
        if st.session_state.papers:
            for paper in st.session_state.papers:
                st.markdown(f"- {paper.file_name}")
        else:
            st.caption("No papers processed yet.")

        st.divider()
        st.subheader("Settings")
        provider = st.selectbox("LLM Provider", list(MODEL_OPTIONS.keys()), index=0)
        api_label = "Gemini API Key" if provider == "Google Gemini" else "OpenAI API Key"
        api_placeholder = "AIza..." if provider == "Google Gemini" else "sk-..."
        api_key = st.text_input(api_label, type="password", placeholder=api_placeholder)
        model_name = st.selectbox(
            "LLM Model",
            MODEL_OPTIONS[provider],
            index=0,
        )
        st.caption("Embeddings use sentence-transformers/all-MiniLM-L6-v2 locally.")

    return get_provider_api_key(provider, api_key or get_secret_key(provider)), provider, model_name


def render_paper_metrics() -> None:
    papers: list[Paper] = st.session_state.papers
    total_words = sum(paper.word_count for paper in papers)
    total_pages = sum(paper.page_count for paper in papers)
    col1, col2, col3 = st.columns(3)
    col1.metric("Papers", len(papers))
    col2.metric("Pages", total_pages)
    col3.metric("Words", f"{total_words:,}")


def render_chat(api_key: str | None, provider: str, model_name: str) -> None:
    st.subheader("Ask Questions")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown(
                    " ".join(f"<span class='source-chip'>{source}</span>" for source in message["sources"]),
                    unsafe_allow_html=True,
                )

    question = st.chat_input("Ask about the uploaded papers")
    if question:
        if not st.session_state.vector_store:
            st.error("Upload and process papers before asking questions.")
            return
        if not require_api_key(api_key, provider):
            return

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Reading the relevant paper sections..."):
                try:
                    answer, sources = answer_question(
                        question,
                        st.session_state.vector_store,
                        api_key=api_key,
                        provider=provider,
                        model_name=model_name,
                    )
                    st.markdown(answer)
                    if sources:
                        st.markdown(
                            " ".join(f"<span class='source-chip'>{source}</span>" for source in sources),
                            unsafe_allow_html=True,
                        )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")


def render_summary_tools(api_key: str | None, provider: str, model_name: str) -> None:
    st.subheader("Summaries and Findings")
    col1, col2, col3 = st.columns(3)

    if col1.button("Short summary", use_container_width=True):
        if require_papers() and require_api_key(api_key, provider):
            with st.spinner("Generating short summary..."):
                try:
                    st.session_state.summary = generate_summary(
                        st.session_state.papers,
                        api_key,
                        "short",
                        provider=provider,
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"Could not generate the summary: {exc}")

    if col2.button("Detailed summary", use_container_width=True):
        if require_papers() and require_api_key(api_key, provider):
            with st.spinner("Generating detailed summary..."):
                try:
                    st.session_state.summary = generate_summary(
                        st.session_state.papers,
                        api_key,
                        "detailed",
                        provider=provider,
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"Could not generate the summary: {exc}")

    if col3.button("Key findings", use_container_width=True):
        if require_papers() and require_api_key(api_key, provider):
            with st.spinner("Extracting key findings..."):
                try:
                    st.session_state.key_findings = extract_key_findings(
                        st.session_state.papers,
                        api_key,
                        provider=provider,
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"Could not extract key findings: {exc}")

    if st.session_state.summary:
        st.markdown("#### Summary")
        st.markdown(st.session_state.summary)
    if st.session_state.key_findings:
        st.markdown("#### Key Findings")
        st.markdown(st.session_state.key_findings)


def render_flashcards(api_key: str | None, provider: str, model_name: str) -> None:
    st.subheader("Flashcards")
    count = st.slider("Number of flashcards", min_value=5, max_value=10, value=8)
    if st.button("Generate flashcards", use_container_width=True):
        if require_papers() and require_api_key(api_key, provider):
            with st.spinner("Creating flashcards..."):
                try:
                    st.session_state.flashcards = generate_flashcards(
                        st.session_state.papers,
                        api_key,
                        count=count,
                        provider=provider,
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"Could not generate flashcards: {exc}")

    for index, card in enumerate(st.session_state.flashcards, start=1):
        with st.expander(f"Flashcard {index}: {card['question']}"):
            st.write(card["answer"])


def render_quiz(api_key: str | None, provider: str, model_name: str) -> None:
    st.subheader("Quiz")
    if st.button("Generate quiz", use_container_width=True):
        if require_papers() and require_api_key(api_key, provider):
            with st.spinner("Building quiz..."):
                try:
                    st.session_state.quiz = generate_quiz(
                        st.session_state.papers,
                        api_key,
                        provider=provider,
                        model_name=model_name,
                    )
                except Exception as exc:
                    st.error(f"Could not generate the quiz: {exc}")

    quiz = st.session_state.quiz
    if quiz.get("multiple_choice"):
        st.markdown("#### Multiple Choice")
        for index, item in enumerate(quiz["multiple_choice"], start=1):
            st.markdown(f"**{index}. {item.get('question', '')}**")
            st.radio(
                "Options",
                item.get("options", []),
                key=f"mc_{index}_{item.get('question', '')}",
                label_visibility="collapsed",
            )
            with st.expander("Show answer"):
                st.write(item.get("answer", "No answer provided."))

    if quiz.get("true_false"):
        st.markdown("#### True / False")
        for index, item in enumerate(quiz["true_false"], start=1):
            st.markdown(f"**{index}. {item.get('question', '')}**")
            st.radio(
                "Answer",
                ["True", "False"],
                key=f"tf_{index}_{item.get('question', '')}",
                horizontal=True,
                label_visibility="collapsed",
            )
            with st.expander("Show answer"):
                st.write("True" if item.get("answer") is True else "False")


def render_paper_details() -> None:
    st.subheader("Paper Details")
    if not st.session_state.papers:
        st.info("Upload and process PDFs to see paper information.")
        return

    for paper in st.session_state.papers:
        st.markdown(
            f"""
            <div class="info-card">
                <strong>{paper.file_name}</strong>
                <div class="small-muted">Pages: {paper.page_count} &middot; Words: {paper.word_count:,} &middot; Uploaded: {paper.upload_time}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    apply_styles()
    initialize_state()
    api_key, provider, model_name = render_sidebar()

    st.title("AI Research Assistant")
    st.caption("Upload research papers, ask grounded questions, and generate study materials.")

    if st.session_state.papers:
        render_paper_metrics()
    else:
        st.info("Start by uploading PDFs in the sidebar, then click Process papers.")

    chat_tab, summary_tab, flashcard_tab, quiz_tab, details_tab = st.tabs(
        ["Chat", "Summaries", "Flashcards", "Quiz", "Paper Info"]
    )
    with chat_tab:
        render_chat(api_key, provider, model_name)
    with summary_tab:
        render_summary_tools(api_key, provider, model_name)
    with flashcard_tab:
        render_flashcards(api_key, provider, model_name)
    with quiz_tab:
        render_quiz(api_key, provider, model_name)
    with details_tab:
        render_paper_details()


if __name__ == "__main__":
    main()
