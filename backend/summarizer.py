from __future__ import annotations

from backend.chat import create_llm
from backend.pdf_processing import Paper
from backend.utils import join_texts


def generate_summary(
    papers: list[Paper],
    api_key: str,
    summary_type: str = "short",
    provider: str = "Google Gemini",
    model_name: str = "gemini-3.5-flash",
) -> str:
    if not papers:
        raise ValueError("Upload and process at least one paper before generating a summary.")

    text = join_texts([paper.text for paper in papers], max_chars=18000)
    style = (
        "Write a concise 1-2 paragraph summary for a busy reader."
        if summary_type == "short"
        else "Write a detailed summary with sections for objective, method, results, and implications."
    )
    prompt = f"""
You are summarizing research papers for a student or practitioner.
{style}

Paper text:
{text}
"""
    llm = create_llm(api_key, provider=provider, model_name=model_name, temperature=0.2)
    return llm.invoke(prompt).content.strip()


def extract_key_findings(
    papers: list[Paper],
    api_key: str,
    provider: str = "Google Gemini",
    model_name: str = "gemini-3.5-flash",
) -> str:
    if not papers:
        raise ValueError("Upload and process at least one paper before extracting findings.")

    text = join_texts([paper.text for paper in papers], max_chars=18000)
    prompt = f"""
Extract key findings from the research paper text below.
Return three Markdown sections:
- Main contributions
- Important results
- Limitations

Use concise bullet points. If limitations are not stated, say so.

Paper text:
{text}
"""
    llm = create_llm(api_key, provider=provider, model_name=model_name, temperature=0.2)
    return llm.invoke(prompt).content.strip()
