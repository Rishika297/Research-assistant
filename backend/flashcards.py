from __future__ import annotations

import json
import re
from typing import Any

from backend.chat import create_llm
from backend.pdf_processing import Paper
from backend.utils import join_texts


def _parse_json_array(raw_text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def generate_flashcards(
    papers: list[Paper],
    api_key: str,
    count: int = 8,
    provider: str = "Google Gemini",
    model_name: str = "gemini-3.5-flash",
) -> list[dict[str, str]]:
    if not papers:
        raise ValueError("Upload and process at least one paper before generating flashcards.")

    text = join_texts([paper.text for paper in papers], max_chars=16000)
    prompt = f"""
Create {count} study flashcards from the research paper text.
Return only a JSON array. Each item must have exactly these keys: "question" and "answer".
Keep answers accurate and brief.

Paper text:
{text}
"""
    llm = create_llm(api_key, provider=provider, model_name=model_name, temperature=0.3)
    raw = llm.invoke(prompt).content
    cards = _parse_json_array(raw)
    return [
        {"question": str(card.get("question", "")).strip(), "answer": str(card.get("answer", "")).strip()}
        for card in cards
        if card.get("question") and card.get("answer")
    ][:count]
