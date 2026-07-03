from __future__ import annotations

import json
import re
from typing import Any

from backend.chat import create_llm
from backend.pdf_processing import Paper
from backend.utils import join_texts


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def generate_quiz(
    papers: list[Paper],
    api_key: str,
    provider: str = "Google Gemini",
    model_name: str = "gemini-3.5-flash",
) -> dict[str, list[dict[str, Any]]]:
    if not papers:
        raise ValueError("Upload and process at least one paper before generating a quiz.")

    text = join_texts([paper.text for paper in papers], max_chars=16000)
    prompt = f"""
Create a short quiz from the research paper text.
Return only valid JSON with this shape:
{{
  "multiple_choice": [
    {{
      "question": "Question text",
      "options": ["A", "B", "C", "D"],
      "answer": "Correct option text"
    }}
  ],
  "true_false": [
    {{
      "question": "Statement text",
      "answer": true
    }}
  ]
}}

Create 4 multiple-choice questions and 3 true/false questions.

Paper text:
{text}
"""
    llm = create_llm(api_key, provider=provider, model_name=model_name, temperature=0.3)
    parsed = _parse_json_object(llm.invoke(prompt).content)
    return {
        "multiple_choice": parsed.get("multiple_choice", []) if isinstance(parsed.get("multiple_choice", []), list) else [],
        "true_false": parsed.get("true_false", []) if isinstance(parsed.get("true_false", []), list) else [],
    }
