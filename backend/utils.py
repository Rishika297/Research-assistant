from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


def ensure_directories() -> None:
    """Create local data directories used by the application."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries where possible."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def safe_filename(filename: str) -> str:
    """Return a filesystem-safe file name with a timestamp prefix."""
    base_name = Path(filename).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{cleaned or 'paper.pdf'}"


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def truncate_text(text: str, max_chars: int = 16000) -> str:
    """Keep prompts within a practical size for medium portfolio deployments."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\n\n[Text truncated for prompt length.]"


def get_provider_api_key(provider: str, user_key: str | None = None) -> str | None:
    """Resolve an API key from the UI input or provider-specific environment variables."""
    if provider == "Google Gemini":
        env_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    else:
        env_key = os.getenv("OPENAI_API_KEY", "").strip()

    key = (user_key or "").strip() or env_key
    return key or None


def join_texts(texts: Iterable[str], max_chars: int = 16000) -> str:
    combined = "\n\n".join(t for t in texts if t.strip())
    return truncate_text(combined, max_chars=max_chars)
