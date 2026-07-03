from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import fitz

from backend.utils import clean_text, safe_filename, word_count


@dataclass
class Paper:
    file_name: str
    file_path: str
    text: str
    page_count: int
    word_count: int
    upload_time: str


def save_uploaded_pdf(uploaded_file: BinaryIO, upload_dir: Path) -> Path:
    """Persist a Streamlit uploaded PDF and return the saved path."""
    filename = getattr(uploaded_file, "name", "paper.pdf")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / safe_filename(filename)
    saved_path.write_bytes(uploaded_file.getbuffer())
    return saved_path


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """Extract text and page count from a PDF using PyMuPDF."""
    try:
        with fitz.open(pdf_path) as document:
            page_count = document.page_count
            text_parts = [page.get_text("text") for page in document]
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{pdf_path.name}'. The file may be invalid or encrypted.") from exc

    text = clean_text("\n\n".join(text_parts))
    if not text:
        raise ValueError(f"No readable text was found in '{pdf_path.name}'. Scanned PDFs may need OCR.")
    return text, page_count


def process_pdf(pdf_path: Path) -> Paper:
    text, page_count = extract_text_from_pdf(pdf_path)
    return Paper(
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        text=text,
        page_count=page_count,
        word_count=word_count(text),
        upload_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

