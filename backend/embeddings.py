from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from backend.pdf_processing import Paper


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOGGER = logging.getLogger(__name__)


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_papers(
    papers: Iterable[Paper],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    documents: list[Document] = []
    for paper in papers:
        chunks = splitter.split_text(paper.text)
        for index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": paper.file_name,
                        "chunk": index + 1,
                        "pages": paper.page_count,
                    },
                )
            )
    return documents


def build_vector_store(
    papers: Iterable[Paper],
    persist_dir: Path,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    save_to_disk: bool = False,
) -> FAISS:
    documents = chunk_papers(papers)
    if not documents:
        raise ValueError("No document text is available to index.")

    embeddings = get_embedding_model(embedding_model_name)
    vector_store = FAISS.from_documents(documents, embeddings)

    if save_to_disk:
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(persist_dir))
        except OSError as exc:
            LOGGER.warning("Could not save FAISS index to %s: %s", persist_dir, exc)

    return vector_store


def retrieve_relevant_chunks(vector_store: FAISS, query: str, k: int = 4) -> list[Document]:
    if not query.strip():
        return []
    return vector_store.similarity_search(query, k=k)
