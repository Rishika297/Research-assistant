from __future__ import annotations

from langchain_openai import ChatOpenAI

from backend.embeddings import retrieve_relevant_chunks


def create_llm(
    api_key: str,
    provider: str = "Google Gemini",
    model_name: str = "gemini-3.5-flash",
    temperature: float = 0.2,
):
    """Create a chat model for the selected provider."""
    if provider == "Google Gemini":
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=temperature,
        )

    return ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=temperature)


def answer_question(
    question: str,
    vector_store,
    api_key: str,
    provider: str = "Google Gemini",
    model_name: str = "gemini-3.5-flash",
    temperature: float = 0.2,
) -> tuple[str, list[str]]:
    if not question.strip():
        raise ValueError("Please enter a question before sending.")

    chunks = retrieve_relevant_chunks(vector_store, question)
    if not chunks:
        raise ValueError("No relevant paper text was found. Upload and process a PDF first.")

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')} | Chunk {doc.metadata.get('chunk', '?')}\n{doc.page_content}"
        for doc in chunks
    )
    sources = sorted({doc.metadata.get("source", "Unknown") for doc in chunks})

    prompt = f"""
You are an AI research assistant. Answer the user's question using only the paper excerpts below.
If the answer is not present in the excerpts, say that the uploaded papers do not provide enough information.

Paper excerpts:
{context}

Question: {question}

Answer clearly and concisely. Mention important caveats when relevant.
"""
    llm = create_llm(api_key, provider=provider, model_name=model_name, temperature=temperature)
    response = llm.invoke(prompt)
    return response.content.strip(), sources
