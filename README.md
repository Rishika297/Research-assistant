# AI Research Assistant

A clean Streamlit application for uploading research papers, asking questions about them, and generating study materials with an LLM. The app supports Google Gemini by default, with optional OpenAI support. The project is intentionally medium-sized: modular enough to show real AI application development, but still readable for a portfolio review.

## Features

- Upload one or more PDF research papers
- Extract and clean PDF text with PyMuPDF
- Build a local FAISS vector index using Sentence Transformers
- Ask grounded questions about uploaded papers
- Generate short and detailed summaries
- Extract key findings, contributions, results, and limitations
- Generate 5-10 flashcards
- Generate multiple-choice and true/false quizzes
- View paper metadata including file name, page count, word count, and upload time
- User-friendly handling for missing API keys, empty uploads, invalid PDFs, and empty questions

## Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- OpenAI API
- Google Gemini API
- PyMuPDF
- Poetry
- Docker

## Folder Structure

```text
research-assistant/
|-- app.py
|-- backend/
|   |-- __init__.py
|   |-- chat.py
|   |-- embeddings.py
|   |-- flashcards.py
|   |-- pdf_processing.py
|   |-- quiz_generator.py
|   |-- summarizer.py
|   `-- utils.py
|-- data/
|   |-- uploads/
|   `-- vector_store/
|-- assets/
|-- Dockerfile
|-- pyproject.toml
|-- .env.example
`-- README.md
```

## Installation

1. Install Poetry if needed:

```bash
pip install poetry
```

2. Install dependencies:

```bash
poetry install
```

3. Create an environment file:

```bash
cp .env.example .env
```

4. Add your Gemini API key to `.env`:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

You can get a Gemini API key from Google AI Studio. You can also paste the API key directly in the Streamlit sidebar.

If you want to use OpenAI later, choose OpenAI in the sidebar and set:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Application

```bash
poetry run streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Docker

Build the image:

```bash
docker build -t ai-research-assistant .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env ai-research-assistant
```

## Deployment

This app can be deployed to Streamlit Community Cloud.

Recommended deployment steps:

1. Push this project to GitHub.
2. Create a Streamlit Community Cloud app from the repository.
3. Set `OPENAI_API_KEY` in Streamlit secrets.
4. For Gemini, set `GOOGLE_API_KEY` or `GEMINI_API_KEY` in Streamlit secrets.
5. Use `app.py` as the entry point.

For Streamlit secrets, you can access the key by setting it as an environment variable or by entering it in the sidebar at runtime.

## Example Screenshots

Add screenshots after running the app locally:

```text
assets/chat_screen.png
assets/summary_screen.png
assets/quiz_screen.png
```

## Future Improvements

- Add OCR support for scanned PDFs
- Add citation snippets with page numbers
- Support persistent multi-user document collections
- Export flashcards and quizzes to CSV
- Add evaluation tests for retrieval quality
- Support additional LLM providers
