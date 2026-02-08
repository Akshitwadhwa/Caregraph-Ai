# CareGraph AI: Multimodal RAG for Clinical Support

CareGraph AI is a clinical decision support assistant that grounds answers in your local clinical PDFs. It supports:
1. Text-only Q&A with retrieval-augmented generation (RAG)
2. Multimodal lab report understanding (image -> biomarkers -> RAG -> reasoning)
3. A lightweight FastAPI web UI

## Key Features
1. Multimodal analysis via Gemini Vision + RAG
2. Local embeddings using `all-MiniLM-L6-v2`
3. FAISS vector store for fast retrieval
4. Structured JSON responses with medical disclaimer, rationale, and report

## Tech Stack
1. LangChain (LCEL + Runnables)
2. Google Gemini (`models/gemini-2.5-flash`)
3. HuggingFace Sentence-Transformers (local)
4. FAISS
5. FastAPI + Jinja2 (web UI)

## Project Structure
```text
Caregraph Ai/
├── data/                # Source PDFs for ingestion
├── medical_db/          # FAISS index (default)
├── ingest.py            # PDF ingestion -> FAISS
├── caregraph.py         # Shared RAG chain (JSON output)
├── main.py              # CLI app (text-only)
├── vision_main.py       # Multimodal lab analysis (image + RAG)
├── web_app.py           # FastAPI app
├── templates/           # HTML templates
├── static/              # CSS/JS
├── .env                 # Secrets (GOOGLE_API_KEY)
└── .env.example         # Env template
```

## Setup
1. Create `.env`:
```
GOOGLE_API_KEY=your_key_here
```

2. Optional: override FAISS path (default is `medical_db`):
```
FAISS_DIR=medical_db
```

3. Install dependencies (inside venv):
```
./venv/bin/python -m pip install \\
  langchain langchain-community langchain-google-genai \\
  langchain-huggingface langchain-text-splitters \\
  faiss-cpu pypdf sentence-transformers \\
  fastapi uvicorn jinja2 python-dotenv
```

## Ingest PDFs
1. Put PDFs into `data/`.
2. Run ingestion:
```
./venv/bin/python ingest.py
```

This creates the FAISS index in `medical_db` (or `FAISS_DIR` if set).

## Run CLI (Text RAG)
```
./venv/bin/python main.py
```

Responses are JSON with fields:
```json
{
  "disclaimer": "...",
  "rationale": "...",
  "ok_report": "..."
}
```
## Run Web UI
```
./venv/bin/python -m uvicorn web_app:app --reload --port 8000
```
Open `http://127.0.0.1:8000`.

## Notes
1. The embeddings model may download on first run if not cached.
2. API keys must be valid and not leaked/disabled.
