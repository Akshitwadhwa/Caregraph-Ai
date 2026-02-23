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
5. Web UI supports text, PDF uploads, and lab image uploads

## Tech Stack
1. LangChain (LCEL + Runnables)
2. Google Gemini (`models/gemini-2.5-flash`)
3. HuggingFace Sentence-Transformers (local)
4. FAISS
5. FastAPI + Jinja2 (web UI)
6. React + Vite (modern frontend)

## Project Structure
```text
Caregraph Ai/
├── data/                # Source PDFs for ingestion
├── medical_db/          # FAISS index (default)
├── ingest.py            # PDF ingestion -> FAISS
├── caregraph.py         # Shared RAG chain (JSON output)
├── config.py            # Centralized config (models, paths, logging)
├── main.py              # CLI app (text-only)
├── vision_main.py       # Multimodal lab analysis (image + RAG)
├── web_app.py           # FastAPI app (CORS-enabled)
├── templates/           # HTML templates (Jinja2 UI)
├── static/              # CSS/JS (Jinja2 UI)
├── frontend/            # React + Vite frontend
├── requirements.txt     # Python dependencies
├── .env                 # Secrets (GOOGLE_API_KEY)
└── .env.example         # Env template
```

## Setup
1. Create `.env` (or copy the example):
```
cp .env.example .env
# then edit .env and add your key
GOOGLE_API_KEY=your_key_here
```

2. Optional: override FAISS path (default is `medical_db`):
```
FAISS_DIR=medical_db
```

3. Install dependencies (inside venv):
```
pip install -r requirements.txt
```

## Ingest PDFs
1. Put PDFs into `data/`.
2. Run ingestion:
```
python ingest.py
```

This creates the FAISS index in `medical_db` (or `FAISS_DIR` if set).

## Run CLI (Text RAG)
```
python main.py
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
uvicorn web_app:app --reload --port 8000
```
Open `http://127.0.0.1:8000`.

### React Frontend (optional)
```
cd frontend
npm install
npm run dev
```
Opens at `http://localhost:5173` and proxies `/api` to the FastAPI backend.

### Uploads (PDF + Image)
1. Use the upload form in the web UI.
2. PDF uploads are saved into `data/` and re-indexed.
3. Image uploads are analyzed via Gemini Vision and returned as JSON.

## Notes
1. The embeddings model may download on first run if not cached.
2. API keys must be valid and not leaked/disabled.
