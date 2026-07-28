# Documind — Multimodal RAG Starter

Technical-document Q&A with a React UI and FastAPI backend:

- Chunking for long documents
- **PDF multimodal ingest**: text, PyMuPDF tables → Markdown, embedded images + optional Tesseract OCR
- Dense embeddings with `BAAI/bge-m3` (downloaded on first API start)
- Hybrid retrieval (`FAISS` + `BM25`)
- FastAPI API with prompt caching (in-memory index; no auth)
- React UI with optional bring-your-own LLM key

The vector index lives in process memory and resets when the API stops.

**OCR (optional):** install [Tesseract](https://github.com/tesseract-ocr/tesseract) on the API host, or set `PDF_OCR_ENABLED=false` (text and tables from PDFs still work).


## 1) Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill in an LLM key in `.env` if you want generated answers. Never commit `.env`.

**Windows PowerShell** (if script activation is blocked):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
```

Or:

```bash
uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
```

First boot downloads the embedding model and can take a few minutes.

### Web UI

1. Start the API on port **8000**.
2. `cd web` → `npm install` → `npm run dev` → **http://127.0.0.1:5173**.
3. Dev proxy: `/api/*` → `http://127.0.0.1:8000`.
4. Settings: optional backend URL, LLM base URL, model id, and `chat_completions` vs `responses`.

BYOK: browser session storage → API header **`X-LLM-Api-Key`**, plus optional body fields **`llm_base_url`**, **`llm_model`**, **`llm_route`**.

## 2) API endpoints

Interactive docs: **http://127.0.0.1:8000/docs**

- `GET /health`
- `GET /index/documents` / `POST /index/clear`
- `POST /ingest` — JSON documents
- `POST /ingest/pdf` — multipart PDF upload
- `POST /query` — retrieve (+ optional generation)

If no chat API key is configured (`LLM_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY`), the API returns retrieval context fallback text.

Responses include a simple `fast` / `deep` **route label** (placeholder for future model routing). Prompt responses are cached in memory with a TTL.

**Groq (not “Grok”):**

- `GROQ_API_KEY=<your key>` (or `LLM_API_KEY`)
- `LLM_BASE_URL=https://api.groq.com/openai/v1`
- `LLM_ROUTE=responses`
- `OPENAI_MODEL=openai/gpt-oss-20b`

**xAI Grok:**

- `LLM_BASE_URL=https://api.x.ai/v1`
- `LLM_API_KEY=<your xAI API key>`
- `OPENAI_MODEL=<exact model id from the xAI console>`

## 3) Projected next milestones

1. Persistent FAISS index + metadata DB (important for hosted demos).
2. Reranker on top of hybrid retrieval.
3. Tracing / eval.
4. Real model routing from the `fast` / `deep` label.

## 4) Publishing checklist

- Keep `.env` out of git. Rotate any keys that ever lived in a local `.env`.
- Commit only `.env.example` (empty keys).
- On Vercel: set `VITE_API_URL`. On the API host: set secrets + `CORS_ORIGINS` including your Vercel domain.
