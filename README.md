# RAG PDF API

FastAPI service for PDF chunking, indexing, retrieval, and citation-grounded chat.

This project ingests PDF files, generates structured chunks with OCR and image support, stores embeddings in Chroma, stores searchable text in SQLite FTS5, and serves query/chat APIs on top of that data.

## What this project does

- Chunks PDF documents (`ppt`, `non_ppt`, `image_only` modes).
- Runs OCR for scanned/image-heavy content.
- Stores vectors in Chroma persistent storage.
- Stores chunk text and metadata in SQLite + FTS5 (BM25 search).
- Retrieves relevant chunks with optional image references or image blobs.
- Runs a LangGraph chat pipeline with inline source citations like `MW[1]`.

## Architecture

### Ingestion and indexing

1. `POST /chunk` receives a PDF upload.
2. File is saved to `data/tmp/`.
3. `PDFChunker` extracts chunks and image artifacts.
4. `embed_and_store` writes:
5. Vectors and lightweight metadata to Chroma.
6. Text and full metadata to SQLite + FTS5.

### Retrieval

1. `POST /retrieve` embeds the query.
2. Vector search always runs on Chroma.
3. Optional BM25 search runs on SQLite FTS5.
4. Optional image references or blobs are attached to each hit.

### Chat agent (LangGraph)

Graph path:

1. `retriever`
2. Conditional edge:
3. If `images=true` -> `image_blob_loader`
4. Else -> `llm`
5. `llm` -> `citation_validation` -> `output`

Citation rules are enforced in prompt and validator. If evidence is missing, the expected fallback is:

`Not found in the document.`

## Repository layout

```text
app.py                   # FastAPI app and API endpoints
config.py                # Runtime settings + ChunkingConfig
chunker/                 # PDF chunking pipeline
store/                   # Vector store (Chroma) + text store (SQLite/FTS)
retriever/               # Retrieval orchestration + image payload loading
agent/                   # LangGraph chat pipeline nodes and state
llm/                     # LLM client (Google Gemini via LangChain)
log/                     # Logger setup
data/                    # Local runtime data (created automatically)
```

## Requirements

- Python 3.11 recommended.
- A Google API key with access to Gemini and embedding models.
- System dependencies for PDF/OCR as needed by your chosen engines:
- `pdfplumber`, `pypdfium2`, `camelot` stack
- OCR engines such as docTR/Tesseract/EasyOCR/PaddleOCR

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy and edit environment variables:

```bash
cp .env.example .env
```

Minimum required values in `.env`:

```env
GOOGLE_API_KEY=your_google_api_key
LLM_MODEL=models/gemini-2.5-flash
LLM_MAX_TOKENS=10000
LLM_TEMPERATURE=0
```

Important optional variables:

- `EMBEDDING_MODEL` (default: `models/gemini-embedding-001`)
- `EMBED_BATCH_SIZE` (default: `100`)
- `DATA_DIR` (default: `./data`)
- `CHROMA_DIR` (default: `${DATA_DIR}/chroma`)
- `SQLITE_PATH` (default: `${DATA_DIR}/db/rag_chunks.db`)
- `IMAGES_DIR` (default: `${DATA_DIR}/images`)
- `TMP_DIR` (default: `${DATA_DIR}/tmp`)
- `CHROMA_COLLECTION` (default: `rag_chunks`)
- `HF_TOKEN` (optional, used by LightOnOCR path)

## Run the API

```bash
uvicorn app:app --reload
```

On startup, the app creates required data directories automatically.

## API reference

### 1) `POST /chunk`

Upload and index a PDF.

- Content type: `multipart/form-data`
- Required field: `file` (PDF)
- Optional fields: chunking and OCR parameters from `ChunkingConfig`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/chunk" \
  -F "file=@/absolute/path/report.pdf" \
  -F "pdf_type=non_ppt" \
  -F "granularity=page"
```

Returns a manifest with `doc_id`, chunk counts, confidence breakdown, and token stats.

### 2) `POST /retrieve`

Retrieve top-k relevant chunks.

Request body:

```json
{
  "query": "module sales in Q1-25",
  "top_k": 5,
  "images": true,
  "image_payload": "ref",
  "text_search": true
}
```

Notes:

- `image_payload="ref"` returns path/size/media-type metadata.
- `image_payload="blob"` returns raw image bytes (larger response).
- `text_search=true` adds BM25 results from SQLite FTS5.

### 3) `POST /chat`

Ask a citation-grounded question over retrieved context.

Request body:

```json
{
  "message": "What were Q1-25 module sales?",
  "top_k": 3,
  "images": true,
  "include_text": false
}
```

Response shape:

```json
{
  "answer": "Module sales for Q1-25 were 1,001 MW[1].",
  "metadata": {
    "sources": [],
    "used_citations": [1],
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    },
    "images_sent": {
      "enabled": true,
      "selected_count": 1,
      "selected_citations": [1],
      "mode": "media_bytes",
      "total_image_bytes": 123456
    }
  }
}
```

## Data storage

Default local storage layout:

```text
data/
  chroma/                 # Chroma persistent files
  db/
    rag_chunks.db         # SQLite + FTS5 tables
  images/                 # Stored page/crop images
  tmp/                    # Uploaded PDFs kept on disk
logs/
  app.log                 # Rotating application logs
```

## Logging

Logger is configured in `log/logs.py` with:

- Console logs (colorized)
- Rotating file logs (`logs/app.log`, 1 MB, 5 backups)

## Troubleshooting

### Chunks are not saved or DB is not created

- Check that startup ran and created data directories.
- Ensure `.env` has valid API credentials.
- Check `logs/app.log` for `/chunk` exceptions.
- Confirm write permissions for `DATA_DIR` and `logs/`.

### `/retrieve` returns vector results but no text results

- BM25 path may have failed. The app falls back to vector-only and logs a warning.
- Verify SQLite file exists and FTS tables were initialized by a successful `/chunk`.

### Chat is not using images

- Check `metadata.images_sent.selected_count` in `/chat` response.
- If `0` while `images=true`, image files were not loaded for the selected sources.
- Ensure `image_path` points to existing files under `data/images/`.

### `RequestsDependencyWarning` (urllib3/chardet/charset_normalizer mismatch)

- This warning comes from the installed `requests` package in your virtual environment.
- It indicates dependency version mismatch, not project logic error.
- Reinstall pinned compatible versions of `requests` dependencies in the same venv.

## Notes for contributors

- Keep sensitive values in `.env` only.
- Do not commit `data/`, `logs/`, virtualenv folders, or local cache files.
- See `.gitignore` for ignored runtime artifacts.
