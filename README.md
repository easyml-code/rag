# RAG PDF API

FastAPI service for PDF ingestion, indexing, retrieval, and citation grounded chat.

## Overview

This project builds a document RAG system over PDF files.

- It chunks PDF content into structured units.
- It runs OCR for scanned pages and visual regions.
- It stores embeddings in Chroma.
- It stores searchable text in SQLite with FTS5 BM25.
- It serves retrieval and chat endpoints on top of indexed data.
- It supports two chat paths:
- `POST /chat`: fixed LangGraph RAG flow.
- `POST /agent_chat`: tool calling agent with chat memory and optional Supabase persistence.

## End to End Flow

### 1) Ingestion and indexing

`POST /chunk` runs:

1. Save uploaded PDF to `data/tmp`.
2. Chunk with `PDFChunker` (`ppt`, `non_ppt`, `image_only`).
3. Save extracted images to `data/images/{pdf_stem}`.
4. Clean and embed chunk text.
5. Store vectors in Chroma.
6. Store text and metadata in SQLite + FTS5.
7. Return indexing manifest JSON.

### 2) Retrieval

`POST /retrieve` runs:

1. Embed query.
2. Vector search in Chroma.
3. Optional BM25 text search in SQLite FTS5.
4. Optional image payload attachment (`ref` or `blob`).
5. Return `vector_results` and optional `text_results`.

### 3) Agent layers

`POST /chat` graph:

1. `retriever`
2. conditional route:
3. if `images=true`: `image_blob_loader`
4. then `llm`
5. `citation_validation`
6. `output`

`POST /agent_chat` graph:

1. `input` (history load from memory and optional DB)
2. `llm_node` (tool calling)
3. `tool_node` (`retrieve` tool, max 5 calls)
4. `citation_validation`
5. `save_node` (response assembly and optional DB write)

If evidence is not available, both chat flows can return:

`Not found in the document.`

## Storage

### Local storage

Default paths under `DATA_DIR` (`./data` by default):

```text
data/
  chroma/                 # Chroma vector store
  db/
    rag_chunks.db         # SQLite tables + FTS5 index
  images/                 # Extracted page and crop images
  tmp/                    # Uploaded PDFs kept on disk
logs/
  app.log                 # Rotating logs
```

### What is stored where

- Chroma stores vectors and lightweight metadata per chunk.
- SQLite stores full chunk text and JSON metadata for BM25 search.
- Image files are saved on disk and linked by relative `image_path`.
- Chat memory for `agent_chat` is cached in process memory.
- If Supabase is configured and available, chat turns are also persisted to DB.

## Project Structure

```text
src/
  app.py                              # FastAPI app and core endpoints
  api/agent_chat.py                   # /agent_chat router
  config.py                           # Settings and ChunkingConfig
  components/
    ingestion/
      chunker/                        # PDF chunking, OCR, image capture
      store/                          # Chroma and SQLite write/read layers
    retriever/                        # Retrieval orchestration
    utils/embeddings.py               # Embedding helpers
  agents/
    rag/                              # Fixed graph chat flow for /chat
    agentic_rag/                      # Tool calling chat flow for /agent_chat
  llm/llm.py                          # Gemini client
  log/logs.py                         # Logger config
```

## Requirements

- Python 3.11 recommended.
- Google API key for LLM and embedding models.
- PDF and OCR dependencies from `requirements.txt`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Minimum required `.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
LLM_MODEL=models/gemini-2.5-flash
LLM_MAX_TOKENS=10000
LLM_TEMPERATURE=0
```

Optional `.env` keys:

- `EMBEDDING_MODEL`
- `EMBED_BATCH_SIZE`
- `DATA_DIR`
- `CHROMA_DIR`
- `SQLITE_PATH`
- `IMAGES_DIR`
- `TMP_DIR`
- `CHROMA_COLLECTION`
- `SUPABASE_URL`
- `SUPABASE_KEY`

## Run

```bash
uvicorn src.app:app --reload
```

Docs:

- `http://127.0.0.1:8000/docs`

## API

### `POST /chunk`

`multipart/form-data`:

- required: `file` (PDF)
- optional: all chunking controls in `ChunkingConfig`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/chunk" \
  -F "file=@/absolute/path/report.pdf" \
  -F "pdf_type=non_ppt" \
  -F "granularity=page"
```

Exact response JSON format:

```json
{
  "doc_id": "string",
  "source_file": "/absolute/path/to/saved/upload.pdf",
  "pdf_type": "non_ppt",
  "granularity": "page",
  "table_strategy": "extract_then_markdown",
  "table_engine": "pdfplumber",
  "ocr_engine": "doctr",
  "image_ocr": true,
  "total_chunks": 0,
  "chunks_by_type": {},
  "chunks_by_page": {},
  "extraction_confidence": {},
  "avg_token_count": 0,
  "max_token_count": 0,
  "generated_at": "2026-03-07T00:00:00+00:00"
}
```

### `POST /retrieve`

Request JSON:

```json
{
  "query": "string",
  "top_k": 5,
  "images": true,
  "image_payload": "ref",
  "text_search": true
}
```

Exact response JSON key format when `images=true` and `text_search=true`:

```json
{
  "vector_results": [
    {
      "chunk_id": "string",
      "text": "string",
      "score": 0.0,
      "doc_id": "string",
      "page": 0,
      "chunk_type": "text",
      "image_path": "pdf_stem/p01_page_ocr_00.webp",
      "section_title": "string",
      "token_count": 0,
      "extraction_confidence": "high",
      "image_width_px": 0,
      "image_height_px": 0,
      "bbox": {
        "x0": 0.0,
        "y0": 0.0,
        "x1": 0.0,
        "y1": 0.0
      },
      "source_uri": "file:///abs/path/file.pdf",
      "source_file": "/abs/path/file.pdf",
      "filename": "original_upload.pdf",
      "total_pages": 0,
      "pdf_type": "non_ppt",
      "created_at": "2026-03-07T00:00:00+00:00",
      "doc_metadata": {},
      "retrieval_type": "vector",
      "images": {
        "page_image": {
          "key": "pdf_stem/p01_page_ocr_00.webp",
          "path": "/abs/path/data/images/pdf_stem/p01_page_ocr_00.webp",
          "size_bytes": 0,
          "media_type": "image/webp"
        },
        "inline_images": [
          {
            "name": "image_0.webp",
            "key": "pdf_stem/image_0.webp",
            "path": "/abs/path/data/images/pdf_stem/image_0.webp",
            "size_bytes": 0,
            "media_type": "image/webp"
          }
        ]
      }
    }
  ],
  "text_results": [
    {
      "chunk_id": "string",
      "text": "string",
      "score": 0.0,
      "doc_id": "string",
      "page": 0,
      "chunk_type": "text",
      "image_path": "pdf_stem/p01_page_ocr_00.webp",
      "section_title": "string",
      "token_count": 0,
      "extraction_confidence": "high",
      "bbox": {
        "x0": 0.0,
        "y0": 0.0,
        "x1": 0.0,
        "y1": 0.0
      },
      "image_width_px": 0,
      "image_height_px": 0,
      "source_uri": "file:///abs/path/file.pdf",
      "source_file": "/abs/path/file.pdf",
      "total_pages": 0,
      "pdf_type": "non_ppt",
      "created_at": "2026-03-07T00:00:00+00:00",
      "doc_metadata": {},
      "retrieval_type": "text",
      "images": {
        "page_image": {
          "key": "pdf_stem/p01_page_ocr_00.webp",
          "path": "/abs/path/data/images/pdf_stem/p01_page_ocr_00.webp",
          "size_bytes": 0,
          "media_type": "image/webp"
        },
        "inline_images": []
      }
    }
  ]
}
```

Notes:

- If `images=false`, `images` key is not attached to results.
- If `text_search=false`, output may contain only `vector_results`.
- If `image_payload="blob"`, each image object includes `"blob"` bytes.

### `POST /chat`

Request JSON:

```json
{
  "message": "string",
  "top_k": 3,
  "images": true,
  "include_text": false
}
```

Exact response JSON format:

```json
{
  "answer": "string",
  "metadata": {
    "sources": [
      {
        "citation": 1,
        "source_file": "/abs/path/file.pdf",
        "filename": "original_upload.pdf",
        "page": 0,
        "chunk_type": "text",
        "chunk_id": "string",
        "bbox": {
          "x0": 0.0,
          "y0": 0.0,
          "x1": 0.0,
          "y1": 0.0
        },
        "text": "string"
      }
    ],
    "used_citations": [1],
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0
    },
    "images_sent": {
      "enabled": true,
      "selected_count": 0,
      "selected_citations": [],
      "mode": "media_bytes",
      "total_image_bytes": 0
    }
  }
}
```

### `POST /agent_chat`

Request JSON:

```json
{
  "message": "string",
  "top_k": 3,
  "images": true,
  "include_text": false,
  "text_search": true,
  "config": {
    "chat_id": "string",
    "session_id": "string"
  }
}
```

Exact response JSON format:

```json
{
  "answer": "string",
  "metadata": {
    "history_turns_loaded": 0,
    "sources": [
      {
        "citation": 1,
        "source_file": "/abs/path/file.pdf",
        "filename": "original_upload.pdf",
        "page": 0,
        "chunk_type": "text",
        "chunk_id": "string",
        "bbox": {
          "x0": 0.0,
          "y0": 0.0,
          "x1": 0.0,
          "y1": 0.0
        },
        "text": "string"
      }
    ],
    "used_citations": [1],
    "rephrased_queries": ["string"],
    "citation_validation": {
      "is_valid": true,
      "issues": [],
      "available_citations": [1],
      "found_citations": [1],
      "invalid_citations": []
    },
    "usage_metadata": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0,
      "llm_calls": 0
    },
    "images_sent": {
      "enabled": true,
      "selected_count": 1,
      "selected_citations": [1],
      "mode": "tool_message_image_blocks",
      "total_image_bytes": 12345
    }
  }
}
```

Optional `usage_metadata` keys that may appear:

- `internal_tokens`
- `final_output_tokens`
- `llm_calls_with_usage`
- `input_token_details`
- `output_token_details`
- `per_call`
- `aggregated_input_token_details`
- `aggregated_output_token_details`

## Notes

- Uploaded PDFs in `data/tmp` are intentionally kept on disk.
- Clean `data/tmp` and other runtime data based on your retention policy.
- For `agent_chat` DB persistence, install `supabase` and set `SUPABASE_URL` and `SUPABASE_KEY`.
