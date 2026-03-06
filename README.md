# RAG PDF API

FastAPI service for PDF ingestion, indexing, retrieval, and citation grounded chat.

## Overview

This project builds a document RAG system over PDF files.

- It chunks PDF content into structured units.
- It runs OCR for scanned pages and visual regions.
- It stores embeddings in Chroma.
- It stores searchable text in SQLite with FTS5 BM25.
- It serves retrieval and chat APIs on top of indexed data.

## Overall Architecture

This repository has two RAG implementations:

1. Linear RAG implementation in `src/agents/rag` exposed by `POST /chat`.
2. Agentic RAG with conversational memory in `src/agents/agentic_rag` (agent_rag) exposed by `POST /agent_chat`.

Main path of this project is the agentic RAG endpoint `POST /agent_chat`.

### Linear RAG architecture (`/chat`)

Linear RAG is a fixed sequence graph.

1. `retriever` gets vector and text candidates.
2. If `images=true`, `image_blob_loader` reads image bytes from disk.
3. `llm` generates the answer using retrieved context.
4. `citation_validation` removes invalid citations and enforces strict grounding.
5. `output` returns final answer, sources, usage, and image send stats.

### Agentic RAG architecture (`/agent_chat`)

Agentic RAG is a tool calling graph with memory.

1. `input` loads conversation history from in memory cache if not fetches from Supabase.
2. `llm_node` decides whether to call tools or answer directly for greeting messages.
3. `tool_node` executes `retrieve` tool calls (up to 5 tool calls in one run).
4. `citation_validation` validates the final answer against tool retrieved sources.
5. `save_node` writes the final turn to cache and optional Supabase.

Conversational memory behavior:

1. Recent turns are cached in process memory by `chat_id`.
2. On each request, prior turns are loaded and inserted into the graph input.
3. If Supabase is configured, turns are persisted and can be restored after restart.

If evidence is missing, both chat implementations return:

`Not found in the document.`

## Chunking Strategy

Chunking entrypoint is `PDFChunker` and supports `ppt`, `non_ppt`, and `image_only`.

Standard profile for this project documentation is `pdf_type=ppt`.
Other valid values are `non_ppt` and `image_only`.

### `ppt`

1. Processes page by page.
2. Uses page screenshot and OCR for visual heavy slides.
3. Produces `page_ocr` chunks and text chunks as available.
4. Stores full page image references for multimodal answering.

### `non_ppt`

1. Detects mixed pages with text, tables, and visual regions.
2. Extracts text chunks by configured granularity (`page`, `paragraph`, `heading`, `fixed`).
3. Extracts table chunks with `extract_then_markdown` using `pdfplumber`, `camelot`, or `cascade`.
4. Supports `ocr_first_then_construct` for OCR on cropped table images.
5. Extracts visual chunks (`chunk_type=image`) for charts and diagrams.

### `image_only`

1. Targets scanned documents with minimal text layer.
2. Runs OCR on rendered page images.
3. Produces OCR driven searchable chunks.

## Indexing and Storing Strategy

After chunking, indexing runs through `embed_and_store`.

1. Clean chunk text for embedding.
2. Create embeddings with Google embedding model.
3. Upsert vectors and lightweight metadata to Chroma.
4. Insert full text and metadata JSON to SQLite.
5. Maintain FTS5 index via triggers for BM25 retrieval.

Storage breakdown:

1. Chroma stores embeddings.
2. Chroma stores lightweight metadata (`doc_id`, `page`, `chunk_type`, `image_path`, and document fields).
3. SQLite stores full chunk text.
4. SQLite stores cleaned text indexed in FTS5.
5. SQLite stores rich metadata JSON including bbox and image dimensions.
6. Filesystem stores extracted images under `data/images/{pdf_stem}`.
7. Filesystem stores uploaded PDFs in `data/tmp`.

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

## Run

```bash
uvicorn src.app:app --reload
```

Docs:

- `http://127.0.0.1:8000/docs`

## API

Primary chat endpoint is `POST /agent_chat`.

### `POST /chunk`

`multipart/form-data`:

- required: `file` (PDF)
- optional: all chunking controls in `ChunkingConfig`

Example:

```bash
curl -X POST "http://127.0.0.1:8000/chunk" \
  -F "file=@/absolute/path/report.pdf" \
  -F "pdf_type=ppt" \
  -F "granularity=page"
```

Exact response JSON format:

```json
{
  "doc_id": "string",
  "source_file": "/absolute/path/to/saved/upload.pdf",
  "pdf_type": "ppt",
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
