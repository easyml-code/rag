"""
app.py
------
FastAPI application with three endpoints:

  POST /chunk    — Upload a PDF, chunk it, embed + store directly to DB.
                   Returns the manifest.

  POST /retrieve — Query the indexed chunks.
                   Returns vector results and optionally text + image results.

  POST /chat     — LangGraph linear chat agent over retrieved chunks.
                   Returns citation-grounded answer + used sources metadata.

File storage:
  Uploaded PDFs are saved to {DATA_DIR}/tmp/{uuid8}_{original_name}.pdf
  Files are kept on disk intentionally — clean data/tmp/ on your own schedule.

Run:
    uvicorn src.app:app --reload
"""

import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.agentic_rag.util import flush_pending_writes, persistence_status
from src.agents.rag.graph import run_chat_agent
from src.api.agent_chat import router as agent_chat_router
from src.config import ChunkingConfig, ensure_data_dirs, settings
from src.components.ingestion.chunker.chunker import PDFChunker
from src.components.ingestion.store.indexer import embed_and_store
from src.components.retriever.retriever import retrieve
from src.log.logs import logger


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """App lifecycle: startup init + graceful shutdown flush."""
    ensure_data_dirs()
    db_state = persistence_status()
    logger.info(
        "startup ready chroma=%s db=%s images=%s tmp=%s supabase_enabled=%s reason=%s",
        settings.chroma_dir,
        Path(settings.sqlite_path).parent,
        settings.images_dir,
        settings.tmp_dir,
        db_state.get("enabled", False),
        db_state.get("reason", ""),
    )
    yield
    flushed = await flush_pending_writes(timeout_sec=5.0)
    if flushed:
        logger.info("shutdown flushed_background_db_writes=%d", flushed)


app = FastAPI(title="RAG API", version="1.0.0", lifespan=_lifespan)
app.include_router(agent_chat_router)


# ── Upload helper ─────────────────────────────────────────────────────────────

_SAFE_NAME_RE = re.compile(r"[^\w\-.]")


def _save_upload(content: bytes, original_filename: str) -> Path:
    """
    Persist an uploaded PDF to data/tmp/ with a short UUID prefix.

    Naming:  {uuid8}_{sanitised_original_name}
    Example: a3f12c89_adani_results.pdf

    The file is kept on disk after processing — clean data/tmp/ on your schedule.
    """
    safe_name = _SAFE_NAME_RE.sub("_", Path(original_filename).name)
    dest = Path(settings.tmp_dir) / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    dest.write_bytes(content)
    logger.info("upload saved  path=%s  original=%s", dest, original_filename)
    return dest

@app.post("/chunk", summary="Chunk a PDF and store to DB")
async def chunk_endpoint(
    file: UploadFile = File(..., description="PDF file to process"),

    # ChunkingConfig params — all optional, defaults match ChunkingConfig dataclass
    pdf_type:          str  = Form("non_ppt",                 description='"ppt" | "non_ppt" | "image_only"'),
    granularity:       str  = Form("page",                    description='"page" | "paragraph" | "heading" | "fixed"'),
    max_tokens:        int  = Form(1024),
    token_overlap:     int  = Form(128),
    table_engine:      str  = Form("pdfplumber",              description='"pdfplumber" | "camelot" | "cascade"'),
    table_strategy:    str  = Form("extract_then_markdown",   description='"ocr_first_then_construct" | "extract_then_markdown"'),
    ocr_engine_name:   str  = Form("doctr",                   description='"doctr" | "tesseract" | "easyocr" | "paddleocr" | "lightonocr"'),
    ocr_det_arch:      str  = Form("fast_base"),
    ocr_reco_arch:     str  = Form("parseq"),
    image_ocr:         bool = Form(True),
    screenshot_dpi:    int  = Form(150),
    visual_min_area_pts:       float = Form(3000.0),
    visual_min_aspect:         float = Form(0.15),
    visual_max_aspect:         float = Form(8.0),
    visual_min_page_fraction:  float = Form(0.01),
    min_image_width_px:        int   = Form(50),
    min_image_height_px:       int   = Form(50),
    ppt_text_only_max_drawings: int  = Form(20),
    paragraph_min_chars:       int   = Form(80),
    heading_font_size_threshold: float = Form(14.0),
    heading_bold_counts:       bool  = Form(True),
    tokenizer_model:           str   = Form("cl100k_base"),
    verbose:                   bool  = Form(False),
):
    """
    Upload a PDF. The API will:
    1. Save the PDF to data/tmp/ (persisted — not deleted after processing).
    2. Chunk it using the supplied config.
    3. Embed all chunks and store to Chroma (vectors) + SQLite (text + BM25).
    4. Save all page/element images to data/images/{pdf_stem}/.
    5. Return the manifest.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    config = ChunkingConfig(
        pdf_type                   = pdf_type,
        granularity                = granularity,
        max_tokens                 = max_tokens,
        token_overlap              = token_overlap,
        table_engine               = table_engine,
        table_strategy             = table_strategy,
        ocr_engine_name            = ocr_engine_name,
        ocr_det_arch               = ocr_det_arch,
        ocr_reco_arch              = ocr_reco_arch,
        image_ocr                  = image_ocr,
        screenshot_dpi             = screenshot_dpi,
        visual_min_area_pts        = visual_min_area_pts,
        visual_min_aspect          = visual_min_aspect,
        visual_max_aspect          = visual_max_aspect,
        visual_min_page_fraction   = visual_min_page_fraction,
        min_image_width_px         = min_image_width_px,
        min_image_height_px        = min_image_height_px,
        ppt_text_only_max_drawings = ppt_text_only_max_drawings,
        paragraph_min_chars        = paragraph_min_chars,
        heading_font_size_threshold= heading_font_size_threshold,
        heading_bold_counts        = heading_bold_counts,
        tokenizer_model            = tokenizer_model,
        verbose                    = verbose,
    )

    # Save upload to data/tmp/ — file persists after this request
    content = await file.read()
    tmp_path = _save_upload(content, file.filename)

    try:
        chunks, manifest = PDFChunker(
            str(tmp_path),
            config=config,
            original_filename=file.filename,
        ).run()
        await embed_and_store(chunks)

    except Exception as exc:
        logger.exception("chunk failed  file=%s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))

    return JSONResponse(content=manifest)


# ── /retrieve ─────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query:         str
    top_k:         int           = 5
    images:        bool          = True
    image_payload: str           = "ref"   # "ref" | "blob"
    text_search:   bool          = True


class RetrieveResponse(BaseModel):
    vector_results: List[Dict[str, Any]]
    text_results:   Optional[List[Dict[str, Any]]] = None


@app.post("/retrieve", response_model=RetrieveResponse, summary="Query indexed chunks")
async def retrieve_endpoint(body: RetrieveRequest):
    """
    Query the indexed chunks.

    - `query`         : natural language search string
    - `top_k`         : number of results per retrieval type
    - `images`        : attach image refs/blobs to results
    - `image_payload` : "ref" (path + metadata) or "blob" (raw bytes)
    - `text_search`   : also run BM25 keyword search alongside vector search
    """
    if body.image_payload not in {"ref", "blob"}:
        raise HTTPException(status_code=400, detail='image_payload must be "ref" or "blob"')

    try:
        results = await retrieve(
            user_query    = body.query,
            top_k         = body.top_k,
            images        = body.images,
            image_payload = body.image_payload,
            text_search   = body.text_search,
        )
    except Exception as exc:
        logger.exception("retrieve endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return results


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    images: bool = True
    include_text: bool = False   # only relevant when images=True; False = images-only context


class ChatResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]


@app.post("/chat", response_model=ChatResponse, summary="Chat over retrieved chunks")
async def chat_endpoint(body: ChatRequest):
    message = body.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="message is required.")
    if body.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0.")

    try:
        logger.info(
            "chat request  top_k=%s images=%s include_text=%s",
            body.top_k, body.images, body.include_text,
        )
        return await run_chat_agent(
            user_input   = message,
            top_k        = body.top_k,
            images       = body.images,
            include_text = body.include_text,
        )
    except Exception as exc:
        logger.exception("chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(exc))
