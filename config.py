"""
config.py
---------
Single source of truth for all configuration.

- Settings        : runtime secrets + paths, loaded from .env
- ChunkingConfig  : dataclass controlling PDF chunking behaviour (passed per-request)
- ensure_data_dirs: call once at startup to create all required directories

Data directory layout (all under DATA_DIR, default ./data):
    data/
        chroma/         ← Chroma vector DB
        db/             ← SQLite + BM25 index
        images/         ← chunked page/element images
        tmp/            ← uploaded PDFs (persisted; clean up manually or via cron)

Usage:
    from config import settings, ChunkingConfig, ensure_data_dirs
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Runtime settings (from .env) ──────────────────────────────────────────────

class Settings(BaseSettings):
    # Google
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Embedding
    embedding_model: str = Field("models/gemini-embedding-001", env="EMBEDDING_MODEL")
    output_dimensionality: int = 1536   # 768 | 1536 | 3072
    embed_batch_size: int = Field(100, env="EMBED_BATCH_SIZE")

    # ── Data root ─────────────────────────────────────────────────────────────
    # All persistent data lives under DATA_DIR. Override in .env if needed.
    data_dir: str = Field("./data", env="DATA_DIR")

    # Sub-paths — override individually in .env only if you need a non-standard layout.
    # Defaults are derived from data_dir at startup via ensure_data_dirs().
    chroma_dir: str = Field("", env="CHROMA_DIR")
    sqlite_path: str = Field("", env="SQLITE_PATH")
    images_dir: str = Field("", env="IMAGES_DIR")
    tmp_dir: str = Field("", env="TMP_DIR")

    # LLM
    LLM_MODEL: str
    LLM_MAX_TOKENS: int
    LLM_TEMPERATURE: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _fill_default_paths(self) -> "Settings":
        """
        Fill any sub-path that was not explicitly set in .env,
        deriving it from data_dir.  This runs after all fields are loaded,
        so explicit .env overrides always win.
        """
        root = Path(self.data_dir)
        if not self.chroma_dir:
            self.chroma_dir = str(root / "chroma")
        if not self.sqlite_path:
            self.sqlite_path = str(root / "db" / "rag_chunks.db")
        if not self.images_dir:
            self.images_dir = str(root / "images")
        if not self.tmp_dir:
            self.tmp_dir = str(root / "tmp")
        return self

    # Chroma collection name — not a path, kept here for colocation
    chroma_collection: str = Field("rag_chunks", env="CHROMA_COLLECTION")


settings = Settings()


def ensure_data_dirs() -> None:
    """
    Create all required data directories if they do not exist.
    Call once at application startup.
    """
    dirs = [
        Path(settings.chroma_dir),
        Path(settings.sqlite_path).parent,   # db/
        Path(settings.images_dir),
        Path(settings.tmp_dir),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ── Per-request chunking config (dataclass, not .env) ─────────────────────────

@dataclass
class ChunkingConfig:
    """
    Controls how a PDF is chunked. Passed directly to PDFChunker.

    Parameters
    ----------
    pdf_type : str
        "ppt"        — Slide-based PDF (PowerPoint export).
        "non_ppt"    — Regular text/mixed PDF (reports, papers).
        "image_only" — Scanned / image-based PDF with no text layer.

    granularity : str
        "page"      — One text chunk per page.
        "paragraph" — Split on paragraph breaks.
        "heading"   — Split at heading boundaries.
        "fixed"     — Fixed token-size windows with overlap.

    table_engine : str
        "pdfplumber" | "camelot" | "cascade"  (non_ppt only)

    table_strategy : str
        "ocr_first_then_construct" | "extract_then_markdown"  (non_ppt only)

    ocr_engine_name : str
        "doctr" | "tesseract" | "easyocr" | "paddleocr" | "lightonocr"

    image_ocr : bool
        Run OCR on each cropped visual in non_ppt chunks.

    screenshot_dpi : int
        DPI for page rendering. Higher = better quality, slower.

    verbose : bool
        Print progress to stdout.
    """

    # ── Core ──────────────────────────────────────────────────────────────────
    pdf_type: str = "non_ppt"           # "ppt" | "non_ppt" | "image_only"

    # ── Text splitting ────────────────────────────────────────────────────────
    granularity: str = "page"           # "page" | "paragraph" | "heading" | "fixed"
    max_tokens: int = 1024
    token_overlap: int = 128

    # ── Table handling (non_ppt only) ─────────────────────────────────────────
    table_engine: str = "pdfplumber"    # "pdfplumber" | "camelot" | "cascade"
    table_strategy: str = "extract_then_markdown"
    # "ocr_first_then_construct" | "extract_then_markdown"

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_engine_name: str = "doctr"      # "doctr" | "tesseract" | "easyocr" | "paddleocr" | "lightonocr"
    ocr_det_arch: str = "fast_base"
    ocr_reco_arch: str = "parseq"
    image_ocr: bool = True

    # ── Screenshot rendering ──────────────────────────────────────────────────
    screenshot_dpi: int = 150

    # ── Visual filtering (non_ppt) ────────────────────────────────────────────
    visual_min_area_pts: float = 3000.0
    visual_min_aspect: float = 0.15
    visual_max_aspect: float = 8.0
    visual_min_page_fraction: float = 0.01
    min_image_width_px: int = 50
    min_image_height_px: int = 50

    # ── PPT text-only detection ───────────────────────────────────────────────
    ppt_text_only_max_drawings: int = 20

    # ── Paragraph / heading granularity ──────────────────────────────────────
    paragraph_min_chars: int = 80
    heading_font_size_threshold: float = 14.0
    heading_bold_counts: bool = True

    # ── Misc ──────────────────────────────────────────────────────────────────
    tokenizer_model: str = "cl100k_base"
    verbose: bool = False

    def __post_init__(self):
        valid_pdf_types = {"ppt", "non_ppt", "image_only"}
        if self.pdf_type not in valid_pdf_types:
            raise ValueError(f"pdf_type must be one of {valid_pdf_types}, got: {self.pdf_type!r}")

        valid_gran = {"page", "paragraph", "heading", "fixed"}
        if self.granularity not in valid_gran:
            raise ValueError(f"granularity must be one of {valid_gran}, got: {self.granularity!r}")

        valid_ts = {"ocr_first_then_construct", "extract_then_markdown"}
        if self.table_strategy not in valid_ts:
            raise ValueError(f"table_strategy must be one of {valid_ts}, got: {self.table_strategy!r}")