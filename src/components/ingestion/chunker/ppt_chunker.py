"""
ppt_chunker.py — PPT PDF chunking strategy.

Rule: exactly 1 chunk per page.

  Page has ONLY text elements
  ────────────────────────────
  → Extract native text (no OCR).
  → chunk_type = "text"
  → chunk_text = extracted_text
  → image      = null

  Page has text + ANY other element (table / image / chart / shape / screenshot)
  ────────────────────────────────────────────────────────────────────────────────
  → Take full-page screenshot (PNG).
  → Run OCR on the full-page screenshot.
  → chunk_type = "page_ocr"
  → chunk_text = OCR(full_page_png)
  → image      = full_page_png

Total chunks == total pages.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional

import pdfplumber

from src.config import ChunkingConfig
from src.components.ingestion.chunker.chunk_builder import ChunkBuilder
from src.components.ingestion.chunker.screenshotter import PageScreenshotter
from src.components.ingestion.chunker.ocr_engine import OCREngine
from src.components.ingestion.chunker.image_store import ImageStore
from src.components.ingestion.chunker.utils import clean_text, page_bbox

logger = logging.getLogger(__name__)


class PPTChunker:
    """
    Processes a pdfplumber.PDF using the PPT strategy.

    Parameters
    ----------
    pdf         : pdfplumber.PDF
    pdf_path    : str
    config      : ChunkingConfig
    builder     : ChunkBuilder
    screenshotter : PageScreenshotter
    ocr_engine  : OCREngine
    image_store : ImageStore | None  (None → images not saved to disk)
    """

    def __init__(
        self,
        pdf: pdfplumber.PDF,
        pdf_path: str,
        config: ChunkingConfig,
        builder: ChunkBuilder,
        screenshotter: Optional[PageScreenshotter],
        ocr_engine: OCREngine,
        image_store: Optional[ImageStore],
    ):
        self.pdf         = pdf
        self.pdf_path    = pdf_path
        self.cfg         = config
        self.builder     = builder
        self.ss          = screenshotter
        self.ocr         = ocr_engine
        self.img_store   = image_store

    # ── Public ────────────────────────────────────────────────────────────────

    def process_all_pages(self) -> List[Dict]:
        """Process every page and return exactly len(pages) chunks."""
        total = len(self.pdf.pages)
        chunks = []
        for i, page in enumerate(self.pdf.pages):
            page_number = i + 1
            print(f"   Processing page {page_number}/{total}…", flush=True)
            chunk = self._process_page(page, page_number)
            if chunk:
                chunks.append(chunk)
        return chunks

    # ── Per-page dispatch ─────────────────────────────────────────────────────

    def _process_page(self, page, page_number: int) -> Optional[Dict]:
        if self._is_text_only(page):
            return self._text_only_chunk(page, page_number)
        else:
            return self._page_ocr_chunk(page, page_number)

    # ── Text-only detection ───────────────────────────────────────────────────

    def _is_text_only(self, page) -> bool:
        """
        A PPT page is text-only when ALL of the following hold:
          1. No embedded raster images.
          2. No pdfplumber-detected tables.
          3. Vector drawing count (rects + curves + lines) < threshold.
             (slide borders / decorative frames are usually < 20 elements)
        """
        # 1. Raster images
        if page.images:
            return False

        # 2. Digital tables
        try:
            tables = page.find_tables() or []
            if tables:
                return False
        except Exception:
            pass

        # 3. Vector drawing count
        drawing_count = (
            len(page.curves or []) +
            len(page.rects  or []) +
            len(page.lines  or [])
        )
        if drawing_count >= self.cfg.ppt_text_only_max_drawings:
            return False

        return True

    # ── Text-only page ────────────────────────────────────────────────────────

    def _text_only_chunk(self, page, page_number: int) -> Optional[Dict]:
        """
        Extract native text — NO OCR.
        chunk_type = "text", image = null
        """
        text = clean_text(page.extract_text() or "")
        if not text or len(text) < 3:
            logger.debug(f"PPT p{page_number}: text-only but empty text — skipping.")
            return None

        return self.builder.text_chunk(
            text       = text,
            page       = page_number,
            bbox       = page_bbox(page),
            confidence = "high",
        )

    # ── Page with visuals → full-page OCR ─────────────────────────────────────

    def _page_ocr_chunk(self, page, page_number: int) -> Optional[Dict]:
        """
        Full-page screenshot → OCR → page_ocr chunk.
        Falls back to native text if screenshot or OCR fails.
        chunk_type = "page_ocr", image = full_page_png
        """
        # ── Render full page ──────────────────────────────────────────────────
        if not self.ss:
            logger.warning(f"PPT p{page_number}: no screenshotter — falling back to native text.")
            return self._text_only_chunk(page, page_number)

        try:
            pil_img = self.ss.render_page_to_pil(page_number)
        except Exception as e:
            logger.warning(f"PPT p{page_number}: render failed ({e}) — falling back to native text.")
            return self._text_only_chunk(page, page_number)

        # ── Save PNG ──────────────────────────────────────────────────────────
        rel_path: Optional[str] = None
        if self.img_store:
            try:
                rel_path = self.img_store.save(pil_img, page_number, label="page_ocr")
            except Exception as e:
                logger.warning(f"PPT p{page_number}: image save failed ({e}).")

        # ── OCR the screenshot ────────────────────────────────────────────────
        ocr_text = ""
        if self.ocr.is_available():
            try:
                ocr_text = self.ocr.extract_from_pil(pil_img) or ""
            except Exception as e:
                logger.warning(f"PPT p{page_number}: OCR failed ({e}).")

        # Fallback: if OCR empty, use native PDF text
        if not ocr_text.strip():
            ocr_text = clean_text(page.extract_text() or "")

        w, h = pil_img.size
        return self.builder.page_ocr_chunk(
            ocr_text        = ocr_text,
            image_path      = rel_path,
            image_width_px  = w,
            image_height_px = h,
            page            = page_number,
            page_width_pts  = float(page.width),
            page_height_pts = float(page.height),
            confidence      = "medium",
        )
