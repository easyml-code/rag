"""
image_only_chunker.py — Image-only (scanned) PDF chunking strategy.

Rule: OCR every page entirely. No native text extraction. No element separation.

  For every page:
  ───────────────
  → Render full-page screenshot.
  → Run OCR on the entire page image.
  → Apply chosen granularity chunking to the OCR text.
  → No native text extraction.

Total chunks ≈ total pages (exactly equal when granularity = "page").
"""

from __future__ import annotations
import logging
from typing import List, Dict, Optional

import pdfplumber

from config import ChunkingConfig
from chunker.chunk_builder import ChunkBuilder
from chunker.screenshotter import PageScreenshotter
from chunker.ocr_engine import OCREngine
from chunker.utils import clean_text, page_bbox, split_by_tokens, count_tokens
import re

logger = logging.getLogger(__name__)


class ImageOnlyChunker:
    """
    Processes a pdfplumber.PDF using the image-only strategy.

    Parameters
    ----------
    pdf           : pdfplumber.PDF
    config        : ChunkingConfig
    builder       : ChunkBuilder
    screenshotter : PageScreenshotter
    ocr_engine    : OCREngine
    """

    def __init__(
        self,
        pdf: pdfplumber.PDF,
        config: ChunkingConfig,
        builder: ChunkBuilder,
        screenshotter: Optional[PageScreenshotter],
        ocr_engine: OCREngine,
    ):
        self.pdf    = pdf
        self.cfg    = config
        self.builder = builder
        self.ss     = screenshotter
        self.ocr    = ocr_engine

    # ── Public ────────────────────────────────────────────────────────────────

    def process_all_pages(self) -> List[Dict]:
        """Run full-page OCR on every page and return chunks."""
        total = len(self.pdf.pages)
        chunks = []
        for i, page in enumerate(self.pdf.pages):
            page_number = i + 1
            print(f'   Processing page {page_number}/{total}...', flush=True)
            chunks.extend(self._process_page(page, page_number))
        return chunks

    # ── Per-page ──────────────────────────────────────────────────────────────

    def _process_page(self, page, page_number: int) -> List[Dict]:
        """
        Render page → OCR → chunk OCR text by granularity.
        No native extraction whatsoever.
        """
        if not self.ss:
            logger.warning(f"ImageOnly p{page_number}: no screenshotter — skipping.")
            return []

        if not self.ocr.is_available():
            logger.warning(f"ImageOnly p{page_number}: OCR not available — skipping.")
            return []

        # ── Render full page ──────────────────────────────────────────────────
        try:
            pil_img = self.ss.render_page_to_pil(page_number)
        except Exception as e:
            logger.warning(f"ImageOnly p{page_number}: render failed ({e}) — skipping.")
            return []

        # ── OCR ───────────────────────────────────────────────────────────────
        try:
            ocr_text = self.ocr.extract_from_pil(pil_img) or ""
        except Exception as e:
            logger.warning(f"ImageOnly p{page_number}: OCR failed ({e}) — skipping.")
            return []

        text = clean_text(ocr_text)
        if not text.strip():
            return []

        # ── Chunk by granularity ──────────────────────────────────────────────
        return self._split_text(text, page, page_number)

    # ── Text splitting ────────────────────────────────────────────────────────

    def _split_text(self, text: str, page, page_number: int) -> List[Dict]:
        gran = self.cfg.granularity
        if gran == "page":
            return self._split_page(text, page, page_number)
        elif gran == "paragraph":
            return self._split_paragraph(text, page, page_number)
        elif gran == "heading":
            return self._split_caps_headings(text, page, page_number)
        else:  # "fixed"
            return self._split_fixed(text, page, page_number)

    def _split_page(self, text: str, page, page_number: int) -> List[Dict]:
        if count_tokens(text, self.cfg.tokenizer_model) <= self.cfg.max_tokens:
            c = self.builder.text_chunk(
                text=text, page=page_number, bbox=page_bbox(page), confidence="medium"
            )
            return [c] if c else []
        return self._split_fixed(text, page, page_number)

    def _split_paragraph(self, text: str, page, page_number: int) -> List[Dict]:
        raw_paras = re.split(r"\n{2,}", text)
        chunks, buf = [], ""
        for p in raw_paras:
            p = p.strip()
            if not p:
                continue
            if len(buf) + len(p) < self.cfg.paragraph_min_chars:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    chunks.extend(self._make_text_chunks(buf, page, page_number))
                buf = p
        if buf:
            chunks.extend(self._make_text_chunks(buf, page, page_number))
        return chunks

    def _split_caps_headings(self, text: str, page, page_number: int) -> List[Dict]:
        """Split at ALL-CAPS lines as headings (heuristic for OCR text)."""
        lines = text.split("\n")
        segs, cur_buf = [], []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            is_heading = (
                len(s) < 80 and s == s.upper() and
                len(s) > 3 and not s.replace(" ", "").isdigit()
            )
            if is_heading and cur_buf:
                segs.append("\n".join(cur_buf))
                cur_buf = [s]
            else:
                cur_buf.append(s)
        if cur_buf:
            segs.append("\n".join(cur_buf))

        chunks = []
        for seg in segs:
            chunks.extend(self._make_text_chunks(seg, page, page_number))
        return chunks

    def _split_fixed(self, text: str, page, page_number: int) -> List[Dict]:
        chunks = []
        for part in split_by_tokens(
            text, self.cfg.max_tokens, self.cfg.token_overlap, self.cfg.tokenizer_model
        ):
            c = self.builder.text_chunk(
                text=part, page=page_number, bbox=page_bbox(page), confidence="medium"
            )
            if c:
                chunks.append(c)
        return chunks

    def _make_text_chunks(self, text: str, page, page_number: int) -> List[Dict]:
        if count_tokens(text, self.cfg.tokenizer_model) > self.cfg.max_tokens:
            return self._split_fixed(text, page, page_number)
        c = self.builder.text_chunk(
            text=text, page=page_number, bbox=page_bbox(page), confidence="medium"
        )
        return [c] if c else []
