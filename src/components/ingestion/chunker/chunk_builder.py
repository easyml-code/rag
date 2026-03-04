"""
chunk_builder.py — Builds standardized chunk dictionaries.

Chunk schema:
  chunk_id           : str  (SHA-256 of content)
  chunk_type         : str  ("text" | "table" | "image" | "page_ocr")
  chunk_text         : str  (searchable text — always populated)
  image              : str | None  (relative path to PNG, or null)
  page               : int
  bbox               : {x0, y0, x1, y1} in PDF points
  token_count        : int
  extraction_confidence : str  ("high" | "medium" | "low")
  doc_metadata       : dict  (doc-level info, same in all chunks)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from src.components.ingestion.chunker.utils import content_sha256, count_tokens, utc_now_iso, page_bbox


class ChunkBuilder:
    """
    Produces consistent chunk dicts for all pdf_type handlers.

    Parameters
    ----------
    doc_metadata : dict
        Document-level metadata (doc_id, source_uri, total_pages, etc.).
    config : ChunkingConfig
        The active chunking config.
    """

    def __init__(self, doc_metadata: Dict, config):
        self.doc_meta = doc_metadata
        self.cfg      = config

    # ── Text chunk ────────────────────────────────────────────────────────────

    def text_chunk(
        self,
        text: str,
        page: int,
        bbox: Dict,
        section_title: Optional[str] = None,
        confidence: str = "high",
    ) -> Optional[Dict]:
        text = (text or "").strip()
        if not text:
            return None
        return self._build(
            chunk_type  = "text",
            chunk_text  = text,
            image       = None,
            page        = page,
            bbox        = bbox,
            section_title = section_title,
            confidence  = confidence,
        )

    # ── Table chunk ───────────────────────────────────────────────────────────

    def table_chunk(
        self,
        chunk_text: str,       # markdown OR OCR text depending on strategy
        page: int,
        bbox: Dict,
        section_title: Optional[str] = None,
        confidence: str = "medium",
        image: Optional[str] = None,  # set when table_strategy=ocr_first_then_construct
        engine: Optional[str] = None,
    ) -> Optional[Dict]:
        chunk_text = (chunk_text or "").strip()
        if not chunk_text:
            return None
        c = self._build(
            chunk_type  = "table",
            chunk_text  = chunk_text,
            image       = image,
            page        = page,
            bbox        = bbox,
            section_title = section_title,
            confidence  = confidence,
        )
        if engine:
            c["extraction_engine"] = engine
        return c

    # ── Image chunk ───────────────────────────────────────────────────────────

    def image_chunk(
        self,
        ocr_text: str,
        image_path: Optional[str],
        image_width_px: int,
        image_height_px: int,
        page: int,
        bbox: Dict,
        section_title: Optional[str] = None,
        confidence: str = "medium",
    ) -> Optional[Dict]:
        """
        Chunk for a cropped visual (chart, graph, diagram, map, etc.).
        chunk_text = OCR output from the cropped image.
        image      = path to the saved PNG crop.
        """
        ocr_text = (ocr_text or "").strip()
        c = self._build(
            chunk_type  = "image",
            chunk_text  = ocr_text,
            image       = image_path,
            page        = page,
            bbox        = bbox,
            section_title = section_title,
            confidence  = confidence,
        )
        c["image_width_px"]  = image_width_px
        c["image_height_px"] = image_height_px
        return c

    # ── Page OCR chunk (PPT pages with visuals) ───────────────────────────────

    def page_ocr_chunk(
        self,
        ocr_text: str,
        image_path: Optional[str],
        image_width_px: int,
        image_height_px: int,
        page: int,
        page_width_pts: float,
        page_height_pts: float,
        section_title: Optional[str] = None,
        confidence: str = "medium",
    ) -> Optional[Dict]:
        """
        Full-page screenshot + OCR chunk for PPT pages that contain visuals.
        chunk_text = OCR(full_page_png)
        image      = path to the full-page PNG
        """
        ocr_text = (ocr_text or "").strip()
        bbox = {"x0": 0.0, "y0": 0.0,
                "x1": round(page_width_pts, 2), "y1": round(page_height_pts, 2)}
        c = self._build(
            chunk_type  = "page_ocr",
            chunk_text  = ocr_text,
            image       = image_path,
            page        = page,
            bbox        = bbox,
            section_title = section_title,
            confidence  = confidence,
        )
        c["image_width_px"]  = image_width_px
        c["image_height_px"] = image_height_px
        return c

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build(
        self,
        chunk_type: str,
        chunk_text: str,
        image: Optional[str],
        page: int,
        bbox: Dict,
        section_title: Optional[str],
        confidence: str,
    ) -> Dict[str, Any]:
        chunk_id = content_sha256(f"{page}:{chunk_type}:{chunk_text[:200]}")
        return {
            "chunk_id":              chunk_id,
            "chunk_type":            chunk_type,
            "chunk_text":            chunk_text,
            "image":                 image,
            "page":                  page,
            "bbox":                  bbox,
            "section_title":         section_title,
            "token_count":           count_tokens(chunk_text, self.cfg.tokenizer_model),
            "extraction_confidence": confidence,
            "doc_metadata":          self.doc_meta,
        }
