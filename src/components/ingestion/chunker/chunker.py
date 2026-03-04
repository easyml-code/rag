"""
chunker.py — Main orchestrator.

Usage:
    from src.components.ingestion.chunker.chunker import PDFChunker
    from src.config import ChunkingConfig

    config = ChunkingConfig(
        pdf_type        = "ppt",
        granularity     = "page",
        ocr_engine_name = "doctr",
        image_ocr       = True,
        screenshot_dpi  = 150,
        verbose         = True,
    )

    chunker = PDFChunker("path/to/file.pdf", config=config)
    chunks, manifest = chunker.run()
"""

from __future__ import annotations
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber

from src.config import ChunkingConfig
from src.components.ingestion.chunker.chunk_builder import ChunkBuilder
from src.components.ingestion.chunker.screenshotter import PageScreenshotter
from src.components.ingestion.chunker.ocr_engine import OCREngine
from src.components.ingestion.chunker.image_store import ImageStore
from src.components.ingestion.chunker.utils import file_sha256, file_uri, utc_now_iso

from src.components.ingestion.chunker.ppt_chunker import PPTChunker
from src.components.ingestion.chunker.non_ppt_chunker import NonPPTChunker
from src.components.ingestion.chunker.image_only_chunker import ImageOnlyChunker


class PDFChunker:
    """
    Main entry point for PDF chunking.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    config : ChunkingConfig
        Configuration object. If None, defaults are used.
    """

    def __init__(self, pdf_path: str, config: Optional[ChunkingConfig] = None, original_filename: Optional[str] = None):
        self.pdf_path = str(Path(pdf_path).resolve())
        self.cfg = config or ChunkingConfig()
        # Original filename as uploaded by the user (e.g. "report.pdf").
        # Falls back to the basename of pdf_path if not provided.
        self.original_filename: str = original_filename or Path(pdf_path).name

        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def run(self) -> Tuple[List[Dict], Dict]:
        """
        Process the PDF and return (chunks, manifest).

        Returns
        -------
        tuple
            chunks   : list of chunk dicts
            manifest : summary stats dict
        """
        self._log(f"📄  {self.pdf_path}")
        self._log(f"   pdf_type        : {self.cfg.pdf_type}")
        self._log(f"   granularity     : {self.cfg.granularity}")
        self._log(f"   table_engine    : {self.cfg.table_engine}")
        self._log(f"   table_strategy  : {self.cfg.table_strategy}")
        self._log(f"   ocr_engine      : {self.cfg.ocr_engine_name}")
        self._log(f"   image_ocr       : {self.cfg.image_ocr}")
        self._log(f"   screenshot_dpi  : {self.cfg.screenshot_dpi}")

        doc_id     = file_sha256(self.pdf_path)
        source_uri = file_uri(self.pdf_path)
        pdf_stem   = Path(self.pdf_path).stem

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            self._log(f"   pages           : {total_pages}")

            doc_meta = self._build_doc_metadata(pdf, doc_id, source_uri, total_pages)
            builder  = ChunkBuilder(doc_meta, self.cfg)

            ocr_engine = OCREngine(self.cfg)
            if ocr_engine.is_available():
                self._log(f"   OCR ready       : {self.cfg.ocr_engine_name}")
            else:
                self._log(f"⚠️  OCR unavailable : {self.cfg.ocr_engine_name}")

            screenshotter: Optional[PageScreenshotter] = None
            try:
                screenshotter = PageScreenshotter(
                    self.pdf_path,
                    dpi=self.cfg.screenshot_dpi,
                    fmt="PNG",
                )
                self._log("   Screenshotter   : ready")
            except RuntimeError as e:
                self._log(f"⚠️  Screenshotter unavailable: {e}")

            # Images always saved to settings.images_dir/{pdf_stem}/
            image_store = ImageStore(pdf_stem=pdf_stem)

            all_chunks = self._dispatch(
                pdf           = pdf,
                pdf_stem      = pdf_stem,
                builder       = builder,
                ocr_engine    = ocr_engine,
                screenshotter = screenshotter,
                image_store   = image_store,
            )

            if screenshotter:
                screenshotter.close()

        self._log(f"\n✅ Total chunks: {len(all_chunks)}")

        manifest = self._build_manifest(all_chunks, doc_id)
        return all_chunks, manifest

    # ── Dispatch by pdf_type ──────────────────────────────────────────────────

    def _dispatch(
        self,
        pdf,
        pdf_stem: str,
        builder: ChunkBuilder,
        ocr_engine: OCREngine,
        screenshotter: Optional[PageScreenshotter],
        image_store: ImageStore,
    ) -> List[Dict]:
        pdf_type = self.cfg.pdf_type

        if pdf_type == "ppt":
            self._log("\n🗂  Strategy: PPT  (1 chunk per page)")
            chunker = PPTChunker(
                pdf           = pdf,
                pdf_path      = self.pdf_path,
                config        = self.cfg,
                builder       = builder,
                screenshotter = screenshotter,
                ocr_engine    = ocr_engine,
                image_store   = image_store,
            )
        elif pdf_type == "non_ppt":
            self._log("\n🗂  Strategy: non_ppt  (multi-chunk per mixed page)")
            chunker = NonPPTChunker(
                pdf           = pdf,
                pdf_path      = self.pdf_path,
                config        = self.cfg,
                builder       = builder,
                screenshotter = screenshotter,
                ocr_engine    = ocr_engine,
                image_store   = image_store,
            )
        elif pdf_type == "image_only":
            self._log("\n🗂  Strategy: image_only  (full-page OCR per page)")
            chunker = ImageOnlyChunker(
                pdf           = pdf,
                config        = self.cfg,
                builder       = builder,
                screenshotter = screenshotter,
                ocr_engine    = ocr_engine,
            )
        else:
            raise ValueError(f"Unknown pdf_type: {self.cfg.pdf_type!r}")

        return chunker.process_all_pages()

    # ── Doc metadata ──────────────────────────────────────────────────────────

    def _build_doc_metadata(self, pdf, doc_id: str, source_uri: str, total_pages: int) -> Dict:
        return {
            "doc_id":      doc_id,
            "source_uri":  source_uri,          # file:// URI of the resolved path
            "source_file": self.pdf_path,        # full resolved path (may be temp on upload)
            "filename":    self.original_filename,  # original upload name e.g. "report.pdf"
            "total_pages": total_pages,
            "pdf_type":    self.cfg.pdf_type,
            "created_at":  utc_now_iso(),
        }

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _build_manifest(self, chunks: List[Dict], doc_id: str) -> Dict:
        type_counts = Counter(c["chunk_type"]  for c in chunks)
        page_counts = Counter(c["page"]         for c in chunks)
        conf_counts = Counter(c["extraction_confidence"] for c in chunks)
        toks        = [c["token_count"] for c in chunks if c.get("token_count")]
        return {
            "doc_id":                doc_id,
            "source_file":           self.pdf_path,
            "pdf_type":              self.cfg.pdf_type,
            "granularity":           self.cfg.granularity,
            "table_strategy":        self.cfg.table_strategy,
            "table_engine":          self.cfg.table_engine,
            "ocr_engine":            self.cfg.ocr_engine_name,
            "image_ocr":             self.cfg.image_ocr,
            "total_chunks":          len(chunks),
            "chunks_by_type":        dict(type_counts),
            "chunks_by_page":        dict(sorted(page_counts.items())),
            "extraction_confidence": dict(conf_counts),
            "avg_token_count":       round(sum(toks) / len(toks), 1) if toks else 0,
            "max_token_count":       max(toks) if toks else 0,
            "generated_at":          utc_now_iso(),
        }

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.cfg.verbose:
            print(msg, flush=True)
