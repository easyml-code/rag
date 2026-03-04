"""
non_ppt_chunker.py — Regular PDF (non-PPT) chunking strategy.

Per-page logic:
────────────────────────────────────────────────────────────────────────────────
Case A — Text-only page (no images, no tables detected)
  → Extract native text.
  → Chunk using chosen granularity strategy.
  → No OCR.

Case B — Mixed page (text + tables + images/graphs/diagrams)
  Text:
    → Extract native text (excluding table regions) → chunk normally.
  Tables:
    Strategy = "ocr_first_then_construct":
      → Crop the table region from the page screenshot.
      → Run OCR on the table PNG.
      → chunk_type = "table", chunk_text = OCR(table_png), image = table_png
    Strategy = "extract_then_markdown":
      → Use configured table_engine (pdfplumber / camelot / cascade).
      → Convert to Markdown.
      → chunk_type = "table", chunk_text = markdown, image = null
  Images / Graphs / Diagrams / Maps (filtered):
    → Keep: charts, graphs, maps, diagrams, block diagrams, tables-as-images.
    → Discard: logos, icons, decorative small images, background graphics.
    → Crop each qualifying visual.
    → Run OCR on each crop.
    → chunk_type = "image", chunk_text = OCR(crop), image = crop_png
────────────────────────────────────────────────────────────────────────────────

A page with 1 text block + 1 table + 1 image produces 3 chunks.
"""

from __future__ import annotations
import io
import logging
import re
from typing import List, Dict, Optional, Tuple

import pdfplumber

from src.config import ChunkingConfig
from src.components.ingestion.chunker.chunk_builder import ChunkBuilder
from src.components.ingestion.chunker.screenshotter import PageScreenshotter
from src.components.ingestion.chunker.ocr_engine import OCREngine
from src.components.ingestion.chunker.image_store import ImageStore
from src.components.ingestion.chunker.utils import (
    clean_text, page_bbox, bbox_dict,
    split_by_tokens, count_tokens,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Visual filtering constants — what to keep vs discard
# ─────────────────────────────────────────────────────────────────────────────
# Keep:    Charts, Graphs, Maps, Tables-as-images, Diagrams, Block diagrams
# Discard: Logos, Icons, Decorative small images, Background graphics
# Using:   Minimum area threshold, Aspect ratio filtering, Page-fraction heuristics


class NonPPTChunker:
    """
    Processes a pdfplumber.PDF using the non-PPT (regular document) strategy.

    Parameters
    ----------
    pdf           : pdfplumber.PDF
    pdf_path      : str
    config        : ChunkingConfig
    builder       : ChunkBuilder
    screenshotter : PageScreenshotter | None
    ocr_engine    : OCREngine
    image_store   : ImageStore | None
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
        self.pdf       = pdf
        self.pdf_path  = pdf_path
        self.cfg       = config
        self.builder   = builder
        self.ss        = screenshotter
        self.ocr       = ocr_engine
        self.img_store = image_store

        # Lazy-load table engine (depends on config)
        self._table_engine = None

        # Per-page caches — cleared after each page to avoid holding all pages in memory
        self._render_cache: dict = {}   # page_number → PIL Image
        self._tables_cache: dict = {}   # page_number → list[Table]

    # ── Public ────────────────────────────────────────────────────────────────

    def process_all_pages(self) -> List[Dict]:
        total = len(self.pdf.pages)
        chunks = []
        for i, page in enumerate(self.pdf.pages):
            page_number = i + 1
            print(f'   Processing page {page_number}/{total}...', flush=True)
            chunks.extend(self._process_page(page, page_number))
            # Free cached render + table data — no need to hold all pages in memory
            self._render_cache.pop(page_number, None)
            self._tables_cache.pop(page_number, None)
        return chunks

    # ── Per-page dispatch ─────────────────────────────────────────────────────

    def _get_page_render(self, page_number: int):
        """
        Render a page to PIL exactly once per page, cache the result.
        All callers within the same page share the same render.
        """
        if page_number not in self._render_cache:
            if self.ss is None:
                return None
            try:
                self._render_cache[page_number] = self.ss.render_page_to_pil(page_number)
            except Exception as e:
                logger.warning("page render failed p%d: %s", page_number, e)
                self._render_cache[page_number] = None
        return self._render_cache[page_number]

    def _get_page_tables(self, page, page_number: int):
        """
        Call pdfplumber.find_tables() exactly once per page, cache the result.
        Avoids the double-call between _detect_tables() and _extract_pdfplumber().
        """
        if page_number not in self._tables_cache:
            try:
                self._tables_cache[page_number] = page.find_tables() or []
            except Exception:
                self._tables_cache[page_number] = []
        return self._tables_cache[page_number]

    def _process_page(self, page, page_number: int) -> List[Dict]:
        """
        Route to Case A (text-only) or Case B (mixed).
        Detection: page has images OR detectable tables → mixed.
        """
        has_images = bool(page.images)
        has_tables = self._detect_tables(page, page_number)

        if not has_images and not has_tables:
            return self._case_a_text_only(page, page_number)
        else:
            return self._case_b_mixed(page, page_number)

    # ─────────────────────────────────────────────────────────────────────────
    #  Case A — Text-only page
    # ─────────────────────────────────────────────────────────────────────────

    def _case_a_text_only(self, page, page_number: int) -> List[Dict]:
        """Extract native text, chunk by granularity. No OCR."""
        text = clean_text(page.extract_text() or "")
        if not text.strip():
            return []
        words = page.extract_words() or []
        bbox  = self._words_bbox(words) if words else page_bbox(page)
        return self._split_text(text, page, page_number, bbox)

    # ─────────────────────────────────────────────────────────────────────────
    #  Case B — Mixed page
    # ─────────────────────────────────────────────────────────────────────────

    def _case_b_mixed(self, page, page_number: int) -> List[Dict]:
        """
        Three passes:
          1. Tables   → table chunks
          2. Text     → text chunks (excluding table regions)
          3. Visuals  → image chunks (filtered, OCR'd)
        """
        chunks: List[Dict] = []

        # ── Pass 1: Tables ────────────────────────────────────────────────────
        table_chunks, table_bboxes = self._extract_tables(page, page_number)
        chunks.extend(table_chunks)

        # ── Pass 2: Text (excluding table regions) ────────────────────────────
        raw_text, outside_words = self._text_outside_bboxes(page, table_bboxes)
        text = clean_text(raw_text)
        if text.strip():
            bbox = self._words_bbox(outside_words) if outside_words else page_bbox(page)
            chunks.extend(self._split_text(text, page, page_number, bbox))

        # ── Pass 3: Filtered visuals ──────────────────────────────────────────
        if self.img_store:
            chunks.extend(self._extract_visuals(page, page_number, table_bboxes))

        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    #  Table extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_tables(self, page, page_number: int) -> bool:
        """Check if page has any tables — uses cached find_tables() result."""
        return bool(self._get_page_tables(page, page_number))

    def _extract_tables(
        self, page, page_number: int
    ) -> Tuple[List[Dict], List[Tuple[float, float, float, float]]]:
        """
        Returns (list_of_table_chunks, list_of_table_bboxes).
        Dispatches on config.table_strategy.
        """
        table_chunks: List[Dict] = []
        table_bboxes: List[Tuple] = []

        strategy = self.cfg.table_strategy

        if strategy == "ocr_first_then_construct":
            table_chunks, table_bboxes = self._tables_via_ocr(page, page_number)
        else:
            # "extract_then_markdown"
            table_chunks, table_bboxes = self._tables_via_engine(page, page_number)

        return table_chunks, table_bboxes

    # ── Strategy: OCR → chunk_text = OCR text, image = table_png ─────────────

    def _tables_via_ocr(
        self, page, page_number: int
    ) -> Tuple[List[Dict], List[Tuple]]:
        """
        For each detected table:
          1. Crop the table region from a full-page render.
          2. Run OCR on the table PNG.
          3. Create chunk_type="table", chunk_text=OCR(table_png), image=table_png.
        """
        chunks, bboxes = [], []

        try:
            raw_tables = self._get_page_tables(page, page_number)
        except Exception:
            return [], []

        if not raw_tables:
            return [], []

        # Render full page once — shared by all tables on this page via cache
        pil_full = self._get_page_render(page_number)

        for t_idx, table in enumerate(raw_tables):
            bbox = table.bbox  # (x0, top, x1, bottom) in PDF pts
            bboxes.append(bbox)

            # Crop table region from rendered page
            table_pil = None
            if pil_full is not None:
                table_pil = self._crop_region(pil_full, page, bbox)

            # OCR the cropped table
            ocr_text = ""
            if table_pil is not None and self.ocr.is_available():
                try:
                    ocr_text = self.ocr.extract_from_pil(table_pil) or ""
                except Exception as e:
                    logger.warning(f"NonPPT p{page_number} table OCR failed: {e}")

            # Fallback: extract native table text
            if not ocr_text.strip():
                try:
                    rows     = table.extract()
                    ocr_text = self._rows_to_plain(rows)
                except Exception:
                    pass

            # Save table PNG
            rel_path: Optional[str] = None
            if table_pil is not None and self.img_store:
                try:
                    rel_path = self.img_store.save(table_pil, page_number, label="table")
                except Exception as e:
                    logger.warning(f"NonPPT p{page_number}: table image save failed ({e})")

            x0, top, x1, bottom = bbox
            chunk = self.builder.table_chunk(
                chunk_text = clean_text(ocr_text),
                page       = page_number,
                bbox       = bbox_dict(x0, top, x1, bottom),
                confidence = "medium",
                image      = rel_path,
                engine     = "ocr_first_then_construct",
            )
            if chunk:
                chunks.append(chunk)

        return chunks, bboxes

    # ── Strategy: table engine → Markdown ─────────────────────────────────────

    def _tables_via_engine(
        self, page, page_number: int
    ) -> Tuple[List[Dict], List[Tuple]]:
        """
        Use the configured table_engine (pdfplumber / camelot / cascade).
        Convert to Markdown. chunk_text = markdown, image = null.
        """
        chunks, bboxes = [], []

        if self.cfg.table_engine == "camelot":
            results = self._extract_camelot(page_number)
        else:
            # pdfplumber or cascade (try pdfplumber first, then camelot)
            results = self._extract_pdfplumber(page, page_number)
            if not results and self.cfg.table_engine == "cascade":
                results = self._extract_camelot(page_number)

        for md, bbox, engine, accuracy in results:
            if not md.strip():
                continue
            bboxes.append(bbox)
            conf = "high" if accuracy >= 0.85 else "medium"
            x0, top, x1, bottom = bbox
            chunk = self.builder.table_chunk(
                chunk_text = md,
                page       = page_number,
                bbox       = bbox_dict(x0, top, x1, bottom),
                confidence = conf,
                image      = None,
                engine     = engine,
            )
            if chunk:
                chunks.append(chunk)

        return chunks, bboxes

    def _extract_pdfplumber(
        self, page, page_number: int
    ) -> List[Tuple[str, Tuple, str, float]]:
        """Returns list of (markdown, bbox, engine, accuracy)."""
        results = []
        try:
            raw_tables = self._get_page_tables(page, page_number)  # cached — no second find_tables()
            for table in raw_tables:
                bbox = table.bbox
                rows = table.extract()
                md   = self._rows_to_markdown(rows)
                if md.strip():
                    results.append((md, bbox, "pdfplumber", 0.8))
        except Exception as e:
            logger.warning(f"pdfplumber table extract p{page_number}: {e}")
        return results

    def _extract_camelot(
        self, page_number: int
    ) -> List[Tuple[str, Tuple, str, float]]:
        """Returns list of (markdown, bbox, engine, accuracy)."""
        try:
            import camelot
        except ImportError:
            logger.warning("camelot not installed: pip install camelot-py[cv]")
            return []

        results = []
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(
                    self.pdf_path,
                    pages   = str(page_number),
                    flavor  = flavor,
                    suppress_stdout = True,
                )
                for t in tables:
                    df   = t.df
                    md   = self._df_to_markdown(df)
                    bbox = (
                        t._bbox[0], t._bbox[1],   # x0, top
                        t._bbox[2], t._bbox[3],   # x1, bottom
                    )
                    results.append((md, bbox, f"camelot_{flavor}", t.accuracy / 100.0))
                if results:
                    break  # lattice worked, don't try stream
            except Exception as e:
                logger.debug(f"Camelot {flavor} p{page_number}: {e}")
        return results

    # ─────────────────────────────────────────────────────────────────────────
    #  Visual extraction (images / charts / graphs / diagrams)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_visuals(
        self,
        page,
        page_number: int,
        table_bboxes: List[Tuple],
    ) -> List[Dict]:
        """
        Find embedded raster images, filter them, crop, OCR, save.
        Returns image chunks.
        """
        chunks: List[Dict] = []
        seen   = set()
        img_idx = 0

        for img in (page.images or []):
            # Dedup by position
            pos_key = (round(img.get("x0", 0)), round(img.get("top", 0)))
            if pos_key in seen:
                continue
            seen.add(pos_key)

            # ── Filter: keep only meaningful visuals ──────────────────────────
            if not self._is_visual_meaningful(img, page):
                continue

            # ── Filter: skip if overlaps a detected table ─────────────────────
            if self._overlaps_bbox(img, table_bboxes):
                continue

            # ── Crop from rendered page ───────────────────────────────────────
            pil_crop = self._crop_image_region(page_number, img, page)
            if pil_crop is None:
                continue

            img_bbox = bbox_dict(
                img["x0"], img["top"], img["x1"], img["bottom"]
            )

            # ── OCR the crop ──────────────────────────────────────────────────
            ocr_text = ""
            if self.ocr.is_available() and self.cfg.image_ocr:
                try:
                    ocr_text = self.ocr.extract_from_pil(pil_crop) or ""
                except Exception as e:
                    logger.warning(f"NonPPT p{page_number} visual OCR failed: {e}")

            # ── Save PNG ──────────────────────────────────────────────────────
            rel_path: Optional[str] = None
            try:
                rel_path = self.img_store.save(pil_crop, page_number, label="img")
            except Exception as e:
                logger.warning(f"NonPPT p{page_number}: visual save failed ({e})")

            w_px, h_px = pil_crop.size
            chunk = self.builder.image_chunk(
                ocr_text        = clean_text(ocr_text),
                image_path      = rel_path,
                image_width_px  = w_px,
                image_height_px = h_px,
                page            = page_number,
                bbox            = img_bbox,
                confidence      = "high",
            )
            if chunk:
                chunks.append(chunk)
                img_idx += 1

        return chunks

    # ── Visual filtering ──────────────────────────────────────────────────────

    def _is_visual_meaningful(self, img: Dict, page) -> bool:
        """
        Keep: charts, graphs, maps, diagrams, block diagrams, large tables-as-images.
        Discard: logos, icons, decorative small images, background graphics.

        Filters applied:
          1. Minimum pixel size (src pixel dimensions)
          2. Minimum bounding-box area (PDF points²)
          3. Aspect ratio within [visual_min_aspect, visual_max_aspect]
          4. Covers at least visual_min_page_fraction of the page area
        """
        # 1. Pixel-size filter (tiny icons)
        src_w = img.get("srcsize", (0, 0))[0]
        src_h = img.get("srcsize", (0, 0))[1]
        if (src_w > 0 and src_w <= self.cfg.min_image_width_px and
                src_h > 0 and src_h <= self.cfg.min_image_height_px):
            return False

        x0 = img.get("x0", 0)
        y0 = img.get("top", 0)
        x1 = img.get("x1", 0)
        y1 = img.get("bottom", 0)
        w  = x1 - x0
        h  = y1 - y0

        if w <= 0 or h <= 0:
            return False

        # 2. Minimum area
        area = w * h
        if area < self.cfg.visual_min_area_pts:
            return False

        # 3. Aspect ratio
        aspect = w / h
        if aspect < self.cfg.visual_min_aspect or aspect > self.cfg.visual_max_aspect:
            return False

        # 4. Page fraction
        page_area = float(page.width) * float(page.height)
        if page_area > 0 and (area / page_area) < self.cfg.visual_min_page_fraction:
            return False

        return True

    def _overlaps_bbox(self, img: Dict, bboxes: List[Tuple]) -> bool:
        """Return True if image center lies inside any of the given bboxes."""
        if not bboxes:
            return False
        cx = (img.get("x0", 0) + img.get("x1", 0)) / 2
        cy = (img.get("top", 0) + img.get("bottom", 0)) / 2
        for (x0, top, x1, bottom) in bboxes:
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    #  Text splitting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _split_text(self, text: str, page, page_number: int, bbox: Dict) -> List[Dict]:
        gran = self.cfg.granularity
        if gran == "page":
            return self._split_page(text, page, page_number, bbox)
        elif gran == "paragraph":
            return self._split_paragraph(text, page, page_number, bbox)
        elif gran == "heading":
            return self._split_heading(text, page, page_number, bbox)
        else:  # "fixed"
            return self._split_fixed(text, page, page_number, bbox)

    def _split_page(self, text: str, page, page_number: int, bbox: Dict) -> List[Dict]:
        if count_tokens(text, self.cfg.tokenizer_model) <= self.cfg.max_tokens:
            c = self.builder.text_chunk(
                text=text, page=page_number, bbox=bbox, confidence="high"
            )
            return [c] if c else []
        return self._split_fixed(text, page, page_number, bbox)

    def _split_paragraph(self, text: str, page, page_number: int, bbox: Dict) -> List[Dict]:
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
                    chunks.extend(self._make_text_chunks(buf, page, page_number, bbox))
                buf = p
        if buf:
            chunks.extend(self._make_text_chunks(buf, page, page_number, bbox))
        return chunks

    def _split_heading(self, text: str, page, page_number: int, bbox: Dict) -> List[Dict]:
        """Split at font-size headings — each segment gets its own word-level bbox."""
        headings = self._find_headings(page)
        if not headings:
            return self._split_paragraph(text, page, page_number, bbox)

        words   = sorted(page.extract_words() or [], key=lambda w: (w["top"], w["x0"]))
        h_tops  = [h[0] for h in headings] + [float("inf")]
        chunks  = []
        for idx, (h_top, h_text) in enumerate(headings):
            next_top  = h_tops[idx + 1]
            seg_words = [w for w in words if h_top <= w["top"] < next_top]
            seg_text  = clean_text(" ".join(w["text"] for w in seg_words))
            if not seg_text.strip():
                continue
            # Each heading segment gets its own precise bbox
            seg_bbox = self._words_bbox(seg_words) if seg_words else bbox
            chunks.extend(self._make_text_chunks(seg_text, page, page_number, seg_bbox))
        return chunks

    def _split_fixed(self, text: str, page, page_number: int, bbox: Dict) -> List[Dict]:
        chunks = []
        for part in split_by_tokens(
            text, self.cfg.max_tokens, self.cfg.token_overlap, self.cfg.tokenizer_model
        ):
            c = self.builder.text_chunk(
                text=part, page=page_number, bbox=bbox, confidence="high"
            )
            if c:
                chunks.append(c)
        return chunks

    def _make_text_chunks(self, text: str, page, page_number: int, bbox: Dict) -> List[Dict]:
        if count_tokens(text, self.cfg.tokenizer_model) > self.cfg.max_tokens:
            return self._split_fixed(text, page, page_number, bbox)
        c = self.builder.text_chunk(
            text=text, page=page_number, bbox=bbox, confidence="high"
        )
        return [c] if c else []

    def _find_headings(self, page) -> List[Tuple[float, str]]:
        """Detect headings by font size / bold in pdfplumber chars."""
        chars    = page.chars
        headings = []
        if not chars:
            return headings
        i = 0
        while i < len(chars):
            c    = chars[i]
            size = c.get("size", 0) or 0
            bold = "Bold" in (c.get("fontname") or "")
            if (size >= self.cfg.heading_font_size_threshold or
                    (self.cfg.heading_bold_counts and bold)):
                run_text = c.get("text", "")
                run_top  = c["top"]
                j = i + 1
                while j < len(chars):
                    nc = chars[j]
                    ns = nc.get("size", 0) or 0
                    nb = "Bold" in (nc.get("fontname") or "")
                    if (abs(nc["top"] - run_top) < 5 and
                            (ns >= self.cfg.heading_font_size_threshold or
                             (self.cfg.heading_bold_counts and nb))):
                        run_text += nc.get("text", "")
                        j += 1
                    else:
                        break
                run_text = run_text.strip()
                if run_text and len(run_text) > 2:
                    headings.append((run_top, run_text))
                i = j
            else:
                i += 1
        return headings

    # ─────────────────────────────────────────────────────────────────────────
    #  Geometry / crop helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _words_bbox(self, words: List[Dict]) -> Dict:
        """
        Compute the union bounding box of a list of pdfplumber word dicts.
        Returns page_bbox fallback dict when words is empty.
        """
        if not words:
            return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}
        return bbox_dict(
            min(w["x0"]     for w in words),
            min(w["top"]    for w in words),
            max(w["x1"]     for w in words),
            max(w["bottom"] for w in words),
        )

    def _text_outside_bboxes(
        self, page, bboxes: List[Tuple]
    ) -> Tuple[str, List[Dict]]:
        """
        Extract words that fall outside all table bboxes.
        Returns (joined_text, word_dicts) so callers can compute accurate bbox.
        """
        words = page.extract_words() or []
        if not bboxes:
            return (page.extract_text() or ""), words

        outside = []
        for w in words:
            wx = (w["x0"] + w["x1"]) / 2
            wy = (w["top"] + w["bottom"]) / 2
            if not any(x0 <= wx <= x1 and top <= wy <= bottom
                       for (x0, top, x1, bottom) in bboxes):
                outside.append(w)
        return " ".join(w["text"] for w in outside), outside

    def _crop_region(self, pil_full, page, bbox: Tuple):
        """Crop a PDF-point bbox from a full-page PIL render."""
        try:
            fw, fh = pil_full.size
            pw, ph = float(page.width), float(page.height)
            sx, sy = fw / pw, fh / ph
            x0, top, x1, bottom = bbox
            px0 = max(0, int(x0    * sx))
            py0 = max(0, int(top   * sy))
            px1 = min(fw, int(x1   * sx))
            py1 = min(fh, int(bottom * sy))
            if (px1 - px0 < self.cfg.min_image_width_px or
                    py1 - py0 < self.cfg.min_image_height_px):
                return None
            return pil_full.crop((px0, py0, px1, py1))
        except Exception:
            return None

    def _crop_image_region(self, page_number: int, img: Dict, page):
        """Crop an embedded image region using the cached full-page render."""
        pil_full = self._get_page_render(page_number)
        if pil_full is None:
            return None
        return self._crop_region(
            pil_full, page,
            (img["x0"], img["top"], img["x1"], img["bottom"]),
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Table formatting helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _rows_to_markdown(self, rows) -> str:
        """Convert pdfplumber table rows (list of lists) to Markdown."""
        if not rows:
            return ""
        rows = [[str(c or "").strip() for c in row] for row in rows if row]
        if not rows:
            return ""
        header    = rows[0]
        separator = ["---"] * len(header)
        lines     = [
            "| " + " | ".join(header)    + " |",
            "| " + " | ".join(separator) + " |",
        ]
        for row in rows[1:]:
            # Pad / trim to header width
            cells = (row + [""] * len(header))[: len(header)]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _rows_to_plain(self, rows) -> str:
        """Convert pdfplumber rows to plain text (fallback for OCR strategy)."""
        if not rows:
            return ""
        lines = []
        for row in rows:
            if row:
                lines.append("\t".join(str(c or "").strip() for c in row))
        return "\n".join(lines)

    def _df_to_markdown(self, df) -> str:
        """Convert a pandas DataFrame (from camelot) to Markdown."""
        try:
            # First row as header if it looks like one
            df = df.reset_index(drop=True)
            rows = df.values.tolist()
            return self._rows_to_markdown(rows)
        except Exception:
            return str(df)