"""
screenshotter.py — Render PDF pages to PIL Images or base64 PNG strings.

Backends (tried in order):
  1. pypdfium2  — no system dependency (recommended): pip install pypdfium2
  2. pdf2image  — requires poppler: pip install pdf2image

Usage:
    ss          = PageScreenshotter(pdf_path, dpi=150)
    pil_img     = ss.render_page_to_pil(page_number)   # 1-based
    b64, (w, h) = ss.render_page(page_number)          # base64 string
    ss.close()
"""

from __future__ import annotations
from typing import Tuple
import io

from src.components.ingestion.chunker.utils import pil_to_b64


def _try_pypdfium2():
    try:
        import pypdfium2 as pdfium
        return pdfium
    except ImportError:
        return None


def _try_pdf2image():
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except ImportError:
        return None


class PageScreenshotter:
    """
    Renders individual PDF pages as PIL Images (or base64 strings).

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    dpi : int
        Render resolution. 150 (default) balances speed and OCR quality.
    fmt : str
        Image format: "PNG" (default) or "JPEG".
    """

    def __init__(self, pdf_path: str, dpi: int = 150, fmt: str = "PNG"):
        self.pdf_path = pdf_path
        self.dpi      = dpi
        self.fmt      = fmt.upper()

        self._pdfium    = _try_pypdfium2()
        self._pdf2image = _try_pdf2image()

        if self._pdfium is None and self._pdf2image is None:
            raise RuntimeError(
                "No PDF rendering backend. "
                "Install: pip install pypdfium2  (or pdf2image + poppler)"
            )

        self._pdfium_doc = None
        if self._pdfium:
            self._pdfium_doc = self._pdfium.PdfDocument(pdf_path)

    # ── PIL (preferred) ───────────────────────────────────────────────────────

    def render_page_to_pil(self, page_number: int):
        """Render page to PIL Image. page_number is 1-based."""
        if self._pdfium_doc is not None:
            return self._pdfium_to_pil(page_number)
        return self._pdf2image_to_pil(page_number)

    def _pdfium_to_pil(self, page_number: int):
        page   = self._pdfium_doc[page_number - 1]
        scale  = self.dpi / 72.0
        bitmap = page.render(scale=scale, rotation=0)
        pil    = bitmap.to_pil()
        if self.fmt == "JPEG":
            pil = pil.convert("RGB")
        return pil

    def _pdf2image_to_pil(self, page_number: int):
        pages = self._pdf2image(
            self.pdf_path,
            dpi        = self.dpi,
            first_page = page_number,
            last_page  = page_number,
            fmt        = self.fmt,
        )
        if not pages:
            raise RuntimeError(f"pdf2image returned no pages for p{page_number}")
        return pages[0]

    # ── Base64 (for OCR engine compatibility) ─────────────────────────────────

    def render_page(self, page_number: int) -> Tuple[str, Tuple[int, int]]:
        """Render page and return (base64_png_string, (width_px, height_px))."""
        pil = self.render_page_to_pil(page_number)
        return pil_to_b64(pil, fmt=self.fmt), pil.size

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        if self._pdfium_doc is not None:
            try:
                self._pdfium_doc.close()
            except Exception:
                pass
            self._pdfium_doc = None

    def __del__(self):
        self.close()
