"""
image_store.py
--------------
Saves PIL images to disk under the configured images directory.

Format strategy:
    page_ocr   → WebP q=80, max 1400px longest side  (full-page screenshots)
    image/table → WebP q=75, max 1000px longest side  (cropped elements)

WebP compresses 8–15× smaller than PNG at equivalent LLM-readable quality.
Hard edges (text, table lines) stay sharp — no JPEG block artefacts.

File layout:
    {settings.images_dir}/{pdf_stem}/p{page:02d}_{label}_{idx:02d}.webp

image_path stored in DB is relative to settings.images_dir:
    "{pdf_stem}/p03_page_ocr_00.webp"
    "{pdf_stem}/p05_img_01.webp"
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, NamedTuple, Tuple

from PIL import Image

from config import settings


# ---------------------------------------------------------------------------
# Format presets
# ---------------------------------------------------------------------------

class _Preset(NamedTuple):
    max_px:  int    # longest-side cap before saving
    quality: int    # WebP quality (0–100)


# Two presets — chosen by label prefix at save time.
# _PRESET_PAGE  = _Preset(max_px=1400, quality=80) 
_PRESET_PAGE = _Preset(max_px=1000, quality=70)  # full-page screenshots
_PRESET_CROP  = _Preset(max_px=1000, quality=75)   # cropped elements (charts, tables)

_PAGE_LABELS  = {"page_ocr"}                        # labels that use the page preset


def _pick_preset(label: str) -> _Preset:
    return _PRESET_PAGE if label in _PAGE_LABELS else _PRESET_CROP


def _resize_if_needed(img: Image.Image, max_px: int) -> Image.Image:
    """Downscale img so its longest side ≤ max_px. Aspect ratio preserved."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_px:
        return img
    scale = max_px / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# ImageStore
# ---------------------------------------------------------------------------

class ImageStore:
    """
    Saves images as WebP and returns paths relative to settings.images_dir.

    Parameters
    ----------
    pdf_stem : str
        PDF filename without extension — used as the subdirectory name.
    """

    def __init__(self, pdf_stem: str) -> None:
        self.pdf_stem  = pdf_stem
        self._base_dir = Path(settings.images_dir) / pdf_stem
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._counters: Dict[Tuple[int, str], int] = {}

    def save(self, pil_image: Image.Image, page_number: int, label: str = "img") -> str:
        """
        Save a PIL image as WebP and return its path relative to settings.images_dir.

        Parameters
        ----------
        pil_image   : PIL.Image
        page_number : int  (1-based)
        label       : str  e.g. "page_ocr", "img", "table"

        Returns
        -------
        str  e.g. "my_report/p03_page_ocr_00.webp"
        """
        preset   = _pick_preset(label)
        # img      = _resize_if_needed(pil_image.convert("RGB"), preset.max_px)
        img = pil_image.convert("L")  # grayscale
        img = _resize_if_needed(img, preset.max_px)

        key      = (page_number, label)
        idx      = self._counters.get(key, 0)
        self._counters[key] = idx + 1

        filename = f"p{page_number:02d}_{label}_{idx:02d}.webp"
        filepath = self._base_dir / filename
        img.save(filepath, format="WEBP", lossless=True, quality=preset.quality, method=6)

        return f"{self.pdf_stem}/{filename}"