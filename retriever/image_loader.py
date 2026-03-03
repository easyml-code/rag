"""
image_loader.py
---------------
Enriches retrieval results with image refs or binary blobs.

Images are stored at: {settings.images_dir}/{image_path}
where image_path is e.g. "my_report/p03_page_ocr_00.webp"

Two image types per chunk:
  1. page_image    — full-page screenshot  (chunk["image_path"])
  2. inline_images — images referenced inside chunk_text as ![alt](image_N.webp)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings


_INLINE_IMG_RE = re.compile(r'!\[.*?\]\(([^)]+)\)')

_MEDIA_TYPES: Dict[str, str] = {
    ".webp": "image/webp",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
}


def _media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "image/webp")


# ---------------------------------------------------------------------------
# Internal readers
# ---------------------------------------------------------------------------

def _read_ref(path: Path, key: str) -> Optional[Dict[str, Any]]:
    """Return lightweight reference metadata (no bytes)."""
    if not path.exists():
        return None
    return {
        "key":        key,
        "path":       str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "media_type": _media_type(path),
    }


def _read_blob(path: Path, key: str) -> Optional[Dict[str, Any]]:
    """Return full image bytes + metadata."""
    if not path.exists():
        return None
    blob = path.read_bytes()
    return {
        "key":        key,
        "path":       str(path.resolve()),
        "blob":       blob,
        "size_bytes": len(blob),
        "media_type": _media_type(path),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_images(result: Dict[str, Any], mode: str = "ref") -> Dict[str, Any]:
    """
    Enrich a retrieval result with image refs or binary blobs.

    Adds result["images"]:
    {
        "page_image":    {...} | None,
        "inline_images": [{...}, ...]
    }

    Parameters
    ----------
    result : retrieval result dict (must contain image_path and text).
    mode   : "ref"  → lightweight metadata only (default, used by agent)
             "blob" → includes raw bytes (used by /retrieve with image_payload="blob")
    """
    if mode not in {"ref", "blob"}:
        raise ValueError(f'mode must be "ref" or "blob", got {mode!r}')

    reader      = _read_blob if mode == "blob" else _read_ref
    images_base = Path(settings.images_dir)
    image_path  = result.get("image_path", "") or ""
    text        = result.get("text", "") or ""

    images: Dict[str, Any] = {"page_image": None, "inline_images": []}

    # ── Page screenshot ────────────────────────────────────────────────────
    if image_path:
        images["page_image"] = reader(images_base / image_path, key=image_path)

    # ── Inline images (same folder as page screenshot) ────────────────────
    if image_path:
        img_folder = images_base / Path(image_path).parent
        for name in _INLINE_IMG_RE.findall(text):
            inline_key = str((Path(image_path).parent / name).as_posix())
            img = reader(img_folder / name, key=inline_key)
            if img:
                images["inline_images"].append({"name": name, **img})

    result["images"] = images
    return result