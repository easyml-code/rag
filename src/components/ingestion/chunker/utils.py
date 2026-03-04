"""
utils.py — Shared utilities: hashing, token counting, text cleaning, geometry helpers.
"""

import hashlib
import base64
import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# ── Optional tiktoken ─────────────────────────────────────────────────────────
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Hashing
# ─────────────────────────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  Token counting
# ─────────────────────────────────────────────────────────────────────────────

_ENCODER = None


def _get_encoder(model: str):
    global _ENCODER
    if _ENCODER is None and _TIKTOKEN_AVAILABLE:
        try:
            _ENCODER = tiktoken.get_encoding(model)
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc:
        return len(enc.encode(text))
    return int(len(text.split()) * 1.33)


def split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    model: str = "cl100k_base",
) -> List[str]:
    enc = _get_encoder(model)
    if enc:
        tokens = enc.encode(text)
        chunks, start = [], 0
        while start < len(tokens):
            end          = min(start + max_tokens, len(tokens))
            chunks.append(enc.decode(tokens[start:end]))
            if end == len(tokens):
                break
            start = end - overlap_tokens
        return chunks
    else:
        words         = text.split()
        approx_words  = int(max_tokens / 1.33)
        overlap_words = int(overlap_tokens / 1.33)
        chunks, start = [], 0
        while start < len(words):
            end = min(start + approx_words, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = end - overlap_words
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_b64(pil_image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Bbox helpers
# ─────────────────────────────────────────────────────────────────────────────

def bbox_dict(x0: float, y0: float, x1: float, y1: float) -> Dict[str, float]:
    return {"x0": round(x0, 2), "y0": round(y0, 2),
            "x1": round(x1, 2), "y1": round(y1, 2)}


def page_bbox(page) -> Dict[str, float]:
    return bbox_dict(0, 0, float(page.width), float(page.height))


# ─────────────────────────────────────────────────────────────────────────────
#  Misc
# ─────────────────────────────────────────────────────────────────────────────

def file_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
#  Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

_PPT_ARTIFACTS = [
    r"STRICTLY CONFIDENTIAL\s*",
    r"S+T+R+I+C+T+L+Y+\s+C+O+N+F+I+D+E+N+T+I+A+L+\s*",
    r"Hamburger Menu Icon with\s+solid fill\s*",
    r"Home outline\s*",
]


def clean_text(text: str) -> str:
    """Light clean: remove known PPT artifacts, collapse excess whitespace."""
    for pat in _PPT_ARTIFACTS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l.rstrip() for l in text.split("\n")]
    return "\n".join(lines).strip()


def markdown_table_to_plain(md: str) -> str:
    """Convert markdown table to plain key: value sentences."""
    lines = [l.strip() for l in md.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return md
    headers = [h.strip() for h in lines[0].strip("|").split("|")]
    rows    = []
    for line in lines[2:]:
        cells     = [c.strip() for c in line.strip("|").split("|")]
        row_parts = [f"{h}: {c}" for h, c in zip(headers, cells) if h and c]
        if row_parts:
            rows.append(" | ".join(row_parts))
    return "\n".join(rows)
