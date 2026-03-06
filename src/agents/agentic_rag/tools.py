from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated

from src.components import retrieve as do_retrieve
from src.log.logs import logger


_DEFAULT_TOP_K = 3
_MAX_TOP_K = 20
_MAX_IMAGES_TO_TOOL_CONTEXT = 6
_MAX_SOURCE_TEXT_CHARS = 2000

_TOOL_CACHE_TTL_SEC = 600.0
_TOOL_CACHE_MAX_ITEMS = 128
_TOOL_PAYLOAD_CACHE: Dict[str, Dict[str, Any]] = {}
_GLOBAL_CITATION_COUNTER = 0


def _cleanup_cache(now: float) -> None:
    stale = [
        key
        for key, entry in _TOOL_PAYLOAD_CACHE.items()
        if now - float(entry.get("ts", 0.0)) > _TOOL_CACHE_TTL_SEC
    ]
    for key in stale:
        _TOOL_PAYLOAD_CACHE.pop(key, None)

    while len(_TOOL_PAYLOAD_CACHE) > _TOOL_CACHE_MAX_ITEMS:
        oldest_key = min(
            _TOOL_PAYLOAD_CACHE.keys(),
            key=lambda k: float(_TOOL_PAYLOAD_CACHE[k].get("ts", 0.0)),
        )
        _TOOL_PAYLOAD_CACHE.pop(oldest_key, None)


def _put_payload(payload: Dict[str, Any]) -> str:
    now = time.monotonic()
    _cleanup_cache(now)

    key = uuid.uuid4().hex
    _TOOL_PAYLOAD_CACHE[key] = {"payload": payload, "ts": now}
    return key


def get_cached_payload(cache_key: str) -> Optional[Dict[str, Any]]:
    entry = _TOOL_PAYLOAD_CACHE.get(cache_key)
    if not entry:
        return None

    now = time.monotonic()
    if now - float(entry.get("ts", 0.0)) > _TOOL_CACHE_TTL_SEC:
        _TOOL_PAYLOAD_CACHE.pop(cache_key, None)
        return None

    entry["ts"] = now
    return entry.get("payload")


def _next_citation() -> int:
    global _GLOBAL_CITATION_COUNTER
    _GLOBAL_CITATION_COUNTER += 1
    return _GLOBAL_CITATION_COUNTER


def _coerce_top_k(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_TOP_K
    return max(1, min(value, _MAX_TOP_K))


def _coerce_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _dedupe_results(retrieval_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    seen = set()
    for bucket in ("vector_results", "text_results"):
        for item in retrieval_results.get(bucket, []) or []:
            key = item.get("chunk_id") or f"{bucket}:{len(ordered)}"
            if key in seen:
                continue
            seen.add(key)
            ordered.append(item)
    return ordered


def _read_blob(path: str) -> Optional[bytes]:
    try:
        return Path(path).read_bytes()
    except Exception:
        return None


def _pick_preferred_image(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    images = item.get("images")
    if not isinstance(images, dict):
        return None

    page = images.get("page_image")
    if isinstance(page, dict):
        path = page.get("path")
        if isinstance(path, str) and path:
            blob = _read_blob(path)
            if blob:
                return {
                    "blob": blob,
                    "mime_type": str(page.get("media_type", "image/webp")),
                }

    for inline in images.get("inline_images", []) or []:
        if not isinstance(inline, dict):
            continue
        path = inline.get("path")
        if not isinstance(path, str) or not path:
            continue
        blob = _read_blob(path)
        if blob:
            return {
                "blob": blob,
                "mime_type": str(inline.get("media_type", "image/webp")),
            }
    return None


def _source_line(source: Dict[str, Any], include_text: bool) -> str:
    lines = [
        f"[{source['citation']}]",
        f"file={source['source_file']}",
        f"page={source['page']}",
        f"chunk_type={source['chunk_type']}",
    ]
    if include_text:
        lines.append(f"text={str(source.get('text', '')).strip()[:_MAX_SOURCE_TEXT_CHARS]}")
    return "\n".join(lines)


@tool
async def retrieve(
    query: str,
    state: Annotated[Dict[str, Any], InjectedState],
) -> List[Dict[str, Any]]:
    """Retrieve relevant context for the query from indexed documents."""
    top_k = _coerce_top_k(state.get("top_k"))
    images_enabled = _coerce_bool(state.get("images"), True)
    include_text_cfg = _coerce_bool(state.get("include_text"), False)
    text_search = _coerce_bool(state.get("text_search"), True)
    include_text = (not images_enabled) or include_text_cfg

    retrieval_results = await do_retrieve(
        user_query=query,
        top_k=top_k,
        images=images_enabled,
        image_payload="ref",
        text_search=text_search,
    )

    deduped = _dedupe_results(retrieval_results)
    sources_by_citation: Dict[int, Dict[str, Any]] = {}
    image_citations: List[int] = []
    blocks: List[Dict[str, Any]] = []
    source_lines: List[str] = []
    total_image_bytes = 0

    for item in deduped:
        citation = _next_citation()
        source = {
            "citation": citation,
            "source_file": item.get("source_file", ""),
            "filename": item.get("filename", ""),
            "page": item.get("page", ""),
            "chunk_type": item.get("chunk_type", ""),
            "chunk_id": item.get("chunk_id", ""),
            "bbox": item.get("bbox"),
            "text": str(item.get("text", "")).strip(),
        }
        sources_by_citation[citation] = source
        source_lines.append(_source_line(source, include_text=include_text))

        if not images_enabled or len(image_citations) >= _MAX_IMAGES_TO_TOOL_CONTEXT:
            continue
        image = _pick_preferred_image(item)
        if not image:
            continue

        blob = image["blob"]
        image_citations.append(citation)
        total_image_bytes += len(blob)
        blocks.append(
            {
                "type": "text",
                "text": (
                    f"Image evidence for source [{citation}] "
                    f"(file={source['source_file']}, page={source['page']})"
                ),
            }
        )
        blocks.append(
            {
                "type": "image",
                "base64": base64.b64encode(blob).decode("utf-8"),
                "mime_type": image["mime_type"],
            }
        )

    if not images_enabled:
        image_mode = "none"
    elif image_citations:
        image_mode = "tool_message_image_blocks"
    else:
        image_mode = "enabled_no_images_found"

    payload = {
        "query": query,
        "sources_by_citation": sources_by_citation,
        "image_citations": image_citations,
        "images_sent_count": len(image_citations),
        "total_image_bytes": total_image_bytes,
        "images_enabled": images_enabled,
        "image_mode": image_mode,
        "retrieval_config": {
            "top_k": top_k,
            "images": images_enabled,
            "include_text": include_text_cfg,
            "include_text_effective": include_text,
            "text_search": text_search,
            "image_payload": "ref",
        },
    }
    cache_key = _put_payload(payload)

    prefix = {
        "type": "text",
        "text": f"[TOOL_META] cache_key={cache_key} query={query}",
    }
    context = {
        "type": "text",
        "text": (
            "Retrieved sources (use only these citation numbers):\n"
            + ("\n\n".join(source_lines) if source_lines else "No relevant sources found.")
        ),
    }

    logger.info(
        "agentic_rag.tool.retrieve query=%r sources=%d images=%s image_count=%d image_bytes=%d cache_key=%s top_k=%d include_text=%s text_search=%s",
        query,
        len(sources_by_citation),
        images_enabled,
        len(image_citations),
        total_image_bytes,
        cache_key[:8],
        top_k,
        include_text,
        text_search,
    )
    return [prefix, context, *blocks]
