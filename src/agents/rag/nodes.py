from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.rag.prompt import SYSTEM_PROMPT, build_context_block
from src.agents.rag.state import AgentState
from src.llm.llm import get_llm
from src.log.logs import logger
from src.components.retriever.retriever import retrieve


_CITATION_RE = re.compile(r"(?:\^)?\[(\d+)\]")
_EXACT_NOT_FOUND = "Not found in the document."
_MAX_IMAGES_TO_LLM = 6
_MAX_SINGLE_IMAGE_BYTES = 2_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedupe_and_number_sources(
    retrieval_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """Merge vector/text retrieval results, dedupe by chunk_id, then number 1..N."""
    ordered: List[Dict[str, Any]] = []
    seen_chunk_ids: set = set()

    for bucket in ("vector_results", "text_results"):
        for item in retrieval_results.get(bucket, []) or []:
            chunk_id = item.get("chunk_id")
            dedupe_key = chunk_id if chunk_id else f"{bucket}:{len(ordered)}"
            if dedupe_key in seen_chunk_ids:
                continue
            seen_chunk_ids.add(dedupe_key)
            ordered.append(item)

    return {i + 1: source for i, source in enumerate(ordered)}


def _normalize_usage_metadata(raw: Any) -> Dict[str, Any]:
    usage = raw if isinstance(raw, dict) else {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


def _iter_image_ref_paths(images: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract (slot_name, path) pairs from image refs returned by the retriever."""
    out: List[Tuple[str, str]] = []
    page_image = images.get("page_image")
    if isinstance(page_image, dict):
        path = page_image.get("path")
        if isinstance(path, str) and path:
            out.append(("page_image", path))
    for idx, inline_image in enumerate(images.get("inline_images", []) or []):
        if not isinstance(inline_image, dict):
            continue
        path = inline_image.get("path")
        if isinstance(path, str) and path:
            out.append((f"inline_images[{idx}]", path))
    return out


def _read_blob_cached(
    path_str: str,
    cache: Dict[str, Optional[bytes]],
) -> Optional[bytes]:
    if path_str in cache:
        return cache[path_str]
    try:
        blob = Path(path_str).read_bytes()
        cache[path_str] = blob
        return blob
    except Exception:
        cache[path_str] = None
        return None


def _include_text_flag(state: AgentState) -> bool:
    """
    Resolve the include_text flag.

    When images=False this is irrelevant (text is always sent).
    When images=True:
      - include_text=True  → text chunks + images both go to LLM (images prioritised)
      - include_text=False → images only (metadata anchor, no chunk text)
    Default: True.
    """
    return bool(state.get("include_text", True))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def retriever_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    top_k = int(state.get("top_k", 5))
    include_images = bool(state.get("images", True))
    iwt = _include_text_flag(state)

    # include_text logic:
    #   images=False               → always send text
    #   images=True, iwt=True      → send text + images (images prioritised in prompt)
    #   images=True, iwt=False     → images only (no chunk text to LLM)
    include_text = (not include_images) or iwt

    logger.info(
        "agent.retriever start top_k=%s images=%s include_text=%s",
        top_k, include_images, include_text,
    )

    retrieval_results = await retrieve(
        user_query=user_input,
        top_k=top_k,
        images=include_images,
        image_payload="ref",
        text_search=True,
    )
    sources_by_citation = _dedupe_and_number_sources(retrieval_results)
    context_block = build_context_block(sources_by_citation, include_text=include_text)

    print("\n\n\n\n\nretrieval_results\n", retrieval_results, "\n\n\n")
    print("\n\n\n\n\nsources_by_citation\n", sources_by_citation, "\n\n\n")
    print("\n\n\n\n\ncontext_block\n", context_block, "\n\n\n")

    logger.info("agent.retriever done sources=%d", len(sources_by_citation))
    return {
        "retrieval_results": retrieval_results,
        "sources_by_citation": sources_by_citation,
        "context_block": context_block,
    }


def image_blob_loader_node(state: AgentState) -> Dict[str, Any]:
    """
    Convert retrieved image refs into in-memory blobs.
    Skipped entirely when images=False.
    """
    if not bool(state.get("images", True)):
        logger.info("agent.image_blob_loader skipped images=False")
        return {"image_blobs_by_citation": {}}

    sources_by_citation: Dict[int, Dict[str, Any]] = state.get("sources_by_citation", {})
    blob_cache: Dict[str, Optional[bytes]] = {}
    image_blobs_by_citation: Dict[int, Dict[str, Any]] = {}
    loaded_items = 0

    for citation, source in sources_by_citation.items():
        images = source.get("images")
        if not isinstance(images, dict):
            continue

        citation_images: Dict[str, Any] = {"page_image": None, "inline_images": []}

        for image_slot, path_str in _iter_image_ref_paths(images):
            blob = _read_blob_cached(path_str, blob_cache)
            if blob is None:
                continue
            loaded_items += 1

            if image_slot == "page_image":
                page_ref = images.get("page_image", {})
                if isinstance(page_ref, dict):
                    citation_images["page_image"] = {**page_ref, "blob": blob}
            else:
                match = re.search(r"\[(\d+)\]", image_slot)
                if not match:
                    continue
                idx = int(match.group(1))
                inline_list = images.get("inline_images", []) or []
                if idx < len(inline_list) and isinstance(inline_list[idx], dict):
                    citation_images["inline_images"].append(
                        {**inline_list[idx], "blob": blob}
                    )

        if citation_images["page_image"] or citation_images["inline_images"]:
            image_blobs_by_citation[citation] = citation_images

    logger.info(
        "agent.image_blob_loader loaded=%d unique_paths=%d citations_with_images=%d",
        loaded_items, len(blob_cache), len(image_blobs_by_citation),
    )
    return {"image_blobs_by_citation": image_blobs_by_citation}


def _select_images_for_llm(
    image_blobs_by_citation: Dict[int, Dict[str, Any]],
    sources_by_citation: Dict[int, Dict[str, Any]],
    max_images: int = _MAX_IMAGES_TO_LLM,
    max_single_bytes: int = _MAX_SINGLE_IMAGE_BYTES,
) -> List[Dict[str, Any]]:
    """
    Pick up to max_images images for the multimodal prompt.
    Returns a list of dicts with: citation, media_type, blob, source_file, page, chunk_type.
    Page image preferred over inline images per citation.
    """
    selected: List[Dict[str, Any]] = []

    for citation in sorted(image_blobs_by_citation.keys()):
        item = image_blobs_by_citation[citation]
        src_meta = sources_by_citation.get(citation, {})

        candidates: List[Dict[str, Any]] = []
        page_img = item.get("page_image")
        if isinstance(page_img, dict):
            candidates.append(page_img)
        for inline_img in item.get("inline_images", []) or []:
            if isinstance(inline_img, dict):
                candidates.append(inline_img)

        for candidate in candidates:
            blob = candidate.get("blob")
            if not isinstance(blob, (bytes, bytearray)) or not blob:
                continue
            if len(blob) > max_single_bytes:
                continue
            selected.append({
                "citation": citation,
                "media_type": str(candidate.get("media_type", "image/webp")),
                "blob": bytes(blob),
                "source_file": src_meta.get("source_file", ""),
                "page": src_meta.get("page", ""),
                "chunk_type": src_meta.get("chunk_type", ""),
            })
            if len(selected) >= max_images:
                return selected

    return selected


async def llm_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    sources_by_citation: Dict[int, Dict[str, Any]] = state.get("sources_by_citation", {})
    image_blobs_by_citation: Dict[int, Dict[str, Any]] = state.get("image_blobs_by_citation", {})
    include_images = bool(state.get("images", True))
    iwt = _include_text_flag(state)
    context_block = state.get("context_block", "No sources were retrieved.")

    if not sources_by_citation:
        return {
            "answer": _EXACT_NOT_FOUND,
            "usage_metadata": _normalize_usage_metadata({}),
            "images_sent_to_llm": {"enabled": include_images, "selected_count": 0},
        }

    llm = get_llm()

    # ------------------------------------------------------------------
    # Select images (only when images=True)
    # ------------------------------------------------------------------
    selected_images: List[Dict[str, Any]] = []
    if include_images:
        selected_images = _select_images_for_llm(
            image_blobs_by_citation, sources_by_citation
        )

    # ------------------------------------------------------------------
    # Build content parts
    # ------------------------------------------------------------------
    if selected_images:
        if iwt:
            # include_text=True: both text and images sent.
            # Instruct LLM to treat images as PRIMARY evidence; text confirms or fills gaps.
            instruction = (
                "Images attached below are the PRIMARY evidence — read them carefully first. "
                "The text in 'Retrieved sources' is secondary and should only be used to "
                "confirm or supplement what the images show. "
                "Return a factual answer using only retrieved sources with inline [n] citations. "
                "If evidence is missing in both images and text, reply exactly: "
                "Not found in the document."
            )
        else:
            # include_text=False: images are the only evidence; metadata anchors citations.
            instruction = (
                "Images attached below are the ONLY evidence. "
                "Use the source metadata (file, page) to form citations. "
                "Return a factual answer with inline [n] citations. "
                "If evidence is missing, reply exactly: Not found in the document."
            )

        prompt_text = (
            f"Question:\n{user_input}\n\n"
            "Retrieved sources (use [n] citation numbers):\n"
            f"{context_block}\n\n"
            f"{instruction}"
        )

        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]

        for img in selected_images:
            # Label so the LLM knows which source this image belongs to
            content_parts.append({
                "type": "text",
                "text": (
                    f"[Image for source [{img['citation']}] — "
                    f"file: {img['source_file']}, page: {img['page']}, "
                    f"type: {img['chunk_type']}]"
                ),
            })
            # Send raw bytes as media part (direct multimodal payload)
            content_parts.append({
                "type": "media",
                "mime_type": img["media_type"],
                "data": img["blob"],
            })

        human_message = HumanMessage(content=content_parts)

    else:
        # Text-only path (images=False, or images=True but no blobs loaded)
        prompt_text = (
            f"Question:\n{user_input}\n\n"
            "Retrieved sources (use [n] citation numbers):\n"
            f"{context_block}\n\n"
            "Return a factual answer using only retrieved sources with inline [n] citations. "
            "If evidence is missing, reply exactly: Not found in the document."
        )
        human_message = HumanMessage(content=prompt_text)

    messages = [SystemMessage(content=SYSTEM_PROMPT), human_message]

    logger.info(
        "agent.llm invoke sources=%d selected_images=%d images=%s include_text=%s",
        len(sources_by_citation), len(selected_images), include_images, iwt,
    )

    llm_response = await llm.ainvoke(messages)
    answer = str(getattr(llm_response, "content", llm_response)).strip()
    usage_metadata = _normalize_usage_metadata(
        getattr(llm_response, "usage_metadata", {})
    )

    logger.info("agent.llm done answer_chars=%d", len(answer))
    return {
        "answer": answer,
        "usage_metadata": usage_metadata,
        "images_sent_to_llm": {
            "enabled": include_images,
            "selected_count": len(selected_images),
            "selected_citations": [int(img["citation"]) for img in selected_images],
            "mode": "media_bytes",
            "total_image_bytes": sum(len(img["blob"]) for img in selected_images),
        },
    }


def citation_validation_node(state: AgentState) -> Dict[str, Any]:
    """
    Normalize and validate citations in the LLM answer.

    - Strips leading ^ from ^[n] → [n]
    - Strips accidental space before citation: "MW [1]" → "MW[1]"
    - Removes hallucinated citation numbers not in sources_by_citation
    - Forces _EXACT_NOT_FOUND when no valid citation-backed content remains (strict mode)
    """
    answer = (state.get("answer", "") or "").strip()
    sources_by_citation: Dict[int, Dict[str, Any]] = state.get("sources_by_citation", {})
    available = set(sources_by_citation.keys())

    if not sources_by_citation:
        return {
            "answer": _EXACT_NOT_FOUND,
            "used_citations": [],
            "citation_validation": {
                "is_valid": False,
                "issues": ["no sources retrieved"],
            },
        }

    # Normalize formatting
    normalized = re.sub(r"\^\[(\d+)\]", r"[\1]", answer)
    normalized = re.sub(r"\s+\[(\d+)\]", r"[\1]", normalized)

    # Collect cited numbers in order of appearance (deduplicated)
    cited_ordered: List[int] = []
    for raw in _CITATION_RE.findall(normalized):
        n = int(raw)
        if n not in cited_ordered:
            cited_ordered.append(n)

    invalid = [n for n in cited_ordered if n not in available]
    used = [n for n in cited_ordered if n in available]

    def _drop_invalid(match: re.Match) -> str:
        n = int(match.group(1))
        return f"[{n}]" if n in available else ""

    validated_answer = _CITATION_RE.sub(_drop_invalid, normalized)
    validated_answer = re.sub(r"[ \t]+", " ", validated_answer).strip()

    issues: List[str] = []
    is_valid = True

    if invalid:
        is_valid = False
        issues.append(f"invalid citations removed: {invalid}")

    if not used or validated_answer == _EXACT_NOT_FOUND:
        is_valid = False
        issues.append("no valid citation-backed content")
        validated_answer = _EXACT_NOT_FOUND
        used = []

    if not is_valid:
        logger.warning("agent.citation_validation issues=%s", issues)
    else:
        logger.info("agent.citation_validation passed used=%s", used)

    return {
        "answer": validated_answer,
        "used_citations": used,
        "citation_validation": {
            "is_valid": is_valid,
            "issues": issues,
        },
    }


def output_node(state: AgentState) -> Dict[str, Any]:
    """
    Assemble the final clean API response.

    IMPORTANT: check metadata.images_sent.selected_count.
    If 0 when images=True: image files failed to load from disk (path not found /
    server restart / temp file deleted). LLM silently fell back to text-only.
    """
    answer = state.get("answer", "")
    usage_metadata = state.get("usage_metadata", {})
    images_sent_to_llm = state.get("images_sent_to_llm", {})
    sources_by_citation: Dict[int, Dict[str, Any]] = state.get("sources_by_citation", {})
    used = list(state.get("used_citations", []))

    sources = []
    for n in used:
        if n not in sources_by_citation:
            continue
        src = sources_by_citation[n]
        sources.append({
            "citation": n,
            "source_file": src.get("source_file", ""),
            "filename": src.get("filename", ""),   # original upload name e.g. "report.pdf"
            "page": src.get("page", ""),
            "chunk_type": src.get("chunk_type", ""),
            "chunk_id": src.get("chunk_id", ""),
            "bbox": src.get("bbox"),               # {x0,y0,x1,y1} PDF pts | null
            "text": str(src.get("text", "")).strip(),
        })

    response = {
        "answer": answer,
        "metadata": {
            "sources": sources,
            "used_citations": used,
            "usage": {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            },
            "images_sent": {
                "enabled": images_sent_to_llm.get("enabled", False),
                "selected_count": images_sent_to_llm.get("selected_count", 0),
                "selected_citations": images_sent_to_llm.get("selected_citations", []),
                "mode": images_sent_to_llm.get("mode", "none"),
                "total_image_bytes": images_sent_to_llm.get("total_image_bytes", 0),
            },
        },
    }

    logger.info(
        "agent.output used_citations=%s images_sent=%d",
        used, images_sent_to_llm.get("selected_count", 0),
    )
    return {"used_citations": used, "response": response}
