from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    top_k: int
    images: bool
    include_text: bool               # default True; when images=True, also send chunk text to LLM
    retrieval_results: Dict[str, List[Dict[str, Any]]]
    sources_by_citation: Dict[int, Dict[str, Any]]
    image_blobs_by_citation: Dict[int, Dict[str, Any]]
    context_block: str
    answer: str
    usage_metadata: Dict[str, Any]
    images_sent_to_llm: Dict[str, Any]
    used_citations: List[int]
    citation_validation: Dict[str, Any]
    response: Dict[str, Any]