from __future__ import annotations

from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgenticRAGState(TypedDict, total=False):
    user_input: str
    top_k: int
    images: bool
    include_text: bool
    text_search: bool
    chat_id: str
    session_id: str

    messages: Annotated[List[BaseMessage], add_messages]
    usage_per_call: List[Dict[str, Any]]
    history_turns_loaded: int

    sources_by_citation: Dict[int, Dict[str, Any]]
    rephrased_queries: List[str]
    images_sent_to_llm: Dict[str, Any]
    citation_validation: Dict[str, Any]

    usage_metadata: Dict[str, Any]
    answer: str
    used_citations: List[int]
    response: Dict[str, Any]
