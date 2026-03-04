from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from src.log.logs import logger

from src.agents.rag.nodes import (
    citation_validation_node,
    image_blob_loader_node,
    llm_node,
    output_node,
    retriever_node,
)
from src.agents.rag.state import AgentState

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = "__end__"
    StateGraph = None


def _route_after_retriever(state: AgentState) -> str:
    return "image_blob_loader" if bool(state.get("images", True)) else "llm"


def _build_graph():
    if StateGraph is None:
        raise RuntimeError(
            "langgraph is not installed. Install it with: pip install langgraph"
        )

    graph = StateGraph(AgentState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("image_blob_loader", image_blob_loader_node)
    graph.add_node("llm", llm_node)
    graph.add_node("citation_validation", citation_validation_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("retriever")
    graph.add_conditional_edges("retriever", _route_after_retriever)
    graph.add_edge("image_blob_loader", "llm")
    graph.add_edge("llm", "citation_validation")
    graph.add_edge("citation_validation", "output")
    graph.add_edge("output", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_chat_graph():
    logger.info("agent.graph initializing")
    return _build_graph()


async def run_chat_agent(
    user_input: str,
    top_k: int = 5,
    images: bool = True,
    include_text: bool = True,
) -> Dict[str, Any]:
    """
    Run the chat agent graph.

    images=False                   → text chunks only to LLM
    images=True, include_text=True → text chunks + images to LLM (images prioritised)
    images=True, include_text=False → images + metadata anchor only to LLM
    """
    state: AgentState = {
        "user_input": user_input,
        "top_k": top_k,
        "images": images,
        "include_text": include_text,
    }
    graph = get_chat_graph()
    out: Dict[str, Any] = await graph.ainvoke(state)
    return out["response"]