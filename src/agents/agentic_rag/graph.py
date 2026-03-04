from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from src.agents.agentic_rag.nodes import (
    citation_validation_node,
    input_node,
    llm_node,
    save_node,
)
from src.agents.agentic_rag.state import AgenticRAGState
from src.agents.agentic_rag.tools import retrieve as retrieve_tool
from src.log.logs import logger

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = "__end__"
    StateGraph = None


_MAX_TOOL_CALLS = 5


def _count_tool_messages(state: AgenticRAGState) -> int:
    return sum(1 for m in state.get("messages", []) if isinstance(m, ToolMessage))


def _route_after_llm(state: AgenticRAGState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "save_node"

    last = messages[-1]
    if isinstance(last, AIMessage):
        tool_calls = list(getattr(last, "tool_calls", []) or [])
        if tool_calls:
            executed = _count_tool_messages(state)
            if executed + len(tool_calls) <= _MAX_TOOL_CALLS:
                return "tool_node"
            logger.warning(
                "agentic_rag max tool calls reached executed=%d new=%d limit=%d",
                executed,
                len(tool_calls),
                _MAX_TOOL_CALLS,
            )

    return "citation_validation" if _count_tool_messages(state) > 0 else "save_node"


def _build_graph():
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed. pip install langgraph")

    graph = StateGraph(AgenticRAGState)
    graph.add_node("input", input_node)
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", ToolNode(tools=[retrieve_tool]))
    graph.add_node("citation_validation", citation_validation_node)
    graph.add_node("save_node", save_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "llm_node")
    graph.add_conditional_edges(
        "llm_node",
        _route_after_llm,
        {
            "tool_node": "tool_node",
            "citation_validation": "citation_validation",
            "save_node": "save_node",
        },
    )
    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("citation_validation", "save_node")
    graph.add_edge("save_node", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_agent_graph():
    logger.info("agentic_rag.graph initializing")
    return _build_graph()


async def run_agent_chat(
    user_input: str,
    top_k: int,
    images: bool,
    include_text: bool,
    text_search: bool,
    chat_id: str,
    session_id: str,
) -> Dict[str, Any]:
    state: AgenticRAGState = {
        "user_input": user_input,
        "top_k": top_k,
        "images": images,
        "include_text": include_text,
        "text_search": text_search,
        "chat_id": chat_id,
        "session_id": session_id,
        "messages": [],
        "usage_per_call": [],
    }
    result: AgenticRAGState = await get_agent_graph().ainvoke(state)
    return result["response"]
