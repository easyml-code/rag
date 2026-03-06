from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.agentic_rag import cache
from src.agents.agentic_rag import util as db
from src.agents.agentic_rag.prompt import SYSTEM_PROMPT
from src.agents.agentic_rag.state import AgenticRAGState
from src.agents.agentic_rag.tools import get_cached_payload, retrieve as retrieve_tool
from src.llm.llm import get_llm
from src.log.logs import logger


_CITATION_RE = re.compile(r"(?:\^)?\[(\d+)\]")
_CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]")
_TOOL_META_RE = re.compile(r"\[TOOL_META\]\s*cache_key=([a-f0-9]+)\s+query=(.*)$")
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|hola|namaste|good morning|good afternoon|good evening|thanks|thank you)\b",
    re.IGNORECASE,
)
_NOT_FOUND = "Not found in the document."


def _token_details(raw: Any) -> Dict[str, int]:
    details = raw if isinstance(raw, dict) else {}
    out: Dict[str, int] = {}
    for k, v in details.items():
        try:
            out[str(k)] = int(v or 0)
        except (TypeError, ValueError):
            continue
    return out


def _normalize_usage_metadata(raw: Any) -> Dict[str, Any]:
    usage = raw if isinstance(raw, dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_details": _token_details(usage.get("input_token_details")),
        "output_token_details": _token_details(usage.get("output_token_details")),
    }


def _merge_detail_maps(calls: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for call in calls:
        details = call.get(key, {})
        if not isinstance(details, dict):
            continue
        for name, value in details.items():
            try:
                merged[str(name)] = merged.get(str(name), 0) + int(value or 0)
            except (TypeError, ValueError):
                continue
    return merged


def _normalize_usage_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for call in calls:
        input_tokens = int(call.get("input_tokens", 0) or 0)
        output_tokens = int(call.get("output_tokens", 0) or 0)
        total_tokens = int(call.get("total_tokens", input_tokens + output_tokens) or 0)
        input_details = _token_details(call.get("input_token_details"))
        output_details = _token_details(call.get("output_token_details"))

        # Skip synthetic/non-provider calls that carry no usage footprint.
        if (
            input_tokens == 0
            and output_tokens == 0
            and total_tokens == 0
            and not input_details
            and not output_details
        ):
            continue

        normalized.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_token_details": input_details,
                "output_token_details": output_details,
            }
        )
    return normalized


def _usage_totals(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    structural_llm_calls = len(calls)
    normalized_calls = _normalize_usage_calls(calls)
    total_input = sum(int(c.get("input_tokens", 0) or 0) for c in normalized_calls)
    outputs = [int(c.get("output_tokens", 0) or 0) for c in normalized_calls]
    output_tokens_cumulative = sum(outputs)
    final_output = outputs[-1] if outputs else 0
    internal = output_tokens_cumulative - final_output
    total_tokens = sum(
        int(c.get("total_tokens", int(c.get("input_tokens", 0) or 0) + int(c.get("output_tokens", 0) or 0)) or 0)
        for c in normalized_calls
    )
    final_input_details = _token_details(
        normalized_calls[-1].get("input_token_details", {}) if normalized_calls else {}
    )
    final_output_details = _token_details(
        normalized_calls[-1].get("output_token_details", {}) if normalized_calls else {}
    )
    return {
        "input_tokens": total_input,
        "output_tokens": output_tokens_cumulative,
        "total_tokens": total_tokens,
        "internal_tokens": internal,
        "final_output_tokens": final_output,
        "llm_calls": structural_llm_calls,
        "llm_calls_with_usage": len(normalized_calls),
        "input_token_details": final_input_details,
        "output_token_details": final_output_details,
        "aggregated_input_token_details": _merge_detail_maps(normalized_calls, "input_token_details"),
        "aggregated_output_token_details": _merge_detail_maps(normalized_calls, "output_token_details"),
        "per_call": normalized_calls,
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return "\n".join(parts).strip()
    return str(content).strip()


def _latest_non_tool_ai_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return _content_to_text(msg.content)
    return ""


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.search(text or ""))


def _tool_meta_from_tool_message(msg: ToolMessage) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    content = msg.content

    def _check_text(text: str) -> None:
        for line in text.splitlines():
            m = _TOOL_META_RE.search(line.strip())
            if m:
                out.append((m.group(1), m.group(2).strip()))

    if isinstance(content, str):
        _check_text(content)
        return out

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    _check_text(text)
    return out


def _expand_citation_groups(text: str) -> str:
    """
    Normalize grouped citations:
      [1, 2, 3] -> [1][2][3]
    """
    def _repl(match: re.Match) -> str:
        raw = match.group(1)
        nums: List[int] = []
        raw = raw.replace(";", ",")
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                nums.append(int(part))
        return "".join(f"[{n}]" for n in nums)

    return _CITATION_GROUP_RE.sub(_repl, text)


def _collect_tool_data(messages: List[BaseMessage]) -> Tuple[Dict[int, Dict[str, Any]], List[str], Dict[str, Any]]:
    sources_by_citation: Dict[int, Dict[str, Any]] = {}
    rephrased_queries: List[str] = []
    seen_cache_keys = set()
    image_citations: List[int] = []
    image_count = 0
    total_image_bytes = 0
    images_enabled = False

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue

        for cache_key, query in _tool_meta_from_tool_message(msg):
            if cache_key in seen_cache_keys:
                continue
            seen_cache_keys.add(cache_key)

            payload = get_cached_payload(cache_key)
            if not payload:
                continue

            if query:
                rephrased_queries.append(query)

            src_map = payload.get("sources_by_citation", {})
            if isinstance(src_map, dict):
                for k, v in src_map.items():
                    try:
                        n = int(k)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(v, dict) and n not in sources_by_citation:
                        sources_by_citation[n] = v

            cits = payload.get("image_citations", [])
            if isinstance(cits, list):
                for c in cits:
                    try:
                        n = int(c)
                    except (TypeError, ValueError):
                        continue
                    if n not in image_citations:
                        image_citations.append(n)
            image_count += int(payload.get("images_sent_count", 0) or 0)
            total_image_bytes += int(payload.get("total_image_bytes", 0) or 0)
            images_enabled = images_enabled or bool(payload.get("images_enabled", False))

    if not images_enabled:
        mode = "none"
    elif image_count > 0:
        mode = "tool_message_image_blocks"
    else:
        mode = "enabled_no_images_found"
    images_sent = {
        "enabled": images_enabled,
        "selected_count": image_count,
        "selected_citations": image_citations,
        "mode": mode,
        "total_image_bytes": total_image_bytes,
    }
    return sources_by_citation, rephrased_queries, images_sent


async def input_node(state: AgenticRAGState) -> Dict[str, Any]:
    chat_id = state.get("chat_id", "")
    user_input = state.get("user_input", "")

    turns = cache.get(chat_id)
    if turns is None:
        turns = await db.load_history(chat_id)
        if turns:
            cache.put(chat_id, turns)

    history_messages: List[BaseMessage] = []
    for t in turns or []:
        human = str(t.get("human", "")).strip()
        ai = str(t.get("ai", "")).strip()
        if human:
            history_messages.append(HumanMessage(content=human))
        if ai:
            history_messages.append(AIMessage(content=ai))

    logger.info(
        "agentic_rag.input chat_id=%s history_turns=%d",
        chat_id,
        len(turns or []),
    )
    return {
        "messages": [*history_messages, HumanMessage(content=user_input)],
        "history_turns_loaded": len(turns or []),
    }


async def llm_node(state: AgenticRAGState) -> Dict[str, Any]:
    messages = list(state.get("messages", []))
    user_input = state.get("user_input", "")
    is_greeting = _is_greeting(user_input)
    prior_tool_msgs = sum(1 for m in messages if isinstance(m, ToolMessage))
    usage: Dict[str, Any]

    # Greeting path: normal conversational reply without tool use.
    if is_greeting and prior_tool_msgs == 0:
        llm = get_llm()
        response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
        usage = _normalize_usage_metadata(getattr(response, "usage_metadata", {}))
    else:
        llm = get_llm().bind_tools([retrieve_tool])
        model_response = await llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
        usage = _normalize_usage_metadata(getattr(model_response, "usage_metadata", {}))
        response = model_response

        # SUPER IMPORTANT requirement:
        # non-greeting query must not be answered from model's own knowledge.
        # If first pass produced no tool call, force a retrieve tool call.
        if prior_tool_msgs == 0 and not list(getattr(model_response, "tool_calls", []) or []):
            forced_id = f"forced_retrieve_{uuid.uuid4().hex[:10]}"
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve",
                        "args": {"query": user_input},
                        "id": forced_id,
                        "type": "tool_call",
                    }
                ],
            )
            logger.info("agentic_rag.llm forced retrieve tool_call id=%s", forced_id)
    usage_per_call = list(state.get("usage_per_call", []))
    usage_per_call.append(usage)

    logger.info(
        "agentic_rag.llm tool_calls=%d",
        len(getattr(response, "tool_calls", []) or []),
    )
    return {"messages": [response], "usage_per_call": usage_per_call}


def citation_validation_node(state: AgenticRAGState) -> Dict[str, Any]:
    messages: List[BaseMessage] = list(state.get("messages", []))
    answer = _latest_non_tool_ai_text(messages) or _NOT_FOUND

    has_tool_call = any(isinstance(m, ToolMessage) for m in messages)
    user_input = state.get("user_input", "")
    is_greeting = _is_greeting(user_input)
    if not has_tool_call:
        if not is_greeting:
            return {
                "answer": _NOT_FOUND,
                "used_citations": [],
                "sources_by_citation": {},
                "rephrased_queries": [],
                "images_sent_to_llm": {
                    "enabled": False,
                    "selected_count": 0,
                    "selected_citations": [],
                    "mode": "none",
                    "total_image_bytes": 0,
                },
                "citation_validation": {
                    "is_valid": False,
                    "issues": ["tool was required for non-greeting query but not used"],
                },
            }
        return {
            "answer": answer,
            "used_citations": [],
            "sources_by_citation": {},
            "rephrased_queries": [],
            "images_sent_to_llm": {
                "enabled": False,
                "selected_count": 0,
                "selected_citations": [],
                "mode": "none",
                "total_image_bytes": 0,
            },
            "citation_validation": {"is_valid": True, "issues": []},
        }

    sources_by_citation, rephrased_queries, images_sent = _collect_tool_data(messages)
    available = set(sources_by_citation.keys())

    normalized = _expand_citation_groups(answer)
    normalized = re.sub(r"\^\[(\d+)\]", r"[\1]", normalized)
    normalized = re.sub(r"\s+\[(\d+)\]", r"[\1]", normalized)

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
    if not used or validated_answer == _NOT_FOUND:
        is_valid = False
        issues.append("no valid citation-backed content")
        validated_answer = _NOT_FOUND
        used = []

    return {
        "answer": validated_answer,
        "used_citations": used,
        "sources_by_citation": sources_by_citation,
        "rephrased_queries": rephrased_queries,
        "images_sent_to_llm": images_sent,
        "citation_validation": {
            "is_valid": is_valid,
            "issues": issues,
            "available_citations": sorted(list(available)),
            "found_citations": cited_ordered,
            "invalid_citations": invalid,
        },
    }


def _compact_usage_payload(usage_metadata: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "input_tokens": int(usage_metadata.get("input_tokens", 0) or 0),
        "output_tokens": int(usage_metadata.get("output_tokens", 0) or 0),
        "total_tokens": int(usage_metadata.get("total_tokens", 0) or 0),
        "llm_calls": int(usage_metadata.get("llm_calls", 0) or 0),
    }

    internal_tokens = int(usage_metadata.get("internal_tokens", 0) or 0)
    if internal_tokens > 0:
        payload["internal_tokens"] = internal_tokens

    final_output_tokens = int(usage_metadata.get("final_output_tokens", 0) or 0)
    if final_output_tokens > 0:
        payload["final_output_tokens"] = final_output_tokens

    llm_calls_with_usage = int(usage_metadata.get("llm_calls_with_usage", 0) or 0)
    if llm_calls_with_usage and llm_calls_with_usage != int(payload["llm_calls"]):
        payload["llm_calls_with_usage"] = llm_calls_with_usage

    input_details = usage_metadata.get("input_token_details", {})
    if isinstance(input_details, dict) and input_details:
        payload["input_token_details"] = input_details

    output_details = usage_metadata.get("output_token_details", {})
    if isinstance(output_details, dict) and output_details:
        payload["output_token_details"] = output_details

    per_call = usage_metadata.get("per_call", [])
    if isinstance(per_call, list) and len(per_call) > 1:
        payload["per_call"] = per_call

        agg_in = usage_metadata.get("aggregated_input_token_details", {})
        if isinstance(agg_in, dict) and agg_in:
            payload["aggregated_input_token_details"] = agg_in

        agg_out = usage_metadata.get("aggregated_output_token_details", {})
        if isinstance(agg_out, dict) and agg_out:
            payload["aggregated_output_token_details"] = agg_out

    return payload


async def save_node(state: AgenticRAGState) -> Dict[str, Any]:
    messages: List[BaseMessage] = list(state.get("messages", []))
    answer = state.get("answer") or _latest_non_tool_ai_text(messages) or _NOT_FOUND
    user_input = state.get("user_input", "")
    chat_id = state.get("chat_id", "")
    session_id = state.get("session_id", "")

    usage_metadata = _usage_totals(state.get("usage_per_call", []))
    used = list(state.get("used_citations", []))
    sources_by_citation = state.get("sources_by_citation", {})
    images_sent = state.get("images_sent_to_llm", {})
    rephrased_queries = [
        q for q in state.get("rephrased_queries", [])
        if isinstance(q, str) and q.strip() and q.strip() != user_input.strip()
    ]

    if chat_id:
        turns = cache.get(chat_id) or []
        turns.append({"human": user_input, "ai": answer, "usage": usage_metadata})
        cache.put(chat_id, turns)

    if chat_id and session_id:
        try:
            scheduled = db.schedule_save_message(
                chat_id=chat_id,
                session_id=session_id,
                human=user_input,
                ai=answer,
                input_tokens=usage_metadata["input_tokens"],
                output_tokens=usage_metadata["output_tokens"],
                internal_tokens=usage_metadata["internal_tokens"],
            )
            if not scheduled:
                await db.save_message(
                    chat_id=chat_id,
                    session_id=session_id,
                    human=user_input,
                    ai=answer,
                    input_tokens=usage_metadata["input_tokens"],
                    output_tokens=usage_metadata["output_tokens"],
                    internal_tokens=usage_metadata["internal_tokens"],
                )
        except Exception as exc:
            logger.warning("agentic_rag.db.save failed: %s", exc)

    sources = []
    for n in used:
        src = sources_by_citation.get(n)
        if not isinstance(src, dict):
            continue
        sources.append(
            {
                "citation": n,
                "source_file": src.get("source_file", ""),
                "filename": src.get("filename", ""),
                "page": src.get("page", ""),
                "chunk_type": src.get("chunk_type", ""),
                "chunk_id": src.get("chunk_id", ""),
                "bbox": src.get("bbox"),
                "text": str(src.get("text", "")).strip(),
            }
        )

    usage_payload = _compact_usage_payload(usage_metadata)

    response = {
        "answer": answer,
        "metadata": {
            "history_turns_loaded": int(state.get("history_turns_loaded", 0) or 0),
            "sources": sources,
            "used_citations": used,
            "rephrased_queries": rephrased_queries,
            "citation_validation": state.get("citation_validation", {}),
            "usage_metadata": usage_payload,
            "images_sent": {
                "enabled": images_sent.get("enabled", False),
                "selected_count": images_sent.get("selected_count", 0),
                "selected_citations": images_sent.get("selected_citations", []),
                "mode": images_sent.get("mode", "none"),
                "total_image_bytes": images_sent.get("total_image_bytes", 0),
            },
        },
    }

    logger.info(
        "agentic_rag.save used_citations=%s tool_used=%s",
        used,
        bool(used),
    )
    return {"usage_metadata": usage_metadata, "response": response}
