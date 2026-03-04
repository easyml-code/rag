from __future__ import annotations

import asyncio
from typing import Dict, List

from src.log.logs import logger

try:
    from supabase import create_client
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    logger.warning("supabase package not installed — DB persistence disabled. pip install supabase")

from src.config import settings


def _client():
    return create_client(settings.supabase_url, settings.supabase_key)


def _ready() -> bool:
    return _AVAILABLE and bool(settings.supabase_url) and bool(settings.supabase_key)


async def load_history(chat_id: str, limit: int = 20) -> List[Dict[str, str]]:
    if not _ready():
        return []

    def _fetch():
        result = (
            _client()
            .table("messages")
            .select("human_message, ai_message")
            .eq("chat_id", chat_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

    try:
        rows = await asyncio.to_thread(_fetch)
        return [{"human": r["human_message"], "ai": r["ai_message"]} for r in reversed(rows)]
    except Exception as e:
        logger.warning("db.load_history failed chat_id=%s: %s", chat_id, e)
        return []


async def save_message(
    chat_id: str,
    session_id: str,
    human: str,
    ai: str,
    input_tokens: int,
    output_tokens: int,
    internal_tokens: int,
) -> None:
    if not _ready():
        return

    def _save():
        client = _client()
        client.table("messages").insert({
            "chat_id": chat_id,
            "session_id": session_id,
            "human_message": human,
            "ai_message": ai,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "internal_tokens": internal_tokens,
        }).execute()

        existing = (
            client.table("chat_sessions")
            .select("id, total_input_tokens, total_output_tokens, total_internal_tokens")
            .eq("chat_id", chat_id)
            .eq("session_id", session_id)
            .execute()
        )

        if existing.data:
            row = existing.data[0]
            client.table("chat_sessions").update({
                "last_active": "now()",
                "total_input_tokens": row["total_input_tokens"] + input_tokens,
                "total_output_tokens": row["total_output_tokens"] + output_tokens,
                "total_internal_tokens": row["total_internal_tokens"] + internal_tokens,
            }).eq("id", row["id"]).execute()
        else:
            client.table("chat_sessions").insert({
                "chat_id": chat_id,
                "session_id": session_id,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_internal_tokens": internal_tokens,
            }).execute()

    try:
        await asyncio.to_thread(_save)
    except Exception as e:
        logger.warning("db.save_message failed chat_id=%s: %s", chat_id, e)
