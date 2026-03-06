from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from src.log.logs import logger

try:
    from supabase import Client, create_client
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
    Client = Any  # type: ignore[assignment,misc]
    logger.warning("supabase package not installed — DB persistence disabled. pip install supabase")

from src.config import settings


_CLIENT: Optional[Client] = None
_PENDING_SAVE_TASKS: Set[asyncio.Task] = set()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client() -> Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = create_client(settings.supabase_url, settings.supabase_key)
    return _CLIENT


def _ready() -> bool:
    return _AVAILABLE and bool(settings.supabase_url) and bool(settings.supabase_key)


def persistence_status() -> Dict[str, Any]:
    if not _AVAILABLE:
        return {"enabled": False, "reason": "supabase_package_missing"}
    if not settings.supabase_url:
        return {"enabled": False, "reason": "missing_SUPABASE_URL"}
    if not settings.supabase_key:
        return {"enabled": False, "reason": "missing_SUPABASE_KEY"}
    return {"enabled": True, "reason": "ready"}


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
            .limit(1)
            .execute()
        )

        if existing.data:
            row = existing.data[0]
            prev_input = int(row.get("total_input_tokens") or 0)
            prev_output = int(row.get("total_output_tokens") or 0)
            prev_internal = int(row.get("total_internal_tokens") or 0)
            client.table("chat_sessions").update({
                "last_active": _utc_now_iso(),
                "total_input_tokens": prev_input + input_tokens,
                "total_output_tokens": prev_output + output_tokens,
                "total_internal_tokens": prev_internal + internal_tokens,
            }).eq("id", row["id"]).execute()
        else:
            client.table("chat_sessions").insert({
                "chat_id": chat_id,
                "session_id": session_id,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "total_internal_tokens": internal_tokens,
                "last_active": _utc_now_iso(),
            }).execute()

    try:
        await asyncio.to_thread(_save)
    except Exception as e:
        logger.warning("db.save_message failed chat_id=%s: %s", chat_id, e)


def schedule_save_message(
    chat_id: str,
    session_id: str,
    human: str,
    ai: str,
    input_tokens: int,
    output_tokens: int,
    internal_tokens: int,
) -> bool:
    if not _ready():
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False

    task = loop.create_task(
        save_message(
            chat_id=chat_id,
            session_id=session_id,
            human=human,
            ai=ai,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            internal_tokens=internal_tokens,
        )
    )
    _PENDING_SAVE_TASKS.add(task)

    def _done(t: asyncio.Task) -> None:
        _PENDING_SAVE_TASKS.discard(t)
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc:
            logger.warning("db.background_save failed chat_id=%s: %s", chat_id, exc)

    task.add_done_callback(_done)
    return True


async def flush_pending_writes(timeout_sec: float = 3.0) -> int:
    if not _PENDING_SAVE_TASKS:
        return 0

    tasks = list(_PENDING_SAVE_TASKS)
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning("db.flush_pending_writes timed out pending=%d", len(_PENDING_SAVE_TASKS))
    return len(tasks)
