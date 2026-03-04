from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

_STORE: Dict[str, Dict[str, Any]] = {}
_TTL = 3600.0
_MAX_TURNS = 20


def get(chat_id: str) -> Optional[List[Dict[str, Any]]]:
    entry = _STORE.get(chat_id)
    if not entry:
        return None
    if time.monotonic() - entry["ts"] > _TTL:
        _STORE.pop(chat_id, None)
        return None
    entry["ts"] = time.monotonic()
    return list(entry["turns"])


def put(chat_id: str, turns: List[Dict[str, Any]]) -> None:
    _STORE[chat_id] = {"turns": list(turns[-_MAX_TURNS:]), "ts": time.monotonic()}
