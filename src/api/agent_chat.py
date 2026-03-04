from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.agentic_rag.graph import run_agent_chat
from src.log.logs import logger

router = APIRouter()


class _Config(BaseModel):
    chat_id: str
    session_id: str


class AgentChatRequest(BaseModel):
    message: str
    top_k: int = 3
    images: bool = True
    include_text: bool = False
    text_search: bool = True
    config: _Config


class AgentChatResponse(BaseModel):
    answer: str
    metadata: Dict[str, Any]


@router.post("/agent_chat", response_model=AgentChatResponse)
async def agent_chat_endpoint(body: AgentChatRequest):
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required.")
    if body.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0.")

    logger.info(
        "agent_chat request chat_id=%s session_id=%s top_k=%d images=%s include_text=%s text_search=%s",
        body.config.chat_id,
        body.config.session_id,
        body.top_k,
        body.images,
        body.include_text,
        body.text_search,
    )

    try:
        return await run_agent_chat(
            user_input=msg,
            top_k=body.top_k,
            images=body.images,
            include_text=body.include_text,
            text_search=body.text_search,
            chat_id=body.config.chat_id,
            session_id=body.config.session_id,
        )
    except Exception as e:
        logger.exception("agent_chat failed chat_id=%s", body.config.chat_id)
        raise HTTPException(status_code=500, detail=str(e))
