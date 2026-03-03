from functools import lru_cache
from typing import Optional

from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from log.logs import logger

class LLMClient:
    def __init__(self):
        self._llm: Optional[ChatGoogleGenerativeAI] = None
    
    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or create LLM instance"""
        if self._llm is None:
            logger.info("Initializing LLM client model=%s", settings.LLM_MODEL)
            self._llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                max_tokens=settings.LLM_MAX_TOKENS,
                api_key=settings.google_api_key,
                temperature=settings.LLM_TEMPERATURE,
            )

        return self._llm
        
llm_client = LLMClient()

@lru_cache(maxsize=3)
def get_llm() -> ChatGoogleGenerativeAI:
    llm = llm_client.get_llm()
    return llm
