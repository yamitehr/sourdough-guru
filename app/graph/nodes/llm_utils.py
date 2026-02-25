"""Shared LLM utilities for all nodes."""

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger("sourdough.llm")


@lru_cache
def get_llm() -> ChatOpenAI:
    """Get a shared ChatOpenAI instance configured for LLMod.ai."""
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.LLMOD_API_KEY,
        base_url=settings.LLMOD_BASE_URL,
        model=settings.LLMOD_CHAT_MODEL,
        max_tokens=8000,
    )
