from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLMod.ai (OpenAI-compatible)
    LLMOD_API_KEY: str
    LLMOD_BASE_URL: str = "https://api.llmod.ai/v1"
    LLMOD_CHAT_MODEL: str = "RPRTHPB-gpt-5-mini"
    LLMOD_EMBEDDING_MODEL: str = "RPRTHPB-text-embedding-3-small"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "sourdough-knowledge"

    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
