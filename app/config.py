from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
    )

    embedding_model: str = "BAAI/bge-m3"
    vector_dim: int = 1024
    top_k: int = 5
    hybrid_alpha: float = 0.7
    chunk_size: int = 1200
    chunk_overlap: int = 150
    prompt_cache_ttl_seconds: int = 900
    prompt_cache_max_size: int = 1000
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Chat LLM: OpenAI by default; Groq uses GROQ_API_KEY + LLM_BASE_URL per Groq docs.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    groq_api_key: str = ""
    # chat_completions (default) or responses — Groq OSS examples use Responses API:
    # https://console.groq.com/docs/responses-api
    llm_route: str = "chat_completions"

    # PDF / multimodal ingestion
    pdf_max_size_mb: float = 25.0
    pdf_ocr_enabled: bool = True
    pdf_ocr_min_text_chars: int = 50
    pdf_ocr_images: bool = True
    pdf_ocr_zoom: float = 2.0

    # Comma-separated origins for browser UI (React dev server, deployed SPA).
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
