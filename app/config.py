"""
app/config.py — Centralized Configuration
-------------------------------------------
Single source of truth for all settings. Every module imports from here
instead of reading os.getenv() or hardcoding values.

Usage:
    from app.config import settings
    settings.GROQ_API_KEY
    settings.LLM_MODEL
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application-wide configuration loaded from environment variables."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    VISION_MODEL: str = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    # ── API Keys ─────────────────────────────────────────────────────────────
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ── Security ─────────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-this-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
    MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
    MAX_INPUT_TOKENS: int = int(os.getenv("MAX_INPUT_TOKENS", "4000"))

    # ── Agent ────────────────────────────────────────────────────────────────
    AGENT_TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
    AGENT_MAX_STEPS: int = int(os.getenv("AGENT_MAX_STEPS", "10"))
    MAX_AGENT_STEPS: int = AGENT_MAX_STEPS
    AGENT_MAX_RETRIES: int = int(os.getenv("AGENT_MAX_RETRIES", "3"))
    AGENT_MAX_TOOL_CALLS: int = int(os.getenv("AGENT_MAX_TOOL_CALLS", "8"))

    # ── RAG ──────────────────────────────────────────────────────────────────
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma_db")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "8"))
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "3"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "5"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # ── Cache ────────────────────────────────────────────────────────────────
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    CACHE_MAX_ENTRIES: int = int(os.getenv("CACHE_MAX_ENTRIES", "200"))

    # ── Memory ───────────────────────────────────────────────────────────────
    MEMORY_STORE_PATH: str = os.getenv("MEMORY_STORE_PATH", "memory_store.json")
    MEMORY_MAX_INJECTION_TOKENS: int = int(os.getenv("MEMORY_MAX_INJECTION_TOKENS", "300"))

    # ── Monitoring ───────────────────────────────────────────────────────────
    METRICS_DIR: str = os.getenv("METRICS_DIR", "metrics")
    MONITOR_LOG_PATH: str = os.getenv("MONITOR_LOG_PATH", "monitor_log.json")

    # ── Python Execution ─────────────────────────────────────────────────────
    PYTHON_EXEC_TIMEOUT: int = int(os.getenv("PYTHON_EXEC_TIMEOUT", "5"))
    PYTHON_EXEC_MAX_OUTPUT: int = int(os.getenv("PYTHON_EXEC_MAX_OUTPUT", "4000"))

    # ── Paths ────────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> list[str]:
        """Check required settings and return list of warnings."""
        warnings = []
        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY not set — LLM calls will fail")
        if self.JWT_SECRET == "change-this-in-production":
            warnings.append("JWT_SECRET is default — change in production")
        if not self.TAVILY_API_KEY:
            warnings.append("TAVILY_API_KEY not set — web search will fallback to DuckDuckGo")
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()


def get_llm_client():
    """Create and return a Groq LLM client using centralized config."""
    from groq import Groq
    return Groq(api_key=settings.GROQ_API_KEY)


def call_llm(
    prompt: str,
    system_prompt: str = "You are a secure, helpful AI assistant.",
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Centralized LLM call — every module uses this instead of creating
    its own Groq client and hardcoding model names.

    Args:
        prompt: User message
        system_prompt: System message
        model: Override model (default: settings.LLM_MODEL)
        temperature: Override temperature (default: settings.LLM_TEMPERATURE)
        max_tokens: Override max tokens (default: settings.LLM_MAX_TOKENS)

    Returns:
        LLM response text (stripped)
    """
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature if temperature is not None else settings.LLM_TEMPERATURE,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def call_llm_with_retry(
    prompt: str,
    system_prompt: str = "You are a secure, helpful AI assistant.",
    retries: int = 3,
    wait_seconds: int = 10,
    **kwargs,
) -> str:
    """LLM call with retry on rate-limit (429) errors."""
    import time

    for attempt in range(retries):
        try:
            return call_llm(prompt, system_prompt, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                logging.warning(
                    f"⏳ Rate limit — waiting {wait_seconds}s (attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait_seconds)
            else:
                raise
    # Should never reach here, but just in case
    return call_llm(prompt, system_prompt, **kwargs)


# ── Configure logging on import ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
