"""
LLM Client — Multi-provider with auto-detection.

Supports OpenAI, Groq, Azure OpenAI, Snowflake Cortex, and Ollama via a single
OpenAI-compatible client wrapped with Instructor for structured output.

Provider is auto-detected from whichever credentials are present in the
environment, with explicit override via LLM_PROVIDER.

Detection priority (when LLM_PROVIDER is unset):
    openai_api_key → openai
    groq_api_key → groq
    azure_openai_api_key + azure_openai_endpoint → azure
    snowflake_account + (snowflake_pat | snowflake_password) → snowflake
    fallback → ollama (localhost)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import instructor
from openai import AzureOpenAI, OpenAI
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

Provider = Literal["openai", "groq", "azure", "snowflake", "ollama"]


class LLMSettings(BaseSettings):
    """Credentials and model config loaded from .env / environment.

    Set whichever provider's credentials you have; the manager auto-detects.
    Override detection by setting LLM_PROVIDER explicitly.
    """

    # Optional explicit override
    llm_provider: Provider | None = None

    # Per-provider credentials — set whichever you use
    openai_api_key: str = ""
    # openai_base_url: str = ""  # Optional custom OpenAI-compatible endpoint

    groq_api_key: str = ""

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"

    snowflake_account: str = ""
    snowflake_pat: str = ""
    snowflake_password: str = ""

    ollama_base_url: str = "http://localhost:11434/v1"

    # Models — used by call_routing / call_mapping
    routing_model: str = "snowflake-arctic-instruct"
    mapping_model: str = "mistral-large2"

    # Instructor retry budget
    max_retries: int = 3

    model_config = {"env_file": ".env", "extra": "ignore"}

    def detect_provider(self) -> Provider:
        """Return the active provider, auto-detecting from credentials if not set."""
        if self.llm_provider:
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.groq_api_key:
            return "groq"
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            return "azure"
        if self.snowflake_account and (self.snowflake_pat or self.snowflake_password):
            return "snowflake"
        return "ollama"


def _make_base_client(s: LLMSettings, provider: Provider) -> OpenAI | AzureOpenAI:
    """Build the raw OpenAI-compatible client for the given provider."""
    if provider == "openai":
        return OpenAI(
            api_key=s.openai_api_key,
            # base_url=s.openai_base_url or None,
        )
    if provider == "groq":
        return OpenAI(
            api_key=s.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    if provider == "azure":
        return AzureOpenAI(
            api_key=s.azure_openai_api_key,
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
        )
    if provider == "snowflake":
        return OpenAI(
            api_key=s.snowflake_pat or s.snowflake_password,
            base_url=f"https://{s.snowflake_account}.snowflakecomputing.com/api/v2/cortex/v1",
        )
    if provider == "ollama":
        return OpenAI(api_key="ollama", base_url=s.ollama_base_url)
    raise ValueError(f"Unsupported provider: {provider}")


class LLMManager:
    """Instructor-wrapped multi-provider LLM client.

    Auto-detects the provider from which credentials are present and exposes
    a single `call()` method with structured-output enforcement via Instructor.
    `call_routing` and `call_mapping` are convenience wrappers that pre-fill
    the configured routing/mapping model and a sensible temperature.
    """

    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or LLMSettings()
        self.provider: Provider = self.settings.detect_provider()

        base = _make_base_client(self.settings, self.provider)
        # Ollama needs JSON mode (no native function/tool calling)
        if self.provider == "ollama":
            self.client: Any = instructor.from_openai(base, mode=instructor.Mode.JSON)
        else:
            self.client = instructor.from_openai(base)

        logger.info(
            f"LLMManager initialized | provider={self.provider} | "
            f"routing={self.settings.routing_model} | mapping={self.settings.mapping_model}"
        )

    def call(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type,
        model: str | None = None,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Any:
        """Make an Instructor-enforced LLM call.

        Args:
            messages: Chat messages (system + user).
            response_model: Pydantic model class for structured output.
            model: Override model name. Defaults to mapping_model.
            temperature: Sampling temperature.
            **kwargs: Passed through to the underlying API.
        """
        return self.client.chat.completions.create(
            model=model or self.settings.mapping_model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=self.settings.max_retries,
            **kwargs,
        )

    def call_routing(self, **kwargs: Any) -> Any:
        """Convenience wrapper using routing_model + low temperature (0.1)."""
        kwargs.setdefault("temperature", 0.1)
        return self.call(model=self.settings.routing_model, **kwargs)

    def call_mapping(self, **kwargs: Any) -> Any:
        """Convenience wrapper using mapping_model + medium temperature (0.3)."""
        kwargs.setdefault("temperature", 0.3)
        return self.call(model=self.settings.mapping_model, **kwargs)
