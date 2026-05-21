"""
Step 2 — LLM Client Configuration

Provides:
- CortexSettings: pydantic-settings BaseSettings for credential management
- LLMManager: Adaptive instructor-patched client that works with any
  OpenAI-compatible endpoint (Snowflake Cortex, OpenAI, Azure OpenAI, Ollama, etc.)

Default: Snowflake Cortex REST API with Personal Access Token.
Override: Set LLM_PROVIDER + provider-specific env vars to use another backend.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import instructor
from openai import AzureOpenAI, OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.1 Credential Management
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    SNOWFLAKE_CORTEX = "snowflake_cortex"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"


class CortexSettings(BaseSettings):
    """Configuration loaded from environment variables.

    Missing required variables raise a clear startup error before pipeline execution.
    """

    # Provider selection
    llm_provider: LLMProvider = Field(
        default=LLMProvider.SNOWFLAKE_CORTEX,
        description="LLM provider: snowflake_cortex, openai, azure_openai, ollama",
    )

    # Snowflake credentials (required when provider=snowflake_cortex)
    snowflake_account: str = Field(default="", description="Snowflake account identifier")
    snowflake_user: str = Field(default="", description="Snowflake username")
    snowflake_password: str = Field(default="", description="Snowflake password (optional if using PAT)")
    snowflake_pat: str = Field(default="", description="Snowflake Personal Access Token")
    snowflake_role: str = Field(default="", description="Snowflake role")
    snowflake_warehouse: str = Field(default="", description="Snowflake warehouse")
    snowflake_database: str = Field(default="", description="Snowflake database")

    # OpenAI credentials (required when provider=openai)
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_base_url: str = Field(default="", description="Custom OpenAI-compatible base URL (optional)")

    # Azure OpenAI credentials (required when provider=azure_openai)
    azure_openai_endpoint: str = Field(default="", description="Azure OpenAI endpoint URL")
    azure_openai_api_key: str = Field(default="", description="Azure OpenAI API key")
    azure_openai_api_version: str = Field(default="2024-12-01-preview", description="Azure OpenAI API version")
    azure_openai_deployment_name: str = Field(default="", description="Azure OpenAI deployment name")

    # Ollama configuration (required when provider=ollama)
    ollama_base_url: str = Field(default="http://localhost:11434/v1", description="Ollama API base URL")

    # Model configuration
    routing_model: str = Field(
        default="snowflake-arctic-instruct",
        description="Model for routing and semantic profiling",
    )
    mapping_model: str = Field(
        default="mistral-large2",
        description="Model for field mapping and transformation extraction",
    )

    # Instructor retry configuration
    max_retries: int = Field(
        default=3,
        description="Max retries on Pydantic validation failure",
    )

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}

    @property
    def cortex_api_key(self) -> str:
        """Return PAT if available, otherwise password (for Snowflake Cortex)."""
        return self.snowflake_pat or self.snowflake_password

    @property
    def cortex_base_url(self) -> str:
        """Construct Snowflake Cortex OpenAI-compatible endpoint URL."""
        return f"https://{self.snowflake_account}.snowflakecomputing.com/api/v2/cortex/v1"


# ---------------------------------------------------------------------------
# 2.2 Adaptive LLM Manager
# ---------------------------------------------------------------------------


class LLMManager:
    """Adaptive LLM client that works with any OpenAI-compatible provider.

    Wraps instructor for structured output enforcement with automatic retry.
    Supports Snowflake Cortex, OpenAI, Azure OpenAI, Ollama, or any custom endpoint.

    Two logical clients:
    - routing: For semantic profiling and table routing (lighter model)
    - mapping: For field mapping and transformation extraction (stronger model)

    For Azure OpenAI: routing_model and mapping_model should be set to the
    deployment name (AZURE_OPENAI_DEPLOYMENT_NAME) unless you have separate
    deployments for each.
    """

    def __init__(self, settings: CortexSettings | None = None):
        self.settings = settings or CortexSettings()
        self._client: Any = None

        logger.info(
            f"LLMManager initialized | "
            f"provider={self.settings.llm_provider.value} | "
            f"routing_model={self.settings.routing_model} | "
            f"mapping_model={self.settings.mapping_model}"
        )

    def _create_base_client(self) -> OpenAI | AzureOpenAI:
        """Create a base client based on the configured provider."""
        provider = self.settings.llm_provider

        if provider == LLMProvider.SNOWFLAKE_CORTEX:
            if not self.settings.snowflake_account:
                raise ValueError("SNOWFLAKE_ACCOUNT is required for Snowflake Cortex provider")
            return OpenAI(
                api_key=self.settings.cortex_api_key,
                base_url=self.settings.cortex_base_url,
            )

        elif provider == LLMProvider.OPENAI:
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
            kwargs: dict[str, Any] = {"api_key": self.settings.openai_api_key}
            if self.settings.openai_base_url:
                kwargs["base_url"] = self.settings.openai_base_url
            return OpenAI(**kwargs)

        elif provider == LLMProvider.AZURE_OPENAI:
            if not self.settings.azure_openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI provider")
            if not self.settings.azure_openai_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY is required for Azure OpenAI provider")
            return AzureOpenAI(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
            )

        elif provider == LLMProvider.OLLAMA:
            return OpenAI(
                api_key="ollama",
                base_url=self.settings.ollama_base_url,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @property
    def client(self) -> Any:
        """Lazily initialize the instructor-patched client."""
        if self._client is None:
            base = self._create_base_client()
            provider = self.settings.llm_provider

            if provider == LLMProvider.OLLAMA:
                self._client = instructor.from_openai(base, mode=instructor.Mode.JSON)
            else:
                self._client = instructor.from_openai(base)

            logger.info(f"Instructor client created | provider={provider.value}")
        return self._client

    @property
    def _effective_routing_model(self) -> str:
        """Return the model name to use for routing calls.

        For Azure OpenAI, falls back to deployment name if routing_model not set.
        """
        if self.settings.llm_provider == LLMProvider.AZURE_OPENAI:
            return self.settings.routing_model or self.settings.azure_openai_deployment_name
        return self.settings.routing_model

    @property
    def _effective_mapping_model(self) -> str:
        """Return the model name to use for mapping calls.

        For Azure OpenAI, falls back to deployment name if mapping_model not set.
        """
        if self.settings.llm_provider == LLMProvider.AZURE_OPENAI:
            return self.settings.mapping_model or self.settings.azure_openai_deployment_name
        return self.settings.mapping_model

    def call_routing(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> Any:
        """Make an instructor-enforced LLM call using the routing model.

        Used by: build_semantic_profiles, route_tables

        Args:
            messages: Chat messages (system + user)
            response_model: Pydantic model class for structured output
            temperature: Sampling temperature (low for factual extraction)
            **kwargs: Additional parameters passed to the API

        Returns:
            Validated Pydantic model instance
        """
        return self.client.chat.completions.create(
            model=self._effective_routing_model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=self.settings.max_retries,
            **kwargs,
        )

    def call_mapping(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.3,
        **kwargs: Any,
    ) -> Any:
        """Make an instructor-enforced LLM call using the mapping model.

        Used by: map_fields, derive_transformations

        Args:
            messages: Chat messages (system + user)
            response_model: Pydantic model class for structured output
            temperature: Slightly higher for nuanced semantic reasoning
            **kwargs: Additional parameters passed to the API

        Returns:
            Validated Pydantic model instance
        """
        return self.client.chat.completions.create(
            model=self._effective_mapping_model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=self.settings.max_retries,
            **kwargs,
        )

    def call(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Any:
        """Generic call with explicit model override.

        Use when you need a model that isn't routing or mapping.

        Args:
            model: Model name to use for this call
            messages: Chat messages
            response_model: Pydantic model class for structured output
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Validated Pydantic model instance
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            response_model=response_model,
            temperature=temperature,
            max_retries=self.settings.max_retries,
            **kwargs,
        )
