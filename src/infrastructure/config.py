"""Environment configuration and the factories that wire the hexagon together.

This is the only module allowed to know which concrete adapter exists. The graph
and the domain receive port instances and never learn the provider name, which
is what makes the whole system switchable through a single environment variable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..domain.policy import WorkflowPolicy
from ..domain.preprocessing import PreprocessingConfig
from ..ports.extractor_port import DocumentExtractorPort, ExtractionError
from ..ports.vlm_port import VLMProviderError, VLMProviderPort

LOGGER = logging.getLogger(__name__)


class ExtractorProvider(StrEnum):
    """Supported document extraction backends."""

    AZURE = "azure"
    AWS = "aws"
    GOOGLE = "google"
    NATIVE_VLM = "native_vlm"


class VLMProvider(StrEnum):
    """Supported vision language model backends."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Typed view over the process environment and the .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Provider selection -------------------------------------------------
    extractor_provider: ExtractorProvider = Field(
        default=ExtractorProvider.NATIVE_VLM,
        description="Which DocumentExtractorPort implementation to inject.",
    )
    vlm_provider: VLMProvider = Field(
        default=VLMProvider.OPENAI,
        description="Which VLMProviderPort implementation to inject.",
    )

    # --- Workflow behaviour -------------------------------------------------
    max_retries: int = Field(default=3, ge=0, le=10, description="Retry budget N.")
    log_level: str = Field(default="INFO")

    # --- Preprocessing ------------------------------------------------------
    blur_critical_threshold: float = Field(default=120.0, gt=0.0)
    blur_reference_width: int = Field(default=1000, gt=0)
    unsharp_amount: float = Field(default=0.8, ge=0.0, le=3.0)
    unsharp_sigma: float = Field(default=2.0, gt=0.0)
    clahe_clip_limit: float = Field(default=2.0, gt=0.0)
    clahe_tile_grid_size: int = Field(default=8, gt=0)
    max_image_dimension: int = Field(default=2000, gt=0)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    original_jpeg_quality: int = Field(default=95, ge=1, le=100)
    roi_padding_ratio: float = Field(default=0.10, ge=0.0, le=1.0)
    roi_min_width: int = Field(default=320, gt=0)
    roi_max_upscale: float = Field(default=4.0, ge=1.0)

    # --- Azure AI Document Intelligence ------------------------------------
    azure_document_intelligence_endpoint: Optional[str] = None
    azure_document_intelligence_key: Optional[str] = None
    azure_model_by_doc_type: dict[str, str] = Field(default_factory=dict)

    # --- AWS Textract -------------------------------------------------------
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None

    # --- Google Document AI -------------------------------------------------
    google_project_id: Optional[str] = None
    google_location: str = Field(default="us")
    google_processor_by_doc_type: dict[str, str] = Field(default_factory=dict)
    google_default_processor_id: Optional[str] = None
    google_processor_version: Optional[str] = None
    google_application_credentials: Optional[str] = None

    # --- OpenAI -------------------------------------------------------------
    openai_api_key: Optional[str] = None
    openai_model: str = Field(default="gpt-4o")
    openai_base_url: Optional[str] = None
    openai_organization: Optional[str] = None

    # --- Anthropic ----------------------------------------------------------
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = Field(default="claude-sonnet-5")
    anthropic_base_url: Optional[str] = None

    # --- Ollama -------------------------------------------------------------
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5vl:7b")
    ollama_keep_alive: str = Field(default="5m")

    # --- Shared VLM tuning --------------------------------------------------
    vlm_max_tokens: int = Field(default=1024, gt=0)
    vlm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    vlm_timeout_seconds: float = Field(default=60.0, gt=0.0)

    # --- Native VLM extractor ----------------------------------------------
    native_vlm_request_bounding_boxes: bool = Field(
        default=False,
        description="Ask the VLM to ground each field, enabling targeted ROI retries.",
    )
    native_vlm_layout_hints: dict[str, str] = Field(default_factory=dict)

    def to_workflow_policy(self) -> WorkflowPolicy:
        """Project the settings onto the domain policy consumed by the graph.

        This projection is the reason graph/nodes.py imports nothing from the
        infrastructure layer: the nodes receive a plain domain dataclass.
        """
        return WorkflowPolicy(
            max_retries=self.max_retries,
            preprocessing=self.to_preprocessing_config(),
        )

    def to_preprocessing_config(self) -> PreprocessingConfig:
        """Project the settings onto the domain preprocessing configuration.

        The domain owns its own configuration object so it never depends on
        pydantic-settings or on the environment.
        """
        return PreprocessingConfig(
            blur_critical_threshold=self.blur_critical_threshold,
            blur_reference_width=self.blur_reference_width,
            unsharp_amount=self.unsharp_amount,
            unsharp_sigma=self.unsharp_sigma,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid_size=self.clahe_tile_grid_size,
            max_dimension=self.max_image_dimension,
            jpeg_quality=self.jpeg_quality,
            original_jpeg_quality=self.original_jpeg_quality,
            roi_padding_ratio=self.roi_padding_ratio,
            roi_min_width=self.roi_min_width,
            roi_max_upscale=self.roi_max_upscale,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process wide settings singleton."""
    return Settings()


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def build_vlm_provider(settings: Optional[Settings] = None) -> VLMProviderPort:
    """Instantiate the configured VLMProviderPort implementation."""
    settings = settings or get_settings()

    if settings.vlm_provider is VLMProvider.OPENAI:
        from .adapters.vlm.openai_vlm import OpenAIVLMAdapter

        if not settings.openai_api_key:
            raise VLMProviderError("openai", "OPENAI_API_KEY is not configured.")
        return OpenAIVLMAdapter(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            organization=settings.openai_organization,
            max_tokens=settings.vlm_max_tokens,
            temperature=settings.vlm_temperature,
            timeout_seconds=settings.vlm_timeout_seconds,
        )

    if settings.vlm_provider is VLMProvider.ANTHROPIC:
        from .adapters.vlm.anthropic_vlm import AnthropicVLMAdapter

        if not settings.anthropic_api_key:
            raise VLMProviderError("anthropic", "ANTHROPIC_API_KEY is not configured.")
        return AnthropicVLMAdapter(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            base_url=settings.anthropic_base_url,
            max_tokens=settings.vlm_max_tokens,
            temperature=settings.vlm_temperature,
            timeout_seconds=settings.vlm_timeout_seconds,
        )

    if settings.vlm_provider is VLMProvider.OLLAMA:
        from .adapters.vlm.ollama_vlm import OllamaVLMAdapter

        return OllamaVLMAdapter(
            model=settings.ollama_model,
            host=settings.ollama_host,
            max_tokens=settings.vlm_max_tokens,
            temperature=settings.vlm_temperature,
            # Local inference is slower, so it gets a floor on the timeout.
            timeout_seconds=max(settings.vlm_timeout_seconds, 120.0),
            keep_alive=settings.ollama_keep_alive,
        )

    raise VLMProviderError(str(settings.vlm_provider), "Unsupported VLM provider.")


def build_extractor(
    settings: Optional[Settings] = None,
    vlm_provider: Optional[VLMProviderPort] = None,
) -> DocumentExtractorPort:
    """Instantiate the configured DocumentExtractorPort implementation.

    Args:
        settings: Configuration to read from, defaulting to the singleton.
        vlm_provider: Provider reused by the native VLM extractor. Passing the
            same instance the retry node uses avoids opening two HTTP clients
            against the same endpoint.
    """
    settings = settings or get_settings()

    if settings.extractor_provider is ExtractorProvider.AZURE:
        from .adapters.extractors.azure_document_intelligence import (
            AzureDocumentIntelligenceAdapter,
        )

        return AzureDocumentIntelligenceAdapter(
            endpoint=settings.azure_document_intelligence_endpoint or "",
            api_key=settings.azure_document_intelligence_key or "",
            model_by_doc_type=settings.azure_model_by_doc_type,
        )

    if settings.extractor_provider is ExtractorProvider.AWS:
        from .adapters.extractors.aws_textract import AWSTextractAdapter

        return AWSTextractAdapter(
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_session_token=settings.aws_session_token,
        )

    if settings.extractor_provider is ExtractorProvider.GOOGLE:
        from .adapters.extractors.google_docai import GoogleDocAIAdapter

        return GoogleDocAIAdapter(
            project_id=settings.google_project_id or "",
            location=settings.google_location,
            processor_by_doc_type=settings.google_processor_by_doc_type,
            default_processor_id=settings.google_default_processor_id,
            processor_version=settings.google_processor_version,
            credentials_path=settings.google_application_credentials,
        )

    if settings.extractor_provider is ExtractorProvider.NATIVE_VLM:
        from .adapters.extractors.native_vlm import NativeVLMExtractorAdapter

        return NativeVLMExtractorAdapter(
            vlm_provider=vlm_provider or build_vlm_provider(settings),
            request_bounding_boxes=settings.native_vlm_request_bounding_boxes,
            layout_hints_by_doc_type=settings.native_vlm_layout_hints,
        )

    raise ExtractionError(str(settings.extractor_provider), "Unsupported extractor provider.")


@dataclass(slots=True)
class Dependencies:
    """Resolved dependency container injected into the LangGraph workflow."""

    settings: Settings
    extractor: DocumentExtractorPort
    vlm_provider: VLMProviderPort

    def close(self) -> None:
        """Release every client held by the adapters."""
        self.extractor.close()
        self.vlm_provider.close()


def build_dependencies(
    settings: Optional[Settings] = None,
    *,
    extractor: Optional[DocumentExtractorPort] = None,
    vlm_provider: Optional[VLMProviderPort] = None,
) -> Dependencies:
    """Resolve every dependency the workflow needs.

    Explicit ``extractor`` and ``vlm_provider`` arguments take precedence over
    the configuration, which is how tests inject fakes without any environment
    variable or monkeypatching.
    """
    settings = settings or get_settings()
    resolved_vlm = vlm_provider or build_vlm_provider(settings)
    resolved_extractor = extractor or build_extractor(settings, vlm_provider=resolved_vlm)
    LOGGER.info(
        "Dependencies resolved: extractor=%s vlm=%s",
        resolved_extractor.provider_name,
        resolved_vlm.provider_name,
    )
    return Dependencies(
        settings=settings,
        extractor=resolved_extractor,
        vlm_provider=resolved_vlm,
    )
