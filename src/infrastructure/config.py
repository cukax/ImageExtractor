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
from typing import Any, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..domain.policy import DEFAULT_STRATEGY_BY_DOC_TYPE, WorkflowPolicy
from ..domain.preprocessing import PreprocessingConfig
from ..ports.extractor_port import DocumentExtractorPort, ExtractionError
from ..ports.pdf_port import PDFProcessingError, PDFProcessorPort
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
    AZURE_FOUNDRY = "azure_foundry"


class PDFProcessor(StrEnum):
    """Supported PDF rasterization backends."""

    PYMUPDF = "pymupdf"


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
    pdf_processor: PDFProcessor = Field(
        default=PDFProcessor.PYMUPDF,
        description="Which PDFProcessorPort implementation to inject.",
    )

    # --- Workflow behaviour -------------------------------------------------
    max_retries: int = Field(default=3, ge=0, le=10, description="Retry budget N.")
    log_level: str = Field(default="INFO")

    # --- Dossier engine -----------------------------------------------------
    pdf_render_dpi: int = Field(
        default=300,
        ge=72,
        le=1200,
        description="Resolution dossier pages are rasterized at.",
    )
    classification_max_dimension: int = Field(
        default=768,
        gt=0,
        description="Longest side of the low resolution copy sent to the page classifier.",
    )
    strategy_by_doc_type: dict[str, str] = Field(
        default_factory=dict,
        description="Overrides of the worker routing table, keyed by logical document type.",
    )
    grounding_extractor_provider: Optional[ExtractorProvider] = Field(
        default=None,
        description=(
            "Extractor serving the visual grounding branch of the dossier worker. "
            "Defaults to the native VLM extractor, which is what makes a dossier able "
            "to mix a prebuilt OCR model with a vision deployment."
        ),
    )

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

    # --- Targeted ROI enhancement (dossier worker retry) --------------------
    roi_upscale_factor: float = Field(default=2.5, ge=1.0, le=8.0)
    roi_denoise_strength: float = Field(default=7.0, ge=0.0, le=30.0)
    roi_denoise_color_strength: float = Field(default=7.0, ge=0.0, le=30.0)
    roi_clahe_clip_limit: float = Field(default=3.0, gt=0.0)
    roi_clahe_tile_grid_size: int = Field(default=8, gt=0)
    roi_enhance_unsharp_amount: float = Field(default=1.0, ge=0.0, le=3.0)
    roi_enhance_unsharp_sigma: float = Field(default=1.5, gt=0.0)

    # --- Azure AI Document Intelligence ------------------------------------
    azure_document_intelligence_endpoint: Optional[str] = None
    azure_document_intelligence_key: Optional[str] = Field(
        default=None,
        description="Leave empty to authenticate with Microsoft Entra ID / Managed Identity.",
    )
    azure_model_by_doc_type: dict[str, str] = Field(default_factory=dict)
    azure_credential_scope: str = Field(
        default="https://cognitiveservices.azure.com/.default",
        description="Token audience used when authenticating with Entra ID.",
    )

    # --- Microsoft AI Foundry ----------------------------------------------
    azure_foundry_endpoint: Optional[str] = None
    azure_foundry_deployment: str = Field(default="gpt-4o")
    azure_foundry_classifier_deployment: str = Field(
        default="gpt-4o-mini",
        description="Cheaper deployment used for the per-page classification calls.",
    )
    azure_foundry_api_version: str = Field(default="2024-10-21")
    azure_foundry_api_key: Optional[str] = Field(
        default=None,
        description="Leave empty to authenticate with Microsoft Entra ID / Managed Identity.",
    )

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

    @field_validator("grounding_extractor_provider", mode="before")
    @classmethod
    def _empty_means_default(cls, raw_value: Any) -> Any:
        """Treat an empty environment variable as "not set".

        A commented-out or blank line in a .env file yields an empty string, not
        an absent key. Every other optional setting here is a string and shrugs
        that off, but an enum would reject it, so a documented blank default
        would fail to start.
        """
        if isinstance(raw_value, str) and not raw_value.strip():
            return None
        return raw_value

    def to_workflow_policy(self) -> WorkflowPolicy:
        """Project the settings onto the domain policy consumed by the graph.

        This projection is the reason graph/nodes.py imports nothing from the
        infrastructure layer: the nodes receive a plain domain dataclass.
        """
        return WorkflowPolicy(
            max_retries=self.max_retries,
            preprocessing=self.to_preprocessing_config(),
            pdf_render_dpi=self.pdf_render_dpi,
            classification_max_dimension=self.classification_max_dimension,
            # Overrides are merged onto the defaults rather than replacing them,
            # so moving one document type between branches does not require
            # restating the whole table.
            strategy_by_doc_type={
                **DEFAULT_STRATEGY_BY_DOC_TYPE,
                **self.strategy_by_doc_type,
            },
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
            roi_upscale_factor=self.roi_upscale_factor,
            roi_denoise_strength=self.roi_denoise_strength,
            roi_denoise_color_strength=self.roi_denoise_color_strength,
            roi_clahe_clip_limit=self.roi_clahe_clip_limit,
            roi_clahe_tile_grid_size=self.roi_clahe_tile_grid_size,
            roi_enhance_unsharp_amount=self.roi_enhance_unsharp_amount,
            roi_enhance_unsharp_sigma=self.roi_enhance_unsharp_sigma,
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

    if settings.vlm_provider is VLMProvider.AZURE_FOUNDRY:
        from .adapters.vlm.azure_foundry_vlm import AzureAIFoundryVLMAdapter

        if not settings.azure_foundry_endpoint:
            raise VLMProviderError("azure_foundry", "AZURE_FOUNDRY_ENDPOINT is not configured.")
        # No key check here on purpose: an empty key is the *preferred*
        # configuration, since it selects Managed Identity authentication.
        return AzureAIFoundryVLMAdapter(
            endpoint=settings.azure_foundry_endpoint,
            deployment=settings.azure_foundry_deployment,
            api_key=settings.azure_foundry_api_key,
            api_version=settings.azure_foundry_api_version,
            credential_scope=settings.azure_credential_scope,
            classifier_deployment=settings.azure_foundry_classifier_deployment,
            max_tokens=settings.vlm_max_tokens,
            temperature=settings.vlm_temperature,
            timeout_seconds=settings.vlm_timeout_seconds,
        )

    raise VLMProviderError(str(settings.vlm_provider), "Unsupported VLM provider.")


def build_pdf_processor(settings: Optional[Settings] = None) -> PDFProcessorPort:
    """Instantiate the configured PDFProcessorPort implementation."""
    settings = settings or get_settings()

    if settings.pdf_processor is PDFProcessor.PYMUPDF:
        from .adapters.processors.pymupdf_processor import PyMuPDFAdapter

        return PyMuPDFAdapter(default_dpi=settings.pdf_render_dpi)

    raise PDFProcessingError(str(settings.pdf_processor), "Unsupported PDF processor.")


def build_extractor(
    settings: Optional[Settings] = None,
    vlm_provider: Optional[VLMProviderPort] = None,
    *,
    provider: Optional[ExtractorProvider] = None,
) -> DocumentExtractorPort:
    """Instantiate a DocumentExtractorPort implementation.

    Args:
        settings: Configuration to read from, defaulting to the singleton.
        vlm_provider: Provider reused by the native VLM extractor. Passing the
            same instance the retry node uses avoids opening two HTTP clients
            against the same endpoint.
        provider: Explicit backend, overriding ``settings.extractor_provider``.
            The dossier composition root uses it to build the second, grounding
            capable extractor without a second Settings object.
    """
    settings = settings or get_settings()
    provider = provider or settings.extractor_provider

    if provider is ExtractorProvider.AZURE:
        from .adapters.extractors.azure_document_intelligence import (
            AzureDocumentIntelligenceAdapter,
        )

        return AzureDocumentIntelligenceAdapter(
            endpoint=settings.azure_document_intelligence_endpoint or "",
            # An empty key selects Managed Identity, which is the preferred
            # enterprise configuration, so it is passed through untouched.
            api_key=settings.azure_document_intelligence_key,
            model_by_doc_type=settings.azure_model_by_doc_type,
            credential_scope=settings.azure_credential_scope,
        )

    if provider is ExtractorProvider.AWS:
        from .adapters.extractors.aws_textract import AWSTextractAdapter

        return AWSTextractAdapter(
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            aws_session_token=settings.aws_session_token,
        )

    if provider is ExtractorProvider.GOOGLE:
        from .adapters.extractors.google_docai import GoogleDocAIAdapter

        return GoogleDocAIAdapter(
            project_id=settings.google_project_id or "",
            location=settings.google_location,
            processor_by_doc_type=settings.google_processor_by_doc_type,
            default_processor_id=settings.google_default_processor_id,
            processor_version=settings.google_processor_version,
            credentials_path=settings.google_application_credentials,
        )

    if provider is ExtractorProvider.NATIVE_VLM:
        from .adapters.extractors.native_vlm import NativeVLMExtractorAdapter

        return NativeVLMExtractorAdapter(
            vlm_provider=vlm_provider or build_vlm_provider(settings),
            request_bounding_boxes=settings.native_vlm_request_bounding_boxes,
            layout_hints_by_doc_type=settings.native_vlm_layout_hints,
        )

    raise ExtractionError(str(provider), "Unsupported extractor provider.")


def build_grounding_extractor(
    settings: Optional[Settings] = None,
    vlm_provider: Optional[VLMProviderPort] = None,
) -> DocumentExtractorPort:
    """Instantiate the extractor serving the dossier's visual grounding branch.

    Defaults to the native VLM extractor with grounding forced on. Grounding is
    optional for the single document workflow, where an ungrounded retry merely
    costs more, but it is the point of this branch: the spec requires field
    values *and* their bounding boxes in one pass.
    """
    settings = settings or get_settings()
    provider = settings.grounding_extractor_provider or ExtractorProvider.NATIVE_VLM

    if provider is ExtractorProvider.NATIVE_VLM:
        from .adapters.extractors.native_vlm import NativeVLMExtractorAdapter

        return NativeVLMExtractorAdapter(
            vlm_provider=vlm_provider or build_vlm_provider(settings),
            request_bounding_boxes=True,
            layout_hints_by_doc_type=settings.native_vlm_layout_hints,
        )

    return build_extractor(settings, vlm_provider=vlm_provider, provider=provider)


@dataclass(slots=True)
class Dependencies:
    """Resolved dependency container injected into the LangGraph workflows."""

    settings: Settings
    extractor: DocumentExtractorPort
    """Template based extractor. Also serves the single document workflow."""

    vlm_provider: VLMProviderPort
    """Vision model used for ROI retries and for page classification."""

    grounding_extractor: DocumentExtractorPort
    """Grounding capable extractor serving the dossier's second branch."""

    pdf_processor: PDFProcessorPort
    """Rasterizer used by the dossier engine."""

    def close(self) -> None:
        """Release every client held by the adapters, each independently.

        A failure to close one adapter must not leak the others, which is why
        the calls are not chained. Adapters are de-duplicated by identity: the
        two extractors are frequently the same object, and the native VLM
        extractor closes the provider it borrowed.
        """
        seen: set[int] = set()
        for resource in (
            self.extractor,
            self.grounding_extractor,
            self.pdf_processor,
            self.vlm_provider,
        ):
            if id(resource) in seen:
                continue
            seen.add(id(resource))
            try:
                resource.close()
            except Exception as error:  # noqa: BLE001 - teardown must never raise
                LOGGER.warning("Could not close %s: %s", type(resource).__name__, error)


def build_dependencies(
    settings: Optional[Settings] = None,
    *,
    extractor: Optional[DocumentExtractorPort] = None,
    vlm_provider: Optional[VLMProviderPort] = None,
    grounding_extractor: Optional[DocumentExtractorPort] = None,
    pdf_processor: Optional[PDFProcessorPort] = None,
) -> Dependencies:
    """Resolve every dependency the workflows need.

    Explicit port arguments take precedence over the configuration, which is how
    tests inject fakes without any environment variable or monkeypatching.

    Nothing here opens a connection: every adapter builds its SDK client lazily,
    so resolving a PDF processor a single document run will never use costs one
    object allocation.
    """
    settings = settings or get_settings()
    resolved_vlm = vlm_provider or build_vlm_provider(settings)
    resolved_extractor = extractor or build_extractor(settings, vlm_provider=resolved_vlm)
    resolved_grounding = grounding_extractor or (
        # When the configured extractor is already the grounding capable native
        # VLM one, reuse it instead of opening a second client against the same
        # endpoint.
        resolved_extractor
        if settings.extractor_provider is ExtractorProvider.NATIVE_VLM
        and settings.native_vlm_request_bounding_boxes
        and settings.grounding_extractor_provider is None
        else build_grounding_extractor(settings, vlm_provider=resolved_vlm)
    )
    resolved_pdf_processor = pdf_processor or build_pdf_processor(settings)

    LOGGER.info(
        "Dependencies resolved: extractor=%s grounding=%s vlm=%s pdf=%s",
        resolved_extractor.provider_name,
        resolved_grounding.provider_name,
        resolved_vlm.provider_name,
        resolved_pdf_processor.provider_name,
    )
    return Dependencies(
        settings=settings,
        extractor=resolved_extractor,
        vlm_provider=resolved_vlm,
        grounding_extractor=resolved_grounding,
        pdf_processor=resolved_pdf_processor,
    )
