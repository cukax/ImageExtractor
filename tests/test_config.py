"""Tests for the composition root: settings parsing and the port factories."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.orchestrator_state import LogicalDocType
from src.domain.worker_state import ExtractionStrategy
from src.infrastructure.config import (
    Dependencies,
    ExtractorProvider,
    PDFProcessor,
    Settings,
    VLMProvider,
    build_dependencies,
    build_pdf_processor,
    build_vlm_provider,
)
from src.ports.vlm_port import VLMProviderError

from .conftest import FakeExtractorAdapter, FakePDFProcessorAdapter, FakeVLMAdapter

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


# --------------------------------------------------------------------------- #
# The documented configuration must actually start the system
# --------------------------------------------------------------------------- #
def test_the_shipped_env_example_parses() -> None:
    # A .env.example that does not load is worse than none at all: it is the
    # file every new deployment copies.
    settings = Settings(_env_file=ENV_EXAMPLE)

    assert settings.pdf_processor is PDFProcessor.PYMUPDF
    assert settings.pdf_render_dpi == 300


def test_a_blank_optional_enum_means_not_set() -> None:
    # A commented out or blank line in a .env file yields an empty string, not
    # an absent key, and an enum would otherwise reject it.
    settings = Settings(grounding_extractor_provider="", _env_file=None)

    assert settings.grounding_extractor_provider is None


# --------------------------------------------------------------------------- #
# Settings to domain projection
# --------------------------------------------------------------------------- #
def test_the_routing_overrides_are_merged_onto_the_defaults() -> None:
    settings = Settings(
        strategy_by_doc_type={
            LogicalDocType.INVOICE.value: ExtractionStrategy.DOC_INTELLIGENCE.value
        },
        _env_file=None,
    )

    policy = settings.to_workflow_policy()

    # Moving one document type must not require restating the whole table.
    assert policy.strategy_for(LogicalDocType.INVOICE) == ExtractionStrategy.DOC_INTELLIGENCE
    assert policy.strategy_for(LogicalDocType.PASSPORT) == ExtractionStrategy.DOC_INTELLIGENCE
    assert policy.strategy_for(LogicalDocType.PROOF_OF_ADDRESS) == ExtractionStrategy.VLM_GROUNDING


def test_the_roi_enhancement_knobs_reach_the_domain_config() -> None:
    settings = Settings(roi_upscale_factor=3.0, roi_clahe_clip_limit=4.5, _env_file=None)

    config = settings.to_preprocessing_config()

    assert config.roi_upscale_factor == 3.0
    assert config.roi_clahe_clip_limit == 4.5


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def test_the_pdf_processor_factory_resolves_pymupdf() -> None:
    processor = build_pdf_processor(Settings(_env_file=None))

    assert processor.provider_name == "pymupdf"


def test_the_foundry_provider_needs_no_key() -> None:
    settings = Settings(
        vlm_provider=VLMProvider.AZURE_FOUNDRY,
        azure_foundry_endpoint="https://example.openai.azure.com/",
        _env_file=None,
    )

    provider = build_vlm_provider(settings)

    # An empty key selects Managed Identity, which is the preferred enterprise
    # configuration rather than a misconfiguration.
    assert provider.provider_name == "azure_foundry"
    assert provider.uses_managed_identity is True


def test_the_foundry_provider_still_needs_an_endpoint() -> None:
    settings = Settings(vlm_provider=VLMProvider.AZURE_FOUNDRY, _env_file=None)

    with pytest.raises(VLMProviderError):
        build_vlm_provider(settings)


# --------------------------------------------------------------------------- #
# Dependency container
# --------------------------------------------------------------------------- #
def test_injected_ports_take_precedence_over_the_configuration() -> None:
    extractor = FakeExtractorAdapter({})
    vlm = FakeVLMAdapter()
    pdf_processor = FakePDFProcessorAdapter([])

    dependencies = build_dependencies(
        Settings(_env_file=None),
        extractor=extractor,
        vlm_provider=vlm,
        grounding_extractor=extractor,
        pdf_processor=pdf_processor,
    )

    # This is how the tests run the whole engine with no credential at all.
    assert dependencies.extractor is extractor
    assert dependencies.pdf_processor is pdf_processor
    assert dependencies.grounding_extractor is extractor


def test_the_grounding_extractor_defaults_to_a_grounding_capable_one() -> None:
    dependencies = build_dependencies(
        Settings(extractor_provider=ExtractorProvider.NATIVE_VLM, _env_file=None),
        vlm_provider=FakeVLMAdapter(),
    )

    # Grounding is optional for the single document workflow but is the point of
    # the dossier branch, so it is forced on there.
    assert dependencies.grounding_extractor.provider_name == "native_vlm"
    assert dependencies.grounding_extractor._request_bounding_boxes is True


def test_the_grounding_extractor_does_not_flip_the_configured_one() -> None:
    """Regression: two extractors share one provider, and must not fight over it.

    Building the grounding extractor used to stamp
    ``request_bounding_boxes=True`` onto the shared VLM provider, so whichever
    adapter was constructed last silently decided for both. A deployment that
    had turned grounding off then paid for boxes it never asked for, and the ROI
    retry cropped whatever region the model invented instead of re-reading the
    page.
    """
    vlm = FakeVLMAdapter(full_document_response="{}")
    dependencies = build_dependencies(
        Settings(
            extractor_provider=ExtractorProvider.NATIVE_VLM,
            native_vlm_request_bounding_boxes=False,
            _env_file=None,
        ),
        vlm_provider=vlm,
    )

    dependencies.extractor.extract_document(b"page", "INE")
    dependencies.grounding_extractor.extract_document(b"page", "INVOICE")

    # Each call carries its own answer, and neither adapter left state behind.
    assert vlm.grounding_flags == [False, True]
    assert not hasattr(vlm, "request_bounding_boxes")


def test_closing_the_container_closes_each_adapter_once() -> None:
    closed: list[str] = []

    class RecordingVLM(FakeVLMAdapter):
        def close(self) -> None:
            closed.append("vlm")

    class RecordingPDF(FakePDFProcessorAdapter):
        def close(self) -> None:
            closed.append("pdf")

    class RecordingExtractor(FakeExtractorAdapter):
        def close(self) -> None:
            closed.append("extractor")

    extractor = RecordingExtractor({})
    dependencies = Dependencies(
        settings=Settings(_env_file=None),
        extractor=extractor,
        # The same instance in both slots is the common case, and must not be
        # closed twice.
        grounding_extractor=extractor,
        vlm_provider=RecordingVLM(),
        pdf_processor=RecordingPDF([]),
    )

    dependencies.close()

    assert sorted(closed) == ["extractor", "pdf", "vlm"]


def test_a_failing_close_does_not_leak_the_other_adapters() -> None:
    closed: list[str] = []

    class ExplodingExtractor(FakeExtractorAdapter):
        def close(self) -> None:
            raise RuntimeError("The client refused to close.")

    class RecordingVLM(FakeVLMAdapter):
        def close(self) -> None:
            closed.append("vlm")

    dependencies = Dependencies(
        settings=Settings(_env_file=None),
        extractor=ExplodingExtractor({}),
        grounding_extractor=FakeExtractorAdapter({}),
        vlm_provider=RecordingVLM(),
        pdf_processor=FakePDFProcessorAdapter([]),
    )

    dependencies.close()

    assert closed == ["vlm"]
