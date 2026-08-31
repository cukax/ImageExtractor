"""Shared fixtures and in memory test doubles.

The fakes below implement the ports exactly as the real adapters do, which is
the practical payoff of hexagonal architecture: the whole graph is exercised end
to end without a single credential, network call or provider SDK.
"""

from __future__ import annotations

import io
from typing import Any, Optional, Sequence

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.domain.models import BoundingBox, ExtractedField, ExtractionResult, RenderedPage
from src.infrastructure.config import ExtractorProvider, Settings, VLMProvider
from src.ports.extractor_port import DocumentExtractorPort, ExtractionError
from src.ports.pdf_port import PDFProcessingError, PDFProcessorPort
from src.ports.vlm_port import UNKNOWN_PAGE_TYPE, UNREADABLE_TOKEN, VLMProviderPort


# --------------------------------------------------------------------------- #
# Synthetic images
# --------------------------------------------------------------------------- #
def make_document_image(blur_radius: float = 0.0, size: tuple[int, int] = (1200, 800)) -> bytes:
    """Render a synthetic document with plenty of high frequency detail.

    A deterministic seed keeps the Laplacian variance stable across runs, so the
    blur thresholds can be asserted exactly.
    """
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    generator = np.random.default_rng(seed=1234)

    for row in range(14):
        top = 40 + row * 50
        draw.rectangle([40, top, size[0] - 40, top + 4], fill="black")
        for column in range(24):
            left = 60 + column * 45
            width = int(generator.integers(8, 30))
            draw.rectangle([left, top + 14, left + width, top + 34], fill="black")

    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def sharp_image_bytes() -> bytes:
    """A crisp document image that passes the fail fast filter."""
    return make_document_image(blur_radius=0.0)


@pytest.fixture
def blurred_image_bytes() -> bytes:
    """A hopelessly blurred image that the fail fast filter must reject."""
    return make_document_image(blur_radius=12.0)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeExtractorAdapter(DocumentExtractorPort):
    """A DocumentExtractorPort returning a scripted payload."""

    provider_name = "fake_extractor"

    def __init__(
        self,
        fields: dict[str, tuple[Optional[str], Optional[tuple[float, float, float, float]]]],
        *,
        structured_extras: Optional[dict[str, Any]] = None,
        raise_error: bool = False,
    ) -> None:
        self._fields = fields
        self._structured_extras = structured_extras or {}
        self._raise_error = raise_error
        self.calls: list[tuple[int, str]] = []

    def extract_document(self, image_bytes: bytes, doc_type: str) -> ExtractionResult:
        self.calls.append((len(image_bytes), doc_type))
        if self._raise_error:
            raise ExtractionError(self.provider_name, "Simulated provider outage.")
        return ExtractionResult(
            doc_type=doc_type,
            provider=self.provider_name,
            fields={
                name: ExtractedField(
                    name=name,
                    value=value,
                    confidence=0.9,
                    bounding_box=BoundingBox.from_tuple(box) if box else None,
                )
                for name, (value, box) in self._fields.items()
            },
            structured_extras=self._structured_extras,
        )


class FakeVLMAdapter(VLMProviderPort):
    """A VLMProviderPort returning scripted answers per field."""

    provider_name = "fake_vlm"

    def __init__(
        self,
        answers_by_field: Optional[dict[str, Any]] = None,
        *,
        full_document_response: str = "{}",
        page_types: Optional[list[str]] = None,
        raise_on_classify: bool = False,
    ) -> None:
        # A value may be a plain string or a list consumed one call at a time,
        # which is how a multi attempt repair sequence is scripted.
        self._answers_by_field = answers_by_field or {}
        self._full_document_response = full_document_response
        # Page labels are consumed in order, one per classify_page call, which
        # is how a dossier layout is scripted.
        self._page_types = list(page_types or [])
        self._raise_on_classify = raise_on_classify
        self.roi_calls: list[tuple[str, str, int]] = []
        self.full_document_calls: list[str] = []
        self.page_calls: list[tuple[str, int]] = []
        self.classify_calls: list[int] = []
        # Records what each caller asked for, so a test can prove the grounding
        # flag travelled with the call rather than living on the provider.
        self.grounding_flags: list[bool] = []

    def analyze_roi_crop(self, crop_bytes: bytes, field_name: str, error_context: str) -> str:
        self.roi_calls.append((field_name, error_context, len(crop_bytes)))
        answer = self._answers_by_field.get(field_name, UNREADABLE_TOKEN)
        if isinstance(answer, list):
            return answer.pop(0) if answer else UNREADABLE_TOKEN
        return str(answer)

    def analyze_full_document(
        self,
        image_bytes: bytes,
        doc_type: str,
        json_schema: dict[str, Any],
        layout_hints: Optional[str] = None,
        request_bounding_boxes: bool = False,
    ) -> str:
        self.full_document_calls.append(doc_type)
        self.grounding_flags.append(request_bounding_boxes)
        return self._full_document_response

    def analyze_document_pages(
        self,
        images_bytes: Sequence[bytes],
        doc_type: str,
        json_schema: dict[str, Any],
        layout_hints: Optional[str] = None,
        request_bounding_boxes: bool = False,
    ) -> str:
        self.page_calls.append((doc_type, len(images_bytes)))
        return self.analyze_full_document(
            images_bytes[0] if images_bytes else b"",
            doc_type,
            json_schema,
            layout_hints,
            request_bounding_boxes,
        )

    def classify_page(self, image_bytes: bytes, allowed_types: Sequence[str]) -> str:
        self.classify_calls.append(len(image_bytes))
        if self._raise_on_classify:
            from src.ports.vlm_port import VLMProviderError

            raise VLMProviderError(self.provider_name, "Simulated classifier outage.")
        if self._page_types:
            return self._page_types.pop(0)
        return UNKNOWN_PAGE_TYPE


class FakePDFProcessorAdapter(PDFProcessorPort):
    """A PDFProcessorPort returning a scripted list of page rasters."""

    provider_name = "fake_pdf"

    def __init__(
        self,
        page_images: Optional[list[bytes]] = None,
        *,
        raise_error: bool = False,
    ) -> None:
        self._page_images = list(page_images or [])
        self._raise_error = raise_error
        self.calls: list[tuple[int, int]] = []

    def render_pages(self, document_bytes: bytes, *, dpi: int = 300) -> list[RenderedPage]:
        self.calls.append((len(document_bytes), dpi))
        if self._raise_error:
            raise PDFProcessingError(self.provider_name, "Simulated rasterization failure.")
        return [
            RenderedPage(
                page_index=index,
                image_bytes=image_bytes,
                width=1200,
                height=800,
                dpi=dpi,
            )
            for index, image_bytes in enumerate(self._page_images)
        ]


def make_pdf_bytes(page_count: int = 3) -> bytes:
    """Render a minimal multi page PDF, for the real PyMuPDF adapter tests.

    Built with PIL rather than a fixture file so the suite stays self contained
    and the page count is a parameter rather than a constant.
    """
    pages = [
        Image.open(io.BytesIO(make_document_image(size=(600, 800)))).convert("RGB")
        for _ in range(page_count)
    ]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


@pytest.fixture
def settings() -> Settings:
    """Settings pinned to the fakes, independent of the developer environment."""
    return Settings(
        extractor_provider=ExtractorProvider.NATIVE_VLM,
        vlm_provider=VLMProvider.OPENAI,
        max_retries=3,
        _env_file=None,
    )


@pytest.fixture
def dossier_pages() -> list[bytes]:
    """Three distinct page rasters, enough to exercise the interleaved pairing."""
    return [
        make_document_image(size=(1200, 800)),
        make_document_image(size=(1100, 850)),
        make_document_image(size=(1000, 900)),
    ]
