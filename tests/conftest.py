"""Shared fixtures and in memory test doubles.

The fakes below implement the ports exactly as the real adapters do, which is
the practical payoff of hexagonal architecture: the whole graph is exercised end
to end without a single credential, network call or provider SDK.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from src.domain.models import BoundingBox, ExtractedField, ExtractionResult
from src.infrastructure.config import ExtractorProvider, Settings, VLMProvider
from src.ports.extractor_port import DocumentExtractorPort, ExtractionError
from src.ports.vlm_port import UNREADABLE_TOKEN, VLMProviderPort


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
    ) -> None:
        # A value may be a plain string or a list consumed one call at a time,
        # which is how a multi attempt repair sequence is scripted.
        self._answers_by_field = answers_by_field or {}
        self._full_document_response = full_document_response
        self.roi_calls: list[tuple[str, str, int]] = []
        self.full_document_calls: list[str] = []

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
    ) -> str:
        self.full_document_calls.append(doc_type)
        return self._full_document_response


@pytest.fixture
def settings() -> Settings:
    """Settings pinned to the fakes, independent of the developer environment."""
    return Settings(
        extractor_provider=ExtractorProvider.NATIVE_VLM,
        vlm_provider=VLMProvider.OPENAI,
        max_retries=3,
        _env_file=None,
    )
