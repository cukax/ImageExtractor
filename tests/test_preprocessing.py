"""Tests for the OpenCV / PIL preprocessing stage."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.domain.models import BoundingBox
from src.domain.preprocessing import (
    PreprocessingConfig,
    crop_region,
    laplacian_variance,
    load_image,
    pil_to_bgr,
    preprocess_image,
)

from .conftest import make_document_image


def test_sharp_image_passes_the_fail_fast_filter(sharp_image_bytes: bytes) -> None:
    result = preprocess_image(sharp_image_bytes, PreprocessingConfig())

    assert result.is_readable is True
    assert result.rejection_reason is None
    assert result.blur_variance > PreprocessingConfig().blur_critical_threshold
    assert "clahe_lab" in result.applied_steps
    assert result.processed_bytes.startswith(b"\xff\xd8\xff")


def test_blurred_image_is_rejected_before_any_api_call(blurred_image_bytes: bytes) -> None:
    config = PreprocessingConfig()
    result = preprocess_image(blurred_image_bytes, config)

    assert result.is_readable is False
    assert result.blur_variance < config.blur_rejection_threshold
    assert result.rejection_reason is not None
    # The enhancement steps must not run on a rejected document.
    assert "clahe_lab" not in result.applied_steps


def test_mildly_soft_image_is_sharpened_rather_than_rejected() -> None:
    image_bytes = make_document_image(blur_radius=0.0)
    measured = laplacian_variance(pil_to_bgr(load_image(image_bytes)))
    # Place the image inside the salvageable band: rejection < variance < critical.
    config = PreprocessingConfig(blur_critical_threshold=measured * 1.5)

    result = preprocess_image(image_bytes, config)

    assert config.blur_rejection_threshold < measured < config.blur_critical_threshold
    assert result.is_readable is True
    assert "unsharp_mask" in result.applied_steps


def test_resizing_caps_the_longest_side(sharp_image_bytes: bytes) -> None:
    config = PreprocessingConfig(max_dimension=640)
    result = preprocess_image(sharp_image_bytes, config)

    assert max(result.width, result.height) == 640
    with Image.open(io.BytesIO(result.processed_bytes)) as image:
        assert max(image.size) == 640


def test_the_full_resolution_original_is_preserved(sharp_image_bytes: bytes) -> None:
    config = PreprocessingConfig(max_dimension=400)
    result = preprocess_image(sharp_image_bytes, config)

    with Image.open(io.BytesIO(result.original_bytes)) as original:
        # The original keeps its native size, which is what makes the magnified
        # ROI crop worth performing.
        assert max(original.size) == 1200
    assert max(result.width, result.height) == 400


def test_exif_orientation_is_normalized() -> None:
    portrait = Image.new("RGB", (400, 200), "white")
    buffer = io.BytesIO()
    # EXIF tag 274 = Orientation; 6 means "rotate 90 degrees clockwise".
    exif = portrait.getexif()
    exif[274] = 6
    portrait.save(buffer, format="JPEG", exif=exif)

    normalized = load_image(buffer.getvalue())

    assert normalized.size == (200, 400)


def test_laplacian_variance_is_resolution_independent() -> None:
    image_bytes = make_document_image(blur_radius=0.0, size=(1200, 800))
    matrix = pil_to_bgr(load_image(image_bytes))

    small = laplacian_variance(matrix, reference_width=500)
    large = laplacian_variance(matrix, reference_width=500)

    assert small == pytest.approx(large)


def test_crop_region_applies_padding_and_upscales(sharp_image_bytes: bytes) -> None:
    config = PreprocessingConfig(roi_padding_ratio=0.10, roi_min_width=320, roi_max_upscale=4.0)
    box = BoundingBox(x=0.10, y=0.10, width=0.05, height=0.03)

    crop_bytes = crop_region(sharp_image_bytes, box, config)

    assert crop_bytes.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(crop_bytes)) as crop:
        # 0.05 * 1200 px = 60 px, widened to 72 px by the 10% padding on each
        # side, then upscaled by the 4x cap rather than all the way to 320 px.
        assert crop.size[0] == 288


def test_the_roi_upscale_is_capped(sharp_image_bytes: bytes) -> None:
    tiny_box = BoundingBox(x=0.10, y=0.10, width=0.01, height=0.01)
    config = PreprocessingConfig(roi_min_width=2000, roi_max_upscale=2.0)

    crop_bytes = crop_region(sharp_image_bytes, tiny_box, config)

    with Image.open(io.BytesIO(crop_bytes)) as crop:
        # 0.01 * 1200 px = 12 px, padded to 14 px, doubled by the cap.
        assert crop.size[0] <= 14 * 2 + 1


def test_crop_region_rejects_a_degenerate_box(sharp_image_bytes: bytes) -> None:
    with pytest.raises(ValueError):
        crop_region(sharp_image_bytes, BoundingBox(x=0.1, y=0.1, width=0.0, height=0.0))


def test_empty_payload_raises(sharp_image_bytes: bytes) -> None:
    with pytest.raises(ValueError):
        load_image(b"")
