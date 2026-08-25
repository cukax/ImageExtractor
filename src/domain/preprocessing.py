"""Deterministic image preprocessing built on OpenCV and PIL.

Pure domain logic: it takes bytes in and returns bytes out, never touching the
filesystem or the network. That makes the whole pipeline unit testable without
any provider credential.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import BoundingBox

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Tuning knobs for the preprocessing and ROI cropping routines."""

    blur_critical_threshold: float = 120.0
    """Laplacian variance below which the image is considered soft."""

    blur_reference_width: int = 1000
    """Width the image is normalized to before measuring blur.

    Laplacian variance scales with resolution, so measuring it at a fixed width
    keeps a single threshold meaningful across phone cameras and flatbed scans.
    """

    unsharp_amount: float = 0.8
    unsharp_sigma: float = 2.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    max_dimension: int = 2000
    jpeg_quality: int = 85
    original_jpeg_quality: int = 95

    roi_padding_ratio: float = 0.10
    """Margin added around a bounding box before cropping, as a ratio."""

    roi_min_width: int = 320
    """Crops narrower than this are upscaled so the VLM sees legible glyphs."""

    roi_max_upscale: float = 4.0
    """Upper bound on the ROI upscale factor, to avoid pure interpolation noise."""

    @property
    def blur_rejection_threshold(self) -> float:
        """Variance below which the document is rejected outright (fail fast)."""
        return self.blur_critical_threshold / 2.0


@dataclass(slots=True)
class PreprocessedImage:
    """Result of the preprocessing stage."""

    processed_bytes: bytes
    """Enhanced, resized and compressed image sent to the extraction provider."""

    original_bytes: bytes
    """EXIF corrected full resolution image, used for high precision ROI crops."""

    blur_variance: float
    is_readable: bool
    width: int
    height: int
    applied_steps: list[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Conversion helpers
# --------------------------------------------------------------------------- #
def load_image(image_bytes: bytes) -> Image.Image:
    """Decode bytes into a PIL image with the EXIF orientation applied.

    Phone cameras store the sensor orientation in EXIF instead of rotating the
    pixels. Skipping this step makes every downstream bounding box wrong for
    portrait photos, so it is the very first thing the pipeline does.
    """
    if not image_bytes:
        raise ValueError("Cannot decode an empty image payload.")
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    transposed = ImageOps.exif_transpose(image)
    return (transposed or image).convert("RGB")


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image into the BGR ndarray OpenCV expects."""
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(matrix: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR ndarray back into a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(matrix, cv2.COLOR_BGR2RGB))


def encode_jpeg(matrix: np.ndarray, quality: int) -> bytes:
    """Encode a BGR ndarray as JPEG in memory."""
    success, buffer = cv2.imencode(".jpg", matrix, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not success:
        raise RuntimeError("OpenCV failed to encode the image as JPEG.")
    return buffer.tobytes()


def encode_png(matrix: np.ndarray) -> bytes:
    """Encode a BGR ndarray as PNG in memory.

    ROI crops are small and contain the exact glyphs the VLM must read, so they
    are kept lossless: JPEG ringing around thin strokes is precisely the kind of
    artifact that causes a re-read to fail again.
    """
    success, buffer = cv2.imencode(".png", matrix)
    if not success:
        raise RuntimeError("OpenCV failed to encode the image as PNG.")
    return buffer.tobytes()


# --------------------------------------------------------------------------- #
# Individual operations
# --------------------------------------------------------------------------- #
def laplacian_variance(matrix: np.ndarray, reference_width: int = 1000) -> float:
    """Measure image sharpness as the variance of the Laplacian.

    A blurred image has few sharp intensity transitions, so the second order
    derivative stays close to zero and its variance collapses.
    """
    grayscale = cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
    height, width = grayscale.shape[:2]
    if reference_width > 0 and width != reference_width and width > 0:
        scale = reference_width / float(width)
        grayscale = cv2.resize(
            grayscale,
            (reference_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
    return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())


def unsharp_mask(matrix: np.ndarray, amount: float = 0.8, sigma: float = 2.0) -> np.ndarray:
    """Apply a mild unsharp mask to recover edge definition on soft images."""
    blurred = cv2.GaussianBlur(matrix, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(matrix, 1.0 + amount, blurred, -amount, 0)


def apply_clahe(matrix: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Equalize local contrast on the L channel of the LAB color space.

    Working in LAB keeps the chroma channels untouched, so shadows cast by a
    hand or a phone are flattened without shifting the document colors, which
    would otherwise confuse downstream color based heuristics.
    """
    lab = cv2.cvtColor(matrix, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    equalized = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((equalized, green_red, blue_yellow)), cv2.COLOR_LAB2BGR)


def resize_max_dimension(matrix: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale so the longest side equals max_dimension, preserving the ratio."""
    height, width = matrix.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension or longest == 0:
        return matrix
    scale = max_dimension / float(longest)
    return cv2.resize(
        matrix,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


# --------------------------------------------------------------------------- #
# Pipeline entry points
# --------------------------------------------------------------------------- #
def preprocess_image(
    image_bytes: bytes,
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessedImage:
    """Run the full preprocessing chain and the fail fast blur filter.

    Order matters. Blur is measured on the raw, EXIF corrected pixels: measuring
    it after sharpening or CLAHE would inflate the variance and let unusable
    documents through the cost saving filter.
    """
    config = config or PreprocessingConfig()

    pil_image = load_image(image_bytes)
    matrix = pil_to_bgr(pil_image)
    height, width = matrix.shape[:2]
    applied_steps = ["exif_transpose"]

    variance = laplacian_variance(matrix, config.blur_reference_width)
    applied_steps.append(f"laplacian_variance={variance:.2f}")

    # The full resolution EXIF corrected original is preserved before any
    # enhancement, so the ROI retry re-reads real pixels rather than the
    # artifacts introduced by CLAHE and by JPEG compression.
    original_bytes = encode_jpeg(matrix, config.original_jpeg_quality)

    if variance < config.blur_rejection_threshold:
        LOGGER.warning(
            "Image rejected by the fail fast blur filter: variance=%.2f threshold=%.2f",
            variance,
            config.blur_rejection_threshold,
        )
        return PreprocessedImage(
            processed_bytes=image_bytes,
            original_bytes=original_bytes,
            blur_variance=variance,
            is_readable=False,
            width=width,
            height=height,
            applied_steps=applied_steps,
            rejection_reason=(
                f"Laplacian variance {variance:.2f} is below the rejection threshold "
                f"{config.blur_rejection_threshold:.2f}; no extraction API was called."
            ),
        )

    if variance < config.blur_critical_threshold:
        matrix = unsharp_mask(matrix, config.unsharp_amount, config.unsharp_sigma)
        applied_steps.append("unsharp_mask")

    matrix = apply_clahe(matrix, config.clahe_clip_limit, config.clahe_tile_grid_size)
    applied_steps.append("clahe_lab")

    matrix = resize_max_dimension(matrix, config.max_dimension)
    applied_steps.append(f"resize_max_dimension={config.max_dimension}")

    processed_bytes = encode_jpeg(matrix, config.jpeg_quality)
    applied_steps.append(f"jpeg_quality={config.jpeg_quality}")
    processed_height, processed_width = matrix.shape[:2]

    return PreprocessedImage(
        processed_bytes=processed_bytes,
        original_bytes=original_bytes,
        blur_variance=variance,
        is_readable=True,
        width=processed_width,
        height=processed_height,
        applied_steps=applied_steps,
    )


def crop_region(
    image_bytes: bytes,
    box: BoundingBox,
    config: Optional[PreprocessingConfig] = None,
) -> bytes:
    """Crop the padded region of interest and return it as PNG bytes.

    Small crops are upscaled with a cubic filter: a 40 pixel wide tax id is
    unreadable for a vision model, and the retry only pays off when the crop is
    genuinely magnified.
    """
    config = config or PreprocessingConfig()
    if box.is_degenerate:
        raise ValueError("Cannot crop a degenerate bounding box.")

    matrix = pil_to_bgr(load_image(image_bytes))
    height, width = matrix.shape[:2]

    padded = box.with_padding(config.roi_padding_ratio)
    left, top, right, bottom = padded.to_pixels(width, height)
    crop = matrix[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("The bounding box produced an empty crop.")

    crop_height, crop_width = crop.shape[:2]
    if crop_width < config.roi_min_width:
        scale = min(config.roi_min_width / float(crop_width), config.roi_max_upscale)
        if scale > 1.0:
            crop = cv2.resize(
                crop,
                (int(round(crop_width * scale)), max(1, int(round(crop_height * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )

    return encode_png(crop)
