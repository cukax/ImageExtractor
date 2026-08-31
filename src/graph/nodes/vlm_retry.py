"""Phase 6: targeted ROI crop and VLM retry inside the worker subgraph.

For every field that failed validation and has not already been declared
unreadable, the node expands the bounding box by a safety margin, crops that
region from the full resolution page it belongs to, magnifies and cleans it, and
asks the vision model to re-read that one value.

Cropping is what makes the second read a genuinely different attempt: the model
sees a magnified, denoised, contrast equalized patch rather than the same page it
already misread once.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ...domain.models import BoundingBox
from ...domain.policy import WorkflowPolicy
from ...domain.preprocessing import crop_and_enhance_region
from ...domain.worker_state import DocumentWorkerState, WorkerStatus
from ...ports.vlm_port import UNREADABLE_TOKEN, VLMProviderError, VLMProviderPort
from ._shared import build_error_context, renormalize

LOGGER = logging.getLogger(__name__)

VLM_RETRY_NODE = "vlm_retry"


def build_vlm_retry_node(
    vlm_provider: VLMProviderPort,
    policy: WorkflowPolicy,
) -> Callable[[DocumentWorkerState], dict[str, Any]]:
    """Build the targeted re-read node around an injected VLMProviderPort."""
    preprocessing_config = policy.preprocessing

    def vlm_retry_node(state: DocumentWorkerState) -> dict[str, Any]:
        doc_type = state.get("doc_type") or ""
        extracted_data = dict(state.get("extracted_data", {}))
        bounding_boxes = state.get("bounding_boxes", {})
        bounding_box_pages = state.get("bounding_box_pages", {})
        validation_errors = state.get("validation_errors", {})
        flagged_fields = set(state.get("flagged_fields", []))
        run_errors = list(state.get("errors", []))

        # The full resolution pages are preferred: the crop is magnified, so
        # every pixel lost to downscaling costs accuracy.
        source_pages = state.get("original_images_bytes") or state.get("images_bytes") or []
        if not source_pages:
            LOGGER.error("Document %s has no page to crop from", state.get("doc_id"))
            return {
                "errors": [*run_errors, "The ROI retry found no source page to crop."],
                "status": WorkerStatus.PROCESSING.value,
            }

        repaired: dict[str, str] = {}
        for field_name in state.get("failed_fields", []):
            if field_name in flagged_fields:
                # Already declared unreadable by a previous pass; retrying it
                # would only burn tokens.
                continue

            error_context = build_error_context(doc_type, field_name, validation_errors)
            box_tuple = bounding_boxes.get(field_name)
            page_index = int(bounding_box_pages.get(field_name, 0))
            page_bytes = _page_at(source_pages, page_index)

            try:
                crop_bytes = _build_retry_payload(
                    page_bytes,
                    box_tuple,
                    preprocessing_config,
                    field_name,
                )
                new_value = vlm_provider.analyze_roi_crop(
                    crop_bytes=crop_bytes,
                    field_name=field_name,
                    error_context=error_context,
                )
            except (VLMProviderError, ValueError) as error:
                LOGGER.error("ROI retry failed for %r: %s", field_name, error)
                run_errors.append(f"ROI retry failed for {field_name}: {error}")
                continue

            if new_value == UNREADABLE_TOKEN:
                LOGGER.warning("The VLM declared %r unreadable; flagging it", field_name)
                flagged_fields.add(field_name)
                continue

            repaired[field_name] = new_value

        if repaired:
            extracted_data.update(repaired)
            extracted_data = renormalize(doc_type, extracted_data)

        LOGGER.info(
            "ROI retry on %s repaired %d field(s): %s",
            state.get("doc_id"),
            len(repaired),
            sorted(repaired),
        )
        return {
            "extracted_data": extracted_data,
            "flagged_fields": sorted(flagged_fields),
            "errors": run_errors,
            "status": WorkerStatus.PROCESSING.value,
        }

    return vlm_retry_node


def _page_at(pages: list[bytes], page_index: int) -> bytes:
    """Return the requested page, falling back to the first one.

    A grounded box always carries a valid page index, but a model that grounded
    nothing leaves the default 0, and an out of range index is a provider bug
    that must not crash a repair pass.
    """
    if 0 <= page_index < len(pages):
        return pages[page_index]
    return pages[0]


def _build_retry_payload(
    page_bytes: bytes,
    box_tuple: Optional[tuple[float, float, float, float]],
    preprocessing_config: Any,
    field_name: str,
) -> bytes:
    """Produce the image the model re-reads: an enhanced crop, or the whole page.

    Falling back to the full page is not a formality. Many providers, and most
    vision models, do not ground their answers; re-reading the page is still a
    real second opinion, just a more expensive one.
    """
    if box_tuple:
        return crop_and_enhance_region(
            page_bytes,
            BoundingBox.from_tuple(box_tuple),
            preprocessing_config,
        )
    LOGGER.info("No bounding box for %r; falling back to a full page re-read", field_name)
    return page_bytes
