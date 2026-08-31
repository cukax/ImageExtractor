"""LangGraph nodes.

Every node is produced by a factory that closes over the ports it needs. That is
the dependency injection seam of the system: the node functions themselves only
ever see a DocumentExtractorPort or a VLMProviderPort, never a concrete SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import ValidationError

from ..domain.models import BoundingBox, get_schema_for
from ..domain.policy import WorkflowPolicy
from ..domain.preprocessing import crop_region, preprocess_image
from ..domain.state import DocumentStatus, OCRState
from ..domain.validation_tools import expected_format_for, validate_document
from ..ports.extractor_port import DocumentExtractorPort, ExtractionError
from ..ports.vlm_port import UNREADABLE_TOKEN, VLMProviderError, VLMProviderPort

LOGGER = logging.getLogger(__name__)

# Node identifiers, shared by the workflow builder and by the routers.
PREPROCESS_NODE = "preprocess"
EXTRACTION_NODE = "extraction"
VALIDATION_NODE = "validation"
CROP_AND_VLM_RETRY_NODE = "crop_and_vlm_retry"
HITL_FALLBACK_NODE = "hitl_fallback"

NodeCallable = Callable[[OCRState], dict[str, Any]]


# --------------------------------------------------------------------------- #
# Node 0: preprocessing and fail fast filter
# --------------------------------------------------------------------------- #
def build_preprocess_node(policy: WorkflowPolicy) -> NodeCallable:
    """Build the preprocessing node.

    Responsibilities: EXIF normalization, blur measurement with an early abort,
    CLAHE contrast equalization, downscaling and JPEG compression.
    """
    preprocessing_config = policy.preprocessing

    def preprocess_node(state: OCRState) -> dict[str, Any]:
        image_bytes = state.get("image_bytes")
        if not image_bytes:
            return {
                "is_readable": False,
                "status": DocumentStatus.EXTRACTION_FAILED.value,
                "errors": [*state.get("errors", []), "The input image payload is empty."],
            }

        try:
            result = preprocess_image(image_bytes, preprocessing_config)
        except Exception as error:  # noqa: BLE001 - a corrupt upload must not crash the graph
            LOGGER.exception("Preprocessing failed")
            return {
                "is_readable": False,
                "status": DocumentStatus.EXTRACTION_FAILED.value,
                "errors": [*state.get("errors", []), f"Preprocessing failed: {error}"],
            }

        report: dict[str, Any] = {
            "applied_steps": result.applied_steps,
            "width": result.width,
            "height": result.height,
            "blur_variance": result.blur_variance,
            "blur_critical_threshold": preprocessing_config.blur_critical_threshold,
            "blur_rejection_threshold": preprocessing_config.blur_rejection_threshold,
            "processed_size_bytes": len(result.processed_bytes),
        }

        if not result.is_readable:
            # Fail fast: aborting here is the whole point of the filter, since a
            # rejected page never reaches a paid extraction API.
            report["rejection_reason"] = result.rejection_reason
            return {
                "image_bytes": result.processed_bytes,
                "original_image_bytes": result.original_bytes,
                "blur_variance": result.blur_variance,
                "is_readable": False,
                "status": DocumentStatus.REJECTED_BLUR.value,
                "preprocessing_report": report,
            }

        return {
            "image_bytes": result.processed_bytes,
            "original_image_bytes": result.original_bytes,
            "blur_variance": result.blur_variance,
            "is_readable": True,
            "status": DocumentStatus.PROCESSING.value,
            "preprocessing_report": report,
        }

    return preprocess_node


# --------------------------------------------------------------------------- #
# Node 1: structured extraction
# --------------------------------------------------------------------------- #
def build_extraction_node(extractor: DocumentExtractorPort) -> NodeCallable:
    """Build the extraction node around an injected DocumentExtractorPort."""

    def extraction_node(state: OCRState) -> dict[str, Any]:
        doc_type = state.get("doc_type", "")
        image_bytes = state.get("image_bytes", b"")

        try:
            result = extractor.extract_document(image_bytes, doc_type)
        except ExtractionError as error:
            LOGGER.error("Extraction failed: %s", error)
            return {
                "status": DocumentStatus.EXTRACTION_FAILED.value,
                "errors": [*state.get("errors", []), str(error)],
            }

        schema_cls = get_schema_for(doc_type)
        try:
            document, bounding_boxes, unmapped = schema_cls.from_extraction(result)
        except ValidationError as error:
            LOGGER.error("The provider payload does not fit %s: %s", schema_cls.__name__, error)
            return {
                "status": DocumentStatus.EXTRACTION_FAILED.value,
                "errors": [
                    *state.get("errors", []),
                    f"Could not map the {result.provider} payload onto {schema_cls.__name__}: {error}",
                ],
            }

        if unmapped:
            LOGGER.info("Provider fields ignored by %s: %s", schema_cls.__name__, unmapped)

        return {
            "extracted_data": document.to_flat_dict(),
            "bounding_boxes": bounding_boxes,
            "status": DocumentStatus.PROCESSING.value,
            "provider_metadata": {
                "provider": result.provider,
                "schema": schema_cls.__name__,
                "page_count": result.page_count,
                "warnings": result.warnings,
                "unmapped_fields": unmapped,
                "located_fields": sorted(bounding_boxes),
            },
        }

    return extraction_node


# --------------------------------------------------------------------------- #
# Node 2: tool validation
# --------------------------------------------------------------------------- #
def build_validation_node(policy: WorkflowPolicy) -> NodeCallable:
    """Build the validation node.

    The node owns the retry counter: incrementing it here, rather than in the
    retry node, keeps the count correct even when the retry node is skipped
    because no failing field can be located.
    """

    def validation_node(state: OCRState) -> dict[str, Any]:
        doc_type = state.get("doc_type", "")
        extracted_data = state.get("extracted_data", {})
        errors = validate_document(doc_type, extracted_data)

        if not errors:
            LOGGER.info("Document validated after %d retries", state.get("retry_count", 0))
            return {
                "validation_errors": {},
                "failed_fields": [],
                "status": DocumentStatus.VALIDATED.value,
            }

        failed_fields = sorted(errors)
        retry_count = int(state.get("retry_count", 0)) + 1
        LOGGER.warning(
            "Validation failed on %s (attempt %d/%d)",
            failed_fields,
            retry_count,
            policy.max_retries,
        )
        return {
            "validation_errors": errors,
            "failed_fields": failed_fields,
            "retry_count": retry_count,
            "status": DocumentStatus.PROCESSING.value,
        }

    return validation_node


# --------------------------------------------------------------------------- #
# Node 3: targeted ROI crop and VLM retry
# --------------------------------------------------------------------------- #
def build_crop_and_vlm_retry_node(
    vlm_provider: VLMProviderPort,
    policy: WorkflowPolicy,
) -> NodeCallable:
    """Build the targeted re-read node around an injected VLMProviderPort."""
    preprocessing_config = policy.preprocessing

    def crop_and_vlm_retry_node(state: OCRState) -> dict[str, Any]:
        doc_type = state.get("doc_type", "")
        extracted_data = dict(state.get("extracted_data", {}))
        bounding_boxes = state.get("bounding_boxes", {})
        validation_errors = state.get("validation_errors", {})
        flagged_fields = set(state.get("flagged_fields", []))
        run_errors = list(state.get("errors", []))

        # The full resolution EXIF corrected original is preferred: the crop is
        # magnified, so every pixel lost to downscaling costs accuracy.
        source_bytes = state.get("original_image_bytes") or state.get("image_bytes", b"")

        repaired: dict[str, str] = {}
        for field_name in state.get("failed_fields", []):
            if field_name in flagged_fields:
                # Already declared unreadable by a previous pass; retrying it
                # would only burn tokens.
                continue

            error_context = _build_error_context(doc_type, field_name, validation_errors)
            box_tuple = bounding_boxes.get(field_name)

            try:
                if box_tuple:
                    crop_bytes = crop_region(
                        source_bytes,
                        BoundingBox.from_tuple(box_tuple),
                        preprocessing_config,
                    )
                else:
                    # No geometry: some providers, and most VLMs, do not ground
                    # their answers. Re-reading the whole page is still a real
                    # second opinion, just a more expensive one.
                    LOGGER.info(
                        "No bounding box for %r; falling back to a full page re-read",
                        field_name,
                    )
                    crop_bytes = state.get("image_bytes", source_bytes)

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
            extracted_data = _renormalize(doc_type, extracted_data)

        LOGGER.info("ROI retry repaired %d field(s): %s", len(repaired), sorted(repaired))
        return {
            "extracted_data": extracted_data,
            "flagged_fields": sorted(flagged_fields),
            "errors": run_errors,
            "status": DocumentStatus.PROCESSING.value,
        }

    return crop_and_vlm_retry_node


def _build_error_context(
    doc_type: str,
    field_name: str,
    validation_errors: dict[str, str],
) -> str:
    """Compose the error context handed to the VLM.

    The documented expected format is appended to the raw validation message:
    telling the model what a correct value looks like measurably reduces the
    number of second failures.
    """
    message = validation_errors.get(field_name, "The extracted value failed validation.")
    expected_format = expected_format_for(doc_type, field_name)
    if expected_format and expected_format not in message:
        return f"{message} The expected format is {expected_format}."
    return message


def _renormalize(doc_type: str, extracted_data: dict[str, Any]) -> dict[str, Any]:
    """Re-run the schema validators over the patched data.

    A value returned by the VLM is raw text, so pushing it back through the
    Pydantic schema restores the invariants the extraction node established
    (uppercase codes, parsed amounts). A failure here is not fatal: the raw
    values simply reach the validation node unchanged.
    """
    schema_cls = get_schema_for(doc_type)
    try:
        return schema_cls.from_flat_dict(extracted_data).to_flat_dict()
    except ValidationError as error:
        LOGGER.warning("Could not renormalize the repaired data: %s", error)
        return extracted_data


# --------------------------------------------------------------------------- #
# Node 4: human in the loop escalation
# --------------------------------------------------------------------------- #
def build_hitl_fallback_node(policy: WorkflowPolicy) -> NodeCallable:
    """Build the escalation node.

    The node calls LangGraph interrupt(), which persists the state in the
    checkpointer and suspends the run until an operator resumes it with a
    Command(resume=...). A checkpointer is therefore mandatory.
    """

    def hitl_fallback_node(state: OCRState) -> dict[str, Any]:
        extracted_data = dict(state.get("extracted_data", {}))
        validation_errors = state.get("validation_errors", {})
        failed_fields = list(state.get("failed_fields", []))
        flagged_fields = sorted(set(state.get("flagged_fields", [])) | set(failed_fields))

        # Everything above this line must stay side effect free: on resume, the
        # node is replayed from its start and interrupt() returns the operator
        # payload instead of suspending again.
        review_payload = {
            "reason": "The retry budget was exhausted while validation errors remained.",
            "doc_type": state.get("doc_type"),
            "retry_count": state.get("retry_count", 0),
            "max_retries": policy.max_retries,
            "extracted_data": extracted_data,
            "validation_errors": validation_errors,
            "flagged_fields": flagged_fields,
            "bounding_boxes": {
                field: state.get("bounding_boxes", {}).get(field)
                for field in flagged_fields
            },
            "provider_metadata": state.get("provider_metadata", {}),
            "blur_variance": state.get("blur_variance"),
            "instructions": (
                "Resume with {'approved': bool, 'corrections': {field: value}} once "
                "the flagged fields have been reviewed."
            ),
        }

        LOGGER.warning("Escalating to a human operator: %s", flagged_fields)
        operator_response = interrupt(review_payload)

        return _apply_human_review(state, operator_response, flagged_fields, review_payload)

    return hitl_fallback_node


def _apply_human_review(
    state: OCRState,
    operator_response: Any,
    flagged_fields: list[str],
    review_payload: dict[str, Any],
) -> dict[str, Any]:
    """Merge the operator decision back into the state after the interrupt."""
    response = operator_response if isinstance(operator_response, dict) else {}
    corrections = response.get("corrections") or {}
    approved = bool(response.get("approved", False))

    doc_type = state.get("doc_type", "")
    extracted_data = {**state.get("extracted_data", {}), **corrections}
    if corrections:
        extracted_data = _renormalize(doc_type, extracted_data)

    # A correction only counts once it passes the very same tools that rejected
    # the automated extraction, so an operator cannot wave through a bad value.
    remaining_errors = validate_document(doc_type, extracted_data)
    resolved = approved and not remaining_errors

    return {
        "extracted_data": extracted_data,
        "validation_errors": remaining_errors,
        "failed_fields": sorted(remaining_errors),
        "flagged_fields": [] if resolved else flagged_fields,
        "human_review_payload": review_payload,
        "human_review_result": {
            "approved": approved,
            "corrections": corrections,
            "unresolved_errors": remaining_errors,
        },
        "status": DocumentStatus.VALIDATED.value if resolved else DocumentStatus.HUMAN_REVIEW_REQUIRED.value,
    }


# --------------------------------------------------------------------------- #
# Conditional routers
# --------------------------------------------------------------------------- #
def route_after_preprocess(state: OCRState) -> str:
    """Abort the run when the fail fast filter rejected the image."""
    if not state.get("is_readable", True):
        return END
    return EXTRACTION_NODE


def route_after_extraction(state: OCRState) -> str:
    """Skip validation when the provider call failed outright."""
    if state.get("status") == DocumentStatus.EXTRACTION_FAILED:
        return END
    return VALIDATION_NODE


def build_route_after_validation(policy: WorkflowPolicy) -> Callable[[OCRState], str]:
    """Build the router that arbitrates between success, retry and escalation."""

    def route_after_validation(state: OCRState) -> str:
        if state.get("status") == DocumentStatus.VALIDATED:
            return END

        flagged_fields = set(state.get("flagged_fields", []))
        retryable = [
            field for field in state.get("failed_fields", []) if field not in flagged_fields
        ]
        if not retryable:
            # Every remaining error is on a field the VLM already declared
            # unreadable, so another pass cannot change the outcome.
            return HITL_FALLBACK_NODE

        if int(state.get("retry_count", 0)) <= policy.max_retries:
            return CROP_AND_VLM_RETRY_NODE
        return HITL_FALLBACK_NODE

    return route_after_validation
