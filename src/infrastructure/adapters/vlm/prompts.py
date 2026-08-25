"""Prompt templates shared by every VLM adapter.

Centralizing the prompts guarantees that OpenAI, Anthropic and Ollama are asked
exactly the same question, which is the only way an A/B comparison between
providers measures the model rather than the prompt.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# A. Primary extraction prompt (lightweight / native VLM adapter)
# --------------------------------------------------------------------------- #
PRIMARY_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert OCR vision assistant. Analyze the provided image and extract all "
    "requested document fields. Return ONLY a valid JSON object matching the requested "
    "schema. If a field is illegible, missing, or unreadable, set its value to null. "
    "Do NOT attempt to guess or hallucinate unreadable characters."
)

_PRIMARY_EXTRACTION_USER_TEMPLATE = """Document type: {doc_type}

Extract every field defined by the following JSON Schema. Use exactly these property
names as the keys of your JSON object.

JSON Schema:
{json_schema}
{layout_section}{grounding_section}
Return ONLY the JSON object, with no markdown fences and no commentary."""

_LAYOUT_SECTION_TEMPLATE = """
Layout and orientation hints:
{layout_hints}
"""

_GROUNDING_SECTION = """
In addition to the schema properties, add one extra top level property named
"_bounding_boxes". Map every field you extracted to its location in the image as
[x, y, width, height], using coordinates normalized between 0.0 and 1.0 relative to the
full image. Omit any field whose location you cannot determine confidently.
"""


def build_primary_extraction_user_prompt(
    doc_type: str,
    json_schema: dict[str, Any],
    layout_hints: Optional[str] = None,
    request_bounding_boxes: bool = False,
) -> str:
    """Build the user prompt for whole document extraction.

    Args:
        doc_type: Business document type injected as context.
        json_schema: JSON Schema the model must populate.
        layout_hints: Optional orientation or layout guidance.
        request_bounding_boxes: When True, also ask the model to ground each
            field. Grounding quality varies a lot between models, so it stays
            opt-in through configuration.
    """
    layout_section = (
        _LAYOUT_SECTION_TEMPLATE.format(layout_hints=layout_hints.strip())
        if layout_hints and layout_hints.strip()
        else ""
    )
    return _PRIMARY_EXTRACTION_USER_TEMPLATE.format(
        doc_type=doc_type,
        json_schema=json.dumps(json_schema, indent=2, ensure_ascii=False),
        layout_section=layout_section,
        grounding_section=_GROUNDING_SECTION if request_bounding_boxes else "",
    )


# --------------------------------------------------------------------------- #
# B. Targeted ROI crop retry prompt (high precision retry adapter)
# --------------------------------------------------------------------------- #
ROI_RETRY_SYSTEM_PROMPT = (
    "You are an expert OCR validation repair assistant specializing in high-resolution "
    "image crop analysis. Your sole task is to re-read a single field from a cropped "
    "region of a document."
)

ROI_RETRY_USER_PROMPT_TEMPLATE = """The field '{field_name}' previously extracted from this document failed validation with the following error: '{error_context}'.

Carefully examine this magnified cropped region of the document where '{field_name}' is located.
Re-extract the value with 100% precision.

CRITICAL RULES:
1. Return ONLY the raw extracted value as a plain string.
2. Do NOT write conversational text, markdown formatting, explanations, or code blocks.
3. If the cropped text is genuinely unreadable, reply with 'UNREADABLE'."""


def build_roi_retry_user_prompt(field_name: str, error_context: str) -> str:
    """Render the targeted re-read prompt for one failing field."""
    return ROI_RETRY_USER_PROMPT_TEMPLATE.format(
        field_name=field_name,
        error_context=error_context,
    )
