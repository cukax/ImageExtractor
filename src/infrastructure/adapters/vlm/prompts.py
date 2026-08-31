"""Prompt templates shared by every VLM adapter.

Centralizing the prompts guarantees that OpenAI, Anthropic and Ollama are asked
exactly the same question, which is the only way an A/B comparison between
providers measures the model rather than the prompt.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

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
{multi_page_section}{layout_section}{grounding_section}
Return ONLY the JSON object, with no markdown fences and no commentary."""

_LAYOUT_SECTION_TEMPLATE = """
Layout and orientation hints:
{layout_hints}
"""

_GROUNDING_SECTION = """
In addition to the schema properties, add one extra top level property named
"_bounding_boxes". Map every field you extracted to its location in the image as
{"x_min": ..., "y_min": ..., "width": ..., "height": ...}, using coordinates normalized
between 0.0 and 1.0 relative to the full image, where x_min and y_min are the top-left
corner. Omit any field whose location you cannot determine confidently.
"""

_MULTI_PAGE_SECTION_TEMPLATE = """
This document spans {page_count} images, supplied in order (image 1 is page 1).
They are pages of ONE single document, for example the front and the back of the same
card. Merge them into ONE JSON object: never return one object per page, and never
report the same field twice. When a field appears on more than one page, keep the
clearest occurrence.
"""


def build_primary_extraction_user_prompt(
    doc_type: str,
    json_schema: dict[str, Any],
    layout_hints: Optional[str] = None,
    request_bounding_boxes: bool = False,
    page_count: int = 1,
) -> str:
    """Build the user prompt for whole document extraction.

    Args:
        doc_type: Business document type injected as context.
        json_schema: JSON Schema the model must populate.
        layout_hints: Optional orientation or layout guidance.
        request_bounding_boxes: When True, also ask the model to ground each
            field. Grounding quality varies a lot between models, so it stays
            opt-in through configuration.
        page_count: Number of images supplied. Above one, the prompt explicitly
            forbids the model's default reflex of answering once per image.
    """
    layout_section = (
        _LAYOUT_SECTION_TEMPLATE.format(layout_hints=layout_hints.strip())
        if layout_hints and layout_hints.strip()
        else ""
    )
    multi_page_section = (
        _MULTI_PAGE_SECTION_TEMPLATE.format(page_count=page_count) if page_count > 1 else ""
    )
    return _PRIMARY_EXTRACTION_USER_TEMPLATE.format(
        doc_type=doc_type,
        json_schema=json.dumps(json_schema, indent=2, ensure_ascii=False),
        multi_page_section=multi_page_section,
        layout_section=layout_section,
        grounding_section=_GROUNDING_SECTION if request_bounding_boxes else "",
    )


#: Alias kept for readability at the call sites of the dossier grounding branch:
#: it is the very same template A, with grounding switched on.
VISUAL_GROUNDING_SYSTEM_PROMPT = (
    "You are an expert OCR vision assistant. Analyze the provided image(s) and extract "
    "all requested document fields. Return a valid JSON object matching the requested "
    "schema. Provide the normalized bounding box (x_min, y_min, width, height) for each "
    "extracted field based on a 0.0 to 1.0 scale. If a field is illegible, missing, or "
    "unreadable, set its value to null. Do NOT guess or hallucinate unreadable characters."
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


# --------------------------------------------------------------------------- #
# C. Page classification prompt (dossier splitter)
# --------------------------------------------------------------------------- #
PAGE_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an expert document triage assistant. You are shown ONE page of a scanned "
    "dossier and must identify what kind of page it is. Answer with a single label from "
    "the allowed list and nothing else."
)

PAGE_CLASSIFICATION_USER_PROMPT_TEMPLATE = """Classify this page into exactly one of the following types:

{allowed_types}

Guidance:
- INE_FRONT carries the photograph, the full name and the CURP.
- INE_BACK carries the machine readable zone, a barcode, or a plain reverse layout.
- PROOF_OF_ADDRESS is a utility bill or bank statement showing a service address.
- INVOICE is a commercial document with a subtotal, a tax amount and a total.
- Use UNKNOWN when the page matches none of the above, or is blank.

CRITICAL RULES:
1. Return ONLY the label, exactly as written in the list above.
2. Do NOT write conversational text, punctuation, explanations, or code blocks."""


def build_page_classification_user_prompt(allowed_types: Sequence[str]) -> str:
    """Render the classification prompt for one page of a dossier."""
    return PAGE_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        allowed_types="\n".join(f"- {page_type}" for page_type in allowed_types),
    )
