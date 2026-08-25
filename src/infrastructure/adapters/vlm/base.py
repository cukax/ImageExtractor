"""Shared plumbing for the concrete VLM adapters.

Everything that is provider independent lives here: media type sniffing, base64
encoding, response sanitizing and JSON recovery. Each adapter is then reduced to
its SDK call, which keeps the differences between providers obvious.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Optional

from ....ports.vlm_port import UNREADABLE_TOKEN, VLMProviderError, VLMProviderPort

LOGGER = logging.getLogger(__name__)

_CODE_FENCE_PATTERN = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
_CONVERSATIONAL_PREFIXES = (
    "the value is",
    "the extracted value is",
    "extracted value:",
    "value:",
    "answer:",
    "result:",
)


def detect_media_type(image_bytes: bytes) -> str:
    """Sniff the media type of an in memory image from its magic bytes."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF8"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    # The pipeline only ever emits JPEG and PNG, so this is a safe last resort.
    return "image/jpeg"


def encode_base64(image_bytes: bytes) -> str:
    """Base64 encode an image payload as an ASCII string."""
    return base64.b64encode(image_bytes).decode("ascii")


def build_data_uri(image_bytes: bytes) -> str:
    """Build the data URI used by OpenAI compatible image inputs."""
    return f"data:{detect_media_type(image_bytes)};base64,{encode_base64(image_bytes)}"


def sanitize_scalar_response(raw_response: str) -> str:
    """Reduce a chatty model answer to the bare value the retry node expects.

    Models occasionally ignore the "plain string only" rule and wrap the answer
    in fences, quotes or a polite sentence. Repairing that here keeps every
    downstream validator simple and provider agnostic.
    """
    if raw_response is None:
        return UNREADABLE_TOKEN

    text = _CODE_FENCE_PATTERN.sub("", str(raw_response)).strip()
    if not text:
        return UNREADABLE_TOKEN

    # Keep the first non empty line: extra lines are always commentary here.
    for line in (candidate.strip() for candidate in text.splitlines()):
        if line:
            text = line
            break

    lowered = text.lower()
    for prefix in _CONVERSATIONAL_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    text = text.strip().strip("`").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    if not text or text.upper() == UNREADABLE_TOKEN:
        return UNREADABLE_TOKEN
    return text


def extract_json_object(raw_response: str, provider: str) -> dict[str, Any]:
    """Parse the first balanced JSON object contained in a model response.

    Raises:
        VLMProviderError: When no parseable JSON object can be recovered.
    """
    text = _CODE_FENCE_PATTERN.sub("", str(raw_response or "")).strip()
    if not text:
        raise VLMProviderError(provider, "The model returned an empty response.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fall back to scanning for the first balanced object, which survives models
    # that prepend a sentence despite the instructions.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)

    raise VLMProviderError(provider, "The model response did not contain a JSON object.")


class BaseVLMAdapter(VLMProviderPort):
    """Common configuration and post processing for the concrete VLM adapters."""

    def __init__(
        self,
        model: str,
        *,
        provider_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.provider_name = provider_name
        self.max_tokens = max_tokens
        # Temperature stays at zero by default: OCR repair is a transcription
        # task, and sampling diversity is pure downside here.
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    # -- Hooks implemented by each concrete adapter ------------------------- #
    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send one multimodal request and return the raw text response."""
        raise NotImplementedError

    # -- Port implementation ------------------------------------------------ #
    def analyze_roi_crop(self, crop_bytes: bytes, field_name: str, error_context: str) -> str:
        """Re-read a single field from a magnified crop (prompt template B)."""
        from .prompts import ROI_RETRY_SYSTEM_PROMPT, build_roi_retry_user_prompt

        if not crop_bytes:
            raise VLMProviderError(self.provider_name, "The ROI crop payload is empty.")

        raw_response = self._invoke(
            system_prompt=ROI_RETRY_SYSTEM_PROMPT,
            user_prompt=build_roi_retry_user_prompt(field_name, error_context),
            image_bytes=crop_bytes,
            # A single field never needs a long completion, and a tight budget
            # discourages the model from adding commentary.
            max_tokens=min(self.max_tokens, 256),
        )
        value = sanitize_scalar_response(raw_response)
        LOGGER.info(
            "ROI re-read via %s for field %r returned %r",
            self.provider_name,
            field_name,
            value,
        )
        return value

    def analyze_full_document(
        self,
        image_bytes: bytes,
        doc_type: str,
        json_schema: dict[str, Any],
        layout_hints: Optional[str] = None,
    ) -> str:
        """Extract a whole document in a single call (prompt template A)."""
        from .prompts import (
            PRIMARY_EXTRACTION_SYSTEM_PROMPT,
            build_primary_extraction_user_prompt,
        )

        if not image_bytes:
            raise VLMProviderError(self.provider_name, "The document payload is empty.")

        return self._invoke(
            system_prompt=PRIMARY_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_primary_extraction_user_prompt(
                doc_type=doc_type,
                json_schema=json_schema,
                layout_hints=layout_hints,
                request_bounding_boxes=getattr(self, "request_bounding_boxes", False),
            ),
            image_bytes=image_bytes,
            max_tokens=self.max_tokens,
        )
