"""Concrete document extraction adapters."""

from __future__ import annotations

from .aws_textract import AWSTextractAdapter
from .azure_document_intelligence import AzureDocumentIntelligenceAdapter
from .google_docai import GoogleDocAIAdapter
from .native_vlm import NativeVLMExtractorAdapter, build_prompt_schema

__all__ = [
    "AWSTextractAdapter",
    "AzureDocumentIntelligenceAdapter",
    "GoogleDocAIAdapter",
    "NativeVLMExtractorAdapter",
    "build_prompt_schema",
]
