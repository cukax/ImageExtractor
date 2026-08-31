"""Concrete VLM provider adapters."""

from __future__ import annotations

from .anthropic_vlm import AnthropicVLMAdapter
from .azure_foundry_vlm import AzureAIFoundryVLMAdapter
from .base import BaseVLMAdapter, extract_json_object, sanitize_scalar_response
from .ollama_vlm import OllamaVLMAdapter
from .openai_vlm import OpenAIVLMAdapter

__all__ = [
    "AnthropicVLMAdapter",
    "AzureAIFoundryVLMAdapter",
    "BaseVLMAdapter",
    "OllamaVLMAdapter",
    "OpenAIVLMAdapter",
    "extract_json_object",
    "sanitize_scalar_response",
]
