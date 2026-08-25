"""Infrastructure layer: adapters and composition root."""

from __future__ import annotations

from .config import (
    Dependencies,
    ExtractorProvider,
    Settings,
    VLMProvider,
    build_dependencies,
    build_extractor,
    build_vlm_provider,
    get_settings,
)

__all__ = [
    "Dependencies",
    "ExtractorProvider",
    "Settings",
    "VLMProvider",
    "build_dependencies",
    "build_extractor",
    "build_vlm_provider",
    "get_settings",
]
