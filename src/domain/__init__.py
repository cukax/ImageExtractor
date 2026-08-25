"""Domain layer: pure business logic with no infrastructure dependency."""

from __future__ import annotations

from .state import DocumentStatus, OCRState, initial_state

__all__ = ["DocumentStatus", "OCRState", "initial_state"]
