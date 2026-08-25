"""Workflow policy: the tuning knobs the graph nodes are allowed to know about.

Keeping this in the domain is what lets ``graph/nodes.py`` stay free of any
infrastructure import. The infrastructure layer projects its environment backed
``Settings`` onto this object, so the direction of dependency stays inwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .preprocessing import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class WorkflowPolicy:
    """Behavioural configuration of the OCR workflow."""

    max_retries: int = 3
    """Retry budget N: the number of ROI re-read passes before escalation."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    """Thresholds and sizes used by the preprocessing and cropping routines."""
