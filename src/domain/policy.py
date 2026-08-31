"""Workflow policy: the tuning knobs the graph nodes are allowed to know about.

Keeping this in the domain is what lets ``graph/nodes`` stay free of any
infrastructure import. The infrastructure layer projects its environment backed
``Settings`` onto this object, so the direction of dependency stays inwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .models import normalize_key
from .orchestrator_state import LogicalDocType
from .preprocessing import PreprocessingConfig
from .worker_state import ExtractionStrategy

#: Default worker routing table (specification phase 3). Identity documents have
#: mature prebuilt models at every OCR vendor, so they take the template based
#: path; everything else is read by a vision model that also grounds its answer.
DEFAULT_STRATEGY_BY_DOC_TYPE: dict[str, str] = {
    LogicalDocType.INE_COMBINED.value: ExtractionStrategy.DOC_INTELLIGENCE.value,
    LogicalDocType.INE_FRONT.value: ExtractionStrategy.DOC_INTELLIGENCE.value,
    LogicalDocType.INE_BACK.value: ExtractionStrategy.DOC_INTELLIGENCE.value,
    LogicalDocType.PASSPORT.value: ExtractionStrategy.DOC_INTELLIGENCE.value,
    LogicalDocType.MEXICAN_CEDULA.value: ExtractionStrategy.DOC_INTELLIGENCE.value,
    LogicalDocType.INVOICE.value: ExtractionStrategy.VLM_GROUNDING.value,
    LogicalDocType.PROOF_OF_ADDRESS.value: ExtractionStrategy.VLM_GROUNDING.value,
    LogicalDocType.UNKNOWN.value: ExtractionStrategy.VLM_GROUNDING.value,
}


@dataclass(frozen=True, slots=True)
class WorkflowPolicy:
    """Behavioural configuration of the OCR workflows."""

    max_retries: int = 3
    """Retry budget N: the number of ROI re-read passes before escalation."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    """Thresholds and sizes used by the preprocessing and cropping routines."""

    # --- Dossier engine ------------------------------------------------------
    pdf_render_dpi: int = 300
    """Resolution the dossier pages are rasterized at before anything else runs."""

    classification_max_dimension: int = 768
    """Longest side of the low resolution copy sent to the page classifier.

    Classification only needs the layout, not the glyphs, so shrinking the page
    first cuts the token bill of the cheapest and most frequent call in the run.
    """

    strategy_by_doc_type: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_STRATEGY_BY_DOC_TYPE)
    )
    """Worker routing table, keyed by logical document type."""

    default_strategy: str = ExtractionStrategy.VLM_GROUNDING.value
    """Strategy used for a document type absent from the routing table.

    Defaulting to visual grounding rather than to a template model is deliberate:
    a vision model degrades to best effort on an unexpected layout, while a
    prebuilt model returns nothing at all.
    """

    def strategy_for(self, doc_type: str) -> str:
        """Resolve the extraction strategy bound to a logical document type."""
        if not doc_type:
            return self.default_strategy
        # Compare on the normalized token so "INE_COMBINED", "ine_combined" and
        # "INECombined" all resolve to the same route.
        normalized = normalize_key(doc_type)
        for registered_type, strategy in self.strategy_by_doc_type.items():
            if normalize_key(registered_type) == normalized:
                return strategy
        return self.default_strategy
