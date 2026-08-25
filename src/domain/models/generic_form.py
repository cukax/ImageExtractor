"""Fallback contract for semi structured documents without a dedicated schema."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import Field

from .base import BBoxTuple, DocumentSchema, ExtractionResult


class GenericFormSchema(DocumentSchema):
    """Schema-less container for key / value documents.

    Used for the "Form" document type, where the field set is only known at
    runtime. It keeps the same contract as the typed schemas so the graph nodes
    never need to branch on the document type.
    """

    fields: dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Every key / value pair reported by the extraction provider.",
    )

    def to_flat_dict(self) -> dict[str, Any]:
        """Expose the dynamic keys at the top level, like every typed schema does."""
        return dict(self.fields)

    @classmethod
    def from_flat_dict(cls, data: Mapping[str, Any]) -> "GenericFormSchema":
        """Wrap a flat state dictionary back into the dynamic container."""
        return cls(fields={key: None if value is None else str(value) for key, value in data.items()})

    @classmethod
    def from_extraction(
        cls,
        result: ExtractionResult,
    ) -> tuple["GenericFormSchema", dict[str, BBoxTuple], list[str]]:
        """Accept every provider field verbatim, so nothing is ever unmapped."""
        document = cls(fields=result.values())
        return document, result.bounding_boxes(), []
