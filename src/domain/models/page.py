"""Value object describing one rasterized page of a dossier.

Produced by any :class:`~src.ports.pdf_port.PDFProcessorPort` implementation and
consumed by the semantic clustering node. Like every other domain model it holds
plain bytes and integers, so a page can cross the port boundary without dragging
PyMuPDF, Poppler or any other rendering library into the inner layers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RenderedPage(BaseModel):
    """One page of a source document, rasterized at a known resolution."""

    model_config = ConfigDict(frozen=True)

    page_index: int = Field(ge=0, description="Zero based index of the page in the source file.")
    image_bytes: bytes = Field(description="Full resolution raster of the page, PNG encoded.")
    width: int = Field(gt=0, description="Raster width in pixels.")
    height: int = Field(gt=0, description="Raster height in pixels.")
    dpi: int = Field(default=300, gt=0, description="Resolution the page was rendered at.")

    @property
    def page_number(self) -> int:
        """One based page number, the convention printed on the document itself."""
        return self.page_index + 1
