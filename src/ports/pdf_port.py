"""Port describing any PDF rasterization service."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..domain.models import RenderedPage


class PDFProcessingError(RuntimeError):
    """Raised when a document cannot be rasterized.

    Adapters wrap their library specific exceptions in this type so the graph
    never has to import PyMuPDF, pdf2image or Poppler to handle a corrupt upload.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


class PDFProcessorPort(ABC):
    """Driven port for turning a multi page document into page rasters.

    The dossier engine depends on this abstraction only, which is what lets the
    rendering library be swapped without touching a single node.
    """

    #: Stable identifier used in logs and in provider metadata.
    provider_name: str = "unknown"

    @abstractmethod
    def render_pages(self, document_bytes: bytes, *, dpi: int = 300) -> list[RenderedPage]:
        """Rasterize every page of a document.

        Args:
            document_bytes: Raw PDF payload. Implementations should also accept a
                bare image and treat it as a single page dossier, so a caller
                never has to branch on the upload format.
            dpi: Target resolution. 300 DPI is the floor at which small print on
                an identity card survives rasterization.

        Returns:
            One :class:`~src.domain.models.page.RenderedPage` per page, in
            document order.

        Raises:
            PDFProcessingError: When the payload cannot be decoded or rendered.
        """

    def page_count(self, document_bytes: bytes) -> int:
        """Report the page count without rasterizing.

        The default implementation renders, which is correct but wasteful;
        adapters whose library exposes a cheap page count override it.
        """
        return len(self.render_pages(document_bytes))

    def close(self) -> None:
        """Release any resource held by the adapter. Optional for most libraries."""
        return None
