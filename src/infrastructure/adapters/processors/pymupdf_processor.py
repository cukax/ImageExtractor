"""PyMuPDF based PDF rasterization adapter."""

from __future__ import annotations

import logging
from typing import Any

from ....domain.models import RenderedPage
from ....ports.pdf_port import PDFProcessingError, PDFProcessorPort

LOGGER = logging.getLogger(__name__)

#: Magic bytes every PDF starts with. Sniffing them is what lets a caller hand
#: this adapter a bare photograph without first asking what it is holding.
_PDF_MAGIC = b"%PDF"

#: PDF user space is defined at 72 points per inch, so this is the divisor that
#: turns a target DPI into a PyMuPDF zoom factor.
_PDF_POINTS_PER_INCH = 72.0


class PyMuPDFAdapter(PDFProcessorPort):
    """Implements PDFProcessorPort on top of PyMuPDF (fitz).

    PyMuPDF is imported lazily, like every other SDK in this layer, so a
    deployment that only processes single images never needs it installed.
    """

    provider_name = "pymupdf"

    def __init__(self, *, default_dpi: int = 300, alpha: bool = False) -> None:
        self._default_dpi = default_dpi
        # Rendering without an alpha channel keeps the PNG a third smaller and
        # matches what every downstream OCR service expects anyway.
        self._alpha = alpha

    @property
    def _fitz(self) -> Any:
        """Import PyMuPDF on first use, with an actionable error when absent.

        The modern package name is ``pymupdf``; ``fitz`` is the legacy alias,
        kept as a fallback for older releases but no longer imported first,
        since it emits a deprecation warning on every use.
        """
        try:
            import pymupdf  # noqa: PLC0415 - lazy SDK import is the convention here

            return pymupdf
        except ImportError:
            pass
        try:
            import fitz  # noqa: PLC0415 - legacy alias, pre 1.24.3 releases

            return fitz
        except ImportError as error:  # pragma: no cover - environment dependent
            raise PDFProcessingError(
                self.provider_name,
                "The pymupdf package is not installed. Run: pip install pymupdf",
            ) from error

    # -- Port implementation ------------------------------------------------ #
    def render_pages(self, document_bytes: bytes, *, dpi: int = 300) -> list[RenderedPage]:
        """Rasterize every page of a PDF, or wrap a bare image as a single page."""
        if not document_bytes:
            raise PDFProcessingError(self.provider_name, "The document payload is empty.")

        resolution = dpi or self._default_dpi
        if not document_bytes.startswith(_PDF_MAGIC):
            LOGGER.info("The payload is not a PDF; treating it as a one page dossier.")
            return [self._single_image_page(document_bytes, resolution)]

        fitz = self._fitz
        zoom = resolution / _PDF_POINTS_PER_INCH
        pages: list[RenderedPage] = []

        try:
            with fitz.open(stream=document_bytes, filetype="pdf") as document:
                for page_index, page in enumerate(document):
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(zoom, zoom),
                        alpha=self._alpha,
                    )
                    pages.append(
                        RenderedPage(
                            page_index=page_index,
                            image_bytes=pixmap.tobytes("png"),
                            width=pixmap.width,
                            height=pixmap.height,
                            dpi=resolution,
                        )
                    )
        except PDFProcessingError:
            raise
        except Exception as error:  # noqa: BLE001 - library exceptions are wrapped by design
            raise PDFProcessingError(self.provider_name, str(error)) from error

        LOGGER.info("Rasterized %d page(s) at %d DPI", len(pages), resolution)
        return pages

    def page_count(self, document_bytes: bytes) -> int:
        """Read the page count from the PDF header instead of rendering."""
        if not document_bytes:
            raise PDFProcessingError(self.provider_name, "The document payload is empty.")
        if not document_bytes.startswith(_PDF_MAGIC):
            return 1
        try:
            with self._fitz.open(stream=document_bytes, filetype="pdf") as document:
                return int(document.page_count)
        except PDFProcessingError:
            raise
        except Exception as error:  # noqa: BLE001 - library exceptions are wrapped by design
            raise PDFProcessingError(self.provider_name, str(error)) from error

    # -- Helpers ------------------------------------------------------------ #
    def _single_image_page(self, image_bytes: bytes, dpi: int) -> RenderedPage:
        """Wrap a standalone image as a one page dossier.

        The pixels are passed through untouched: re-encoding an image the caller
        already supplied at full resolution would only lose detail.
        """
        try:
            from PIL import Image  # noqa: PLC0415 - lazy import, mirrors the SDK convention
            import io

            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except Exception as error:  # noqa: BLE001 - a corrupt upload must not crash the graph
            raise PDFProcessingError(
                self.provider_name,
                f"The payload is neither a PDF nor a decodable image: {error}",
            ) from error

        return RenderedPage(
            page_index=0,
            image_bytes=image_bytes,
            width=width,
            height=height,
            dpi=dpi,
        )
