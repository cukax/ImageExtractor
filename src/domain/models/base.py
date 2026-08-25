"""Shared value objects used by every document schema and every adapter.

This module belongs to the domain layer: it must never import an SDK, an HTTP
client or any other piece of infrastructure. Adapters translate their native
provider payloads into the structures defined here, which keeps the LangGraph
workflow completely provider agnostic.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

# Normalized bounding box expressed as (x, y, width, height) in the 0.0 - 1.0 range.
BBoxTuple = tuple[float, float, float, float]

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_CURRENCY_NOISE = re.compile(r"[^0-9,.\-]")
# Thin, narrow and non breaking spaces frequently produced by OCR engines.
_EXOTIC_SPACES = (" ", " ", " ", " ")


def normalize_key(raw_key: str) -> str:
    """Reduce a provider specific field name to a comparable token.

    "Vendor Name", "vendor_name" and "VendorName" all collapse to "vendorname",
    which lets a single alias table serve every provider.
    """
    return _NON_ALPHANUMERIC.sub("", raw_key.lower())


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a float into the closed [lower, upper] interval."""
    return max(lower, min(upper, value))


def parse_decimal(raw_value: Any) -> Optional[Decimal]:
    """Parse a monetary or numeric OCR string into a Decimal.

    Handles currency symbols, exotic whitespace and both the anglo ("1,234.56")
    and the continental ("1.234,56") separator conventions. Returns None when
    the text cannot be interpreted as a number, which the validation layer then
    reports as a recoverable field level error.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, Decimal):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return Decimal(str(raw_value))

    text = str(raw_value)
    for exotic_space in _EXOTIC_SPACES:
        text = text.replace(exotic_space, " ")
    text = text.strip()
    if not text:
        return None

    # Accounting notation wraps negative amounts in parentheses.
    negative = text.startswith("(") and text.endswith(")")
    text = _CURRENCY_NOISE.sub("", text)
    if not text or text in {"-", ",", "."}:
        return None

    last_comma = text.rfind(",")
    last_dot = text.rfind(".")
    if last_comma > last_dot:
        # Continental notation: the comma is the decimal separator.
        text = text.replace(".", "").replace(",", ".")
    else:
        # Anglo notation: commas are thousand separators.
        text = text.replace(",", "")

    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return -parsed if negative else parsed


class BoundingBox(BaseModel):
    """Axis aligned region of a page in normalized coordinates.

    Normalized coordinates are deliberate: they survive the resizing performed
    during preprocessing, so the same box can crop either the optimized image
    sent to the extraction provider or the full resolution original used by the
    high precision VLM retry.
    """

    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, le=1.0, description="Left edge, normalized to page width.")
    y: float = Field(ge=0.0, le=1.0, description="Top edge, normalized to page height.")
    width: float = Field(ge=0.0, le=1.0, description="Box width, normalized to page width.")
    height: float = Field(ge=0.0, le=1.0, description="Box height, normalized to page height.")
    page: int = Field(default=0, ge=0, description="Zero based page index.")

    @classmethod
    def from_absolute(
        cls,
        x: float,
        y: float,
        width: float,
        height: float,
        page_width: float,
        page_height: float,
        page: int = 0,
    ) -> "BoundingBox":
        """Build a normalized box from absolute page units (pixels, inches, points)."""
        if page_width <= 0 or page_height <= 0:
            raise ValueError("Page dimensions must be strictly positive.")
        normalized_x = _clamp(x / page_width)
        normalized_y = _clamp(y / page_height)
        return cls(
            x=normalized_x,
            y=normalized_y,
            width=_clamp(width / page_width, upper=1.0 - normalized_x),
            height=_clamp(height / page_height, upper=1.0 - normalized_y),
            page=page,
        )

    @classmethod
    def from_polygon(
        cls,
        polygon: Sequence[float],
        page_width: float,
        page_height: float,
        page: int = 0,
    ) -> "BoundingBox":
        """Build the enclosing box of a flat [x1, y1, x2, y2, ...] polygon.

        Azure Document Intelligence returns polygons in this shape, expressed in
        the same unit as page_width and page_height (usually inches).
        """
        if len(polygon) < 4 or len(polygon) % 2 != 0:
            raise ValueError("A polygon needs an even number of at least four coordinates.")
        xs = [float(value) for value in polygon[0::2]]
        ys = [float(value) for value in polygon[1::2]]
        return cls.from_absolute(
            x=min(xs),
            y=min(ys),
            width=max(xs) - min(xs),
            height=max(ys) - min(ys),
            page_width=page_width,
            page_height=page_height,
            page=page,
        )

    @classmethod
    def from_normalized_vertices(cls, vertices: Sequence[Any], page: int = 0) -> "BoundingBox":
        """Build a box from Google Document AI normalized_vertices objects."""
        xs = [_clamp(float(getattr(vertex, "x", 0.0) or 0.0)) for vertex in vertices]
        ys = [_clamp(float(getattr(vertex, "y", 0.0) or 0.0)) for vertex in vertices]
        if not xs or not ys:
            raise ValueError("At least one vertex is required to build a bounding box.")
        left, top = min(xs), min(ys)
        return cls(
            x=left,
            y=top,
            width=_clamp(max(xs) - left, upper=1.0 - left),
            height=_clamp(max(ys) - top, upper=1.0 - top),
            page=page,
        )

    @classmethod
    def from_tuple(cls, values: Sequence[float], page: int = 0) -> "BoundingBox":
        """Rebuild a box from the (x, y, w, h) tuple persisted in the graph state."""
        x, y, width, height = (float(value) for value in values)
        left, top = _clamp(x), _clamp(y)
        return cls(
            x=left,
            y=top,
            width=_clamp(width, upper=1.0 - left),
            height=_clamp(height, upper=1.0 - top),
            page=page,
        )

    def with_padding(self, ratio: float) -> "BoundingBox":
        """Expand the box by ratio on each side, clamped to the page.

        The targeted ROI retry uses a 10% margin so that ascenders, accents and
        thin borders are not sliced off the crop handed to the VLM.
        """
        if ratio <= 0:
            return self
        pad_x = self.width * ratio
        pad_y = self.height * ratio
        left = _clamp(self.x - pad_x)
        top = _clamp(self.y - pad_y)
        return BoundingBox(
            x=left,
            y=top,
            width=_clamp(self.width + 2 * pad_x, upper=1.0 - left),
            height=_clamp(self.height + 2 * pad_y, upper=1.0 - top),
            page=self.page,
        )

    def to_tuple(self) -> BBoxTuple:
        """Return the (x, y, width, height) tuple stored in the LangGraph state."""
        return (self.x, self.y, self.width, self.height)

    def to_pixels(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Project the box onto a raster of the given size as (left, top, right, bottom)."""
        left = int(round(self.x * image_width))
        top = int(round(self.y * image_height))
        right = int(round((self.x + self.width) * image_width))
        bottom = int(round((self.y + self.height) * image_height))
        # Guarantee a non degenerate crop even for hairline boxes.
        left = min(left, max(image_width - 1, 0))
        top = min(top, max(image_height - 1, 0))
        right = min(max(right, left + 1), image_width)
        bottom = min(max(bottom, top + 1), image_height)
        return (left, top, right, bottom)

    @property
    def is_degenerate(self) -> bool:
        """True when the box has no usable area and therefore cannot be cropped."""
        return self.width <= 0.0 or self.height <= 0.0


class ExtractedField(BaseModel):
    """A single field as returned by an extraction provider, before validation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Field name as reported by the provider.")
    value: Optional[str] = Field(default=None, description="Raw text value, None when illegible.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = Field(
        default=None,
        description="Location of the field, required by the targeted ROI retry.",
    )


class ExtractionResult(BaseModel):
    """Provider agnostic contract returned by every DocumentExtractorPort."""

    doc_type: str
    provider: str = Field(description="Identifier of the adapter that produced the result.")
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    structured_extras: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Nested values such as invoice line items, which do not fit the flat "
            "ExtractedField shape but still belong to the typed schema."
        ),
    )
    page_count: int = Field(default=1, ge=0)
    warnings: list[str] = Field(default_factory=list)
    raw_payload: Optional[dict[str, Any]] = Field(
        default=None,
        description="Trimmed provider response, kept for auditing and debugging.",
    )

    def values(self) -> dict[str, Optional[str]]:
        """Flatten the result into a {field_name: value} mapping."""
        return {name: field.value for name, field in self.fields.items()}

    def bounding_boxes(self) -> dict[str, BBoxTuple]:
        """Flatten the located fields into a {field_name: (x, y, w, h)} mapping."""
        return {
            name: field.bounding_box.to_tuple()
            for name, field in self.fields.items()
            if field.bounding_box is not None and not field.bounding_box.is_degenerate
        }


class DocumentSchema(BaseModel):
    """Base class for every typed document contract (ID card, invoice, form).

    Subclasses declare ALIASES to map the heterogeneous field names used by
    Azure, AWS, Google and the native VLM onto one canonical domain vocabulary.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {}

    def to_flat_dict(self) -> dict[str, Any]:
        """Return the JSON safe {field: value} projection kept in the graph state.

        The graph state holds a flat dictionary rather than the model instance so
        that it stays serializable for any LangGraph checkpointer (Postgres,
        Redis, SQLite) and so that the ROI retry can patch a single field.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_flat_dict(cls, data: Mapping[str, Any]) -> "DocumentSchema":
        """Rebuild the typed document from the flat state dictionary.

        Used after the ROI retry has patched a field with raw model output, so
        the schema level validators run again over the repaired document.
        """
        return cls.model_validate(dict(data))

    @classmethod
    def alias_lookup(cls) -> dict[str, str]:
        """Build the {normalized_provider_key: canonical_field} translation table."""
        lookup: dict[str, str] = {
            normalize_key(field_name): field_name for field_name in cls.model_fields
        }
        for canonical_field, aliases in cls.ALIASES.items():
            for alias in aliases:
                lookup[normalize_key(alias)] = canonical_field
        return lookup

    @classmethod
    def from_extraction(
        cls,
        result: ExtractionResult,
    ) -> tuple["DocumentSchema", dict[str, BBoxTuple], list[str]]:
        """Project a raw ExtractionResult onto this typed schema.

        Returns the parsed document, the bounding boxes re-keyed by canonical
        field name, and the list of provider fields that could not be mapped.
        """
        lookup = cls.alias_lookup()
        payload: dict[str, Any] = {}
        boxes: dict[str, BBoxTuple] = {}
        unmapped: list[str] = []

        for provider_name, field in result.fields.items():
            canonical_field = lookup.get(normalize_key(provider_name))
            if canonical_field is None:
                unmapped.append(provider_name)
                continue
            # The first provider field wins, so a fuzzy alias never overwrites a
            # value already supplied by an exact field name match.
            if payload.get(canonical_field) in (None, ""):
                payload[canonical_field] = field.value
            if field.bounding_box is not None and not field.bounding_box.is_degenerate:
                boxes.setdefault(canonical_field, field.bounding_box.to_tuple())

        # Nested provider payloads (invoice line items, address blocks) are merged
        # last so a flat field value always wins over a structured duplicate.
        for extra_name, extra_value in result.structured_extras.items():
            canonical_field = lookup.get(normalize_key(extra_name))
            if canonical_field is None:
                unmapped.append(extra_name)
                continue
            if payload.get(canonical_field) in (None, "", [], {}):
                payload[canonical_field] = extra_value

        return cls.model_validate(payload), boxes, unmapped
