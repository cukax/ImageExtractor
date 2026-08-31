"""Typed contract for proof of address documents (utility bills, bank statements).

A proof of address is the third pillar of a KYC dossier, next to the identity card
and the invoice. It has no prebuilt model at any OCR vendor, which is exactly why
the router sends it down the visual grounding path instead.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import Field, field_validator

from .base import DocumentSchema


class ProofOfAddressSchema(DocumentSchema):
    """Canonical representation of a utility bill or equivalent address proof.

    Every field is optional for the same reason as on the identity card: a missing
    value must reach the validation node as a recoverable error rather than raise
    a Pydantic exception inside the extraction node.
    """

    # --- Holder -------------------------------------------------------------
    holder_name: Optional[str] = Field(default=None, description="Name the service is billed to.")

    # --- Issuer -------------------------------------------------------------
    service_provider: Optional[str] = Field(
        default=None,
        description="Utility or institution that issued the document.",
    )
    service_type: Optional[str] = Field(
        default=None,
        description="Kind of service billed, for example electricity, water or telephone.",
    )
    account_number: Optional[str] = Field(default=None, description="Customer or service number.")

    # --- Address ------------------------------------------------------------
    address: Optional[str] = Field(default=None, description="Full service address block.")
    postal_code: Optional[str] = Field(default=None, description="Five digit Mexican postal code.")
    city: Optional[str] = Field(default=None)
    state: Optional[str] = Field(default=None)

    # --- Dates and amount ---------------------------------------------------
    issue_date: Optional[str] = Field(default=None, description="Issue date as printed.")
    due_date: Optional[str] = Field(default=None)
    billing_period: Optional[str] = Field(default=None, description="Period covered, as printed.")
    total_amount: Optional[str] = Field(
        default=None,
        description="Amount due, kept as printed text rather than parsed.",
    )

    # Provider vocabularies mapped onto the canonical field names above.
    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "holder_name": (
            "Name",
            "CustomerName",
            "AccountHolder",
            "NOMBRE",
            "titular",
            "nombre_titular",
        ),
        "service_provider": (
            "VendorName",
            "MerchantName",
            "Issuer",
            "SupplierName",
            "proveedor",
            "emisor",
        ),
        "service_type": ("ServiceType", "Service", "tipo_servicio", "servicio"),
        "account_number": (
            "AccountNumber",
            "CustomerId",
            "ServiceNumber",
            "numero_cuenta",
            "numero_servicio",
        ),
        "address": ("Address", "ServiceAddress", "CustomerAddress", "DOMICILIO", "domicilio"),
        "postal_code": ("PostalCode", "ZipCode", "CP", "codigo_postal"),
        "city": ("City", "CIUDAD", "ciudad", "municipio"),
        "state": ("State", "StateName", "ESTADO", "estado", "entidad"),
        "issue_date": (
            "InvoiceDate",
            "IssueDate",
            "DateOfIssue",
            "fecha_emision",
            "fecha_factura",
        ),
        "due_date": ("DueDate", "PaymentDueDate", "fecha_limite", "fecha_vencimiento"),
        "billing_period": ("BillingPeriod", "ServicePeriod", "periodo", "periodo_facturado"),
        "total_amount": ("InvoiceTotal", "Total", "AmountDue", "total", "importe_total"),
    }

    @field_validator("postal_code", "account_number", mode="before")
    @classmethod
    def _normalize_code(cls, raw_value: Optional[str]) -> Optional[str]:
        """Uppercase machine readable codes and drop the whitespace OCR inserts."""
        if raw_value is None:
            return None
        token = "".join(str(raw_value).split()).upper()
        return token or None
