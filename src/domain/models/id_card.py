"""Typed contract for identity documents (Mexican INE / IFE, passports, IDs)."""

from __future__ import annotations

from typing import ClassVar, Optional

from pydantic import Field, field_validator

from .base import DocumentSchema


class IDCardSchema(DocumentSchema):
    """Canonical representation of an identity card.

    Every field is optional on purpose: OCR is a lossy process and a missing
    value must reach the validation node as a recoverable error rather than
    blowing up the graph with a Pydantic exception.
    """

    # --- Identity -----------------------------------------------------------
    full_name: Optional[str] = Field(default=None, description="Complete name as printed.")
    given_names: Optional[str] = Field(default=None, description="First and middle names.")
    paternal_surname: Optional[str] = Field(default=None, description="First (paternal) surname.")
    maternal_surname: Optional[str] = Field(default=None, description="Second (maternal) surname.")
    sex: Optional[str] = Field(default=None, description="Sex marker, normalized to H / M / X.")
    date_of_birth: Optional[str] = Field(default=None, description="Birth date as printed.")
    nationality: Optional[str] = Field(default=None)

    # --- Document identifiers ----------------------------------------------
    document_number: Optional[str] = Field(default=None, description="Primary document number.")
    curp: Optional[str] = Field(default=None, description="Mexican population registry code.")
    voter_key: Optional[str] = Field(default=None, description="INE 18 character elector key.")
    registration_year: Optional[str] = Field(default=None, description="Year of registration.")
    issue_date: Optional[str] = Field(default=None)
    expiration_date: Optional[str] = Field(default=None)
    issuing_authority: Optional[str] = Field(default=None)
    issuing_state: Optional[str] = Field(default=None, description="State or entity code.")

    # --- Address ------------------------------------------------------------
    address: Optional[str] = Field(default=None, description="Full address block.")
    postal_code: Optional[str] = Field(default=None)

    # --- Machine readable zone ---------------------------------------------
    mrz_line_1: Optional[str] = Field(default=None)
    mrz_line_2: Optional[str] = Field(default=None)

    # Provider vocabularies mapped onto the canonical field names above.
    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "full_name": ("Name", "FullName", "NOMBRE", "nombre_completo", "holder_name"),
        "given_names": ("FirstName", "GivenName", "GivenNames", "given_name", "nombres"),
        "paternal_surname": ("LastName", "Surname", "FamilyName", "apellido_paterno"),
        "maternal_surname": ("MiddleName", "SecondSurname", "apellido_materno"),
        "sex": ("Sex", "Gender", "SEXO"),
        "date_of_birth": ("DateOfBirth", "DOB", "BirthDate", "FECHA_NACIMIENTO", "fecha_nacimiento"),
        "nationality": ("Nationality", "CountryRegion", "NACIONALIDAD"),
        "document_number": ("DocumentNumber", "IdNumber", "ID_NUMBER", "numero_documento"),
        "curp": ("CURP", "Curp", "PersonalNumber", "curp_code"),
        "voter_key": (
            "VoterKey",
            "ClaveDeElector",
            "CLAVE_DE_ELECTOR",
            "clave_elector",
            "elector_key",
        ),
        "registration_year": ("RegistrationYear", "ANO_REGISTRO", "anio_registro"),
        "issue_date": ("DateOfIssue", "IssueDate", "ISSUE_DATE", "fecha_emision"),
        "expiration_date": (
            "DateOfExpiration",
            "ExpirationDate",
            "EXPIRATION_DATE",
            "vigencia",
            "fecha_vigencia",
        ),
        "issuing_authority": ("IssuingAuthority", "Issuer", "autoridad_emisora"),
        "issuing_state": ("StateName", "IssuingState", "ESTADO", "entidad"),
        "address": ("Address", "ADDRESS", "DOMICILIO", "domicilio"),
        "postal_code": ("PostalCode", "ZipCode", "codigo_postal"),
        "mrz_line_1": ("MachineReadableZone", "MRZLine1", "mrz1"),
        "mrz_line_2": ("MRZLine2", "mrz2"),
    }

    @field_validator("sex", mode="before")
    @classmethod
    def _normalize_sex(cls, raw_value: Optional[str]) -> Optional[str]:
        """Normalize the sex marker so downstream regex rules stay simple."""
        if raw_value is None:
            return None
        token = str(raw_value).strip().upper()
        if not token:
            return None
        # Spanish "HOMBRE" / "MUJER" and English "MALE" / "FEMALE" both collapse
        # to the single letter printed on the card.
        mapping = {"HOMBRE": "H", "MUJER": "M", "MALE": "H", "FEMALE": "M", "M": "M", "H": "H"}
        return mapping.get(token, token[:1])

    @field_validator(
        "curp",
        "voter_key",
        "document_number",
        "postal_code",
        "mrz_line_1",
        "mrz_line_2",
        mode="before",
    )
    @classmethod
    def _normalize_code(cls, raw_value: Optional[str]) -> Optional[str]:
        """Uppercase machine readable codes and drop the whitespace OCR inserts."""
        if raw_value is None:
            return None
        token = "".join(str(raw_value).split()).upper()
        return token or None
