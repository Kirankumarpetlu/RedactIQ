"""
42-type PII taxonomy, copied verbatim (label spelling, casing, and grouping)
from the "Supported PII Labels (42 types)" table on the live model card:
https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi
(fetched and cross-checked against that page directly -- not transcribed
from memory). GLiNER2 PII Primary only ever emits labels from this list,
using these exact lowercase snake_case strings -- the model card's own
quick-start example passes labels the same way:
    labels = ["email", "phone_number", "person"]

If Fastino revises the model card's label set in a future model version,
re-fetch the card and update this list and PRESIDIO_BUILTIN_TO_TAXONOMY
below to match -- everything else in the pipeline is decoupled from the
exact label strings via `is_outside_taxonomy()`.
"""
from __future__ import annotations

# Group: Person / names
_PERSON_NAMES = [
    "person", "full_name", "first_name", "middle_name", "last_name",
    "date_of_birth",
]

# Group: Contact / address
_CONTACT_ADDRESS = [
    "email", "phone_number", "address", "street_address", "city",
    "state_or_region", "postal_code", "country",
]

# Group: Government / tax IDs
_GOVERNMENT_TAX_IDS = [
    "government_id", "national_id_number", "passport_number",
    "drivers_license_number", "license_number", "tax_id", "tax_number",
]

# Group: Banking / payment
_BANKING_PAYMENT = [
    "bank_account", "account_number", "routing_number", "iban",
    "payment_card", "card_number", "card_expiry", "card_cvv",
]

# Group: Digital identity
_DIGITAL_IDENTITY = [
    "username", "ip_address", "account_id", "sensitive_account_id",
]

# Group: Secrets / credentials
_SECRETS_CREDENTIALS = [
    "password", "secret", "api_key", "access_token", "recovery_code",
]

# Group: Sensitive dates
_SENSITIVE_DATES = [
    "sensitive_date", "document_date", "expiration_date", "transaction_date",
]

PII_TAXONOMY_LABELS = (
    _PERSON_NAMES
    + _CONTACT_ADDRESS
    + _GOVERNMENT_TAX_IDS
    + _BANKING_PAYMENT
    + _DIGITAL_IDENTITY
    + _SECRETS_CREDENTIALS
    + _SENSITIVE_DATES
)  # 6 + 8 + 7 + 8 + 4 + 5 + 4 = 42 labels, matches the model card's count

# GLiNER 2.5 is a general schema-extraction checkpoint, unlike the legacy
# privacy-filter checkpoint.  Supplying domain descriptions prevents generic
# terms such as ``person`` from being interpreted without PII context.
PII_LABEL_DESCRIPTIONS = {
    "person": "A person's name; do not include generic pronouns or clinical terms.",
    "full_name": "A complete personal name identifying an individual.",
    "first_name": "A given name identifying an individual.",
    "middle_name": "A middle name identifying an individual.",
    "last_name": "A family or surname identifying an individual.",
    "date_of_birth": "A person's date of birth, birthday, or birth date.",
    "email": "An email address belonging to a person or account.",
    "phone_number": "A telephone or mobile phone number.",
    "address": "A full residential or mailing address identifying a person.",
    "street_address": "A street address including house or apartment details.",
    "city": "A city in a person's address.",
    "state_or_region": "A state, province, or region in a person's address.",
    "postal_code": "A postal or ZIP code in an address.",
    "country": "A country in a person's address or identity record.",
    "government_id": "A government-issued personal identification number.",
    "national_id_number": "A national identity or social-security-style number.",
    "passport_number": "A passport number.",
    "drivers_license_number": "A driver's license number.",
    "license_number": "A personal professional or government license number.",
    "tax_id": "A personal tax identification number.",
    "tax_number": "A tax registration or taxpayer number.",
    "bank_account": "A bank account number.",
    "account_number": "A financial or customer account number.",
    "routing_number": "A bank routing or sort code.",
    "iban": "An International Bank Account Number.",
    "payment_card": "A payment or credit card number.",
    "card_number": "A payment card number.",
    "card_expiry": "A payment card expiration date.",
    "card_cvv": "A payment-card CVV, CVC, or security code.",
    "username": "A username, handle, or login identifier.",
    "ip_address": "An IPv4 or IPv6 network address.",
    "account_id": "An account identifier tied to a person or service.",
    "sensitive_account_id": "A sensitive account, member, or customer identifier.",
    "password": "A password or passphrase.",
    "secret": "A secret value that grants access or identifies a person.",
    "api_key": "An API key or access credential.",
    "access_token": "An access, bearer, refresh, or session token.",
    "recovery_code": "A recovery, backup, or one-time security code.",
    "sensitive_date": "A sensitive date associated with a person or account.",
    "document_date": "A date printed on an identity or financial document.",
    "expiration_date": "An expiration date on an identity or financial document.",
    "transaction_date": "A date associated with a financial transaction.",
}

# Medical / PHI labels emitted by the Presidio Medical recognizers. These are
# deliberately NOT part of the 42-type taxonomy above -- that's what drives
# the "Label Outside Taxonomy? Yes" routing to the GLiNER2 Vanilla fallback.
PHI_MEDICAL_LABELS = [
    "MEDICAL_LICENSE",  # Presidio built-in MedicalLicenseRecognizer
    "NPI_NUMBER",
    "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_BENEFICIARY_NUM",
    "DEVICE_IDENTIFIER",
    "ICD10_CODE",
]

PHI_LABEL_DESCRIPTIONS = {
    "medical_license": "A clinician's medical or professional license number.",
    "npi_number": "A National Provider Identifier number.",
    "medical_record_number": "A patient medical-record, chart, or MRN identifier.",
    "health_plan_beneficiary_num": "A health-insurance beneficiary, member, or policy identifier.",
    "device_identifier": "A medical-device UDI, serial, or lot identifier.",
    "icd10_code": "An ICD-10 diagnosis code in a patient health record.",
}


def get_entity_schema(profile: str = "pii") -> dict[str, str]:
    """Return a documented model-facing schema without changing legacy defaults."""
    normalized = (profile or "pii").strip().lower()
    if normalized == "pii":
        return dict(PII_LABEL_DESCRIPTIONS)
    if normalized == "pii_phi":
        return {**PII_LABEL_DESCRIPTIONS, **PHI_LABEL_DESCRIPTIONS}
    raise ValueError("GLiNER entity_schema must be one of: pii, pii_phi.")

# Presidio's own built-in regex/checksum recognizers (credit card, IBAN, US
# SSN/ITIN, email, phone, IP, person via spaCy NER, etc.) emit their own
# native, uppercase entity-type names. Where one of those clearly
# corresponds to a real label in the 42-type taxonomy above, map it so it
# resolves as "in taxonomy" instead of spuriously triggering a fallback
# route. Only confident, unambiguous matches are mapped here -- anything
# without a clean equivalent (URL, CRYPTO, LOCATION, DATE_TIME, NRP) is left
# unmapped on purpose: those legitimately fall outside the 42-type taxonomy
# (there is no "url" or "crypto_wallet" label in it) or are too ambiguous to
# guess a mapping for (e.g. Presidio's generic LOCATION/DATE_TIME/NRP don't
# reliably correspond to any single taxonomy label), so they correctly fall
# through to the GLiNER2 Vanilla fallback for a confidence check instead of
# being force-mapped to something that might be wrong.
PRESIDIO_BUILTIN_TO_TAXONOMY = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone_number",
    "CREDIT_CARD": "payment_card",
    "US_SSN": "national_id_number",
    "US_ITIN": "tax_id",
    "US_DRIVER_LICENSE": "drivers_license_number",
    "US_PASSPORT": "passport_number",
    "US_BANK_NUMBER": "bank_account",
    "IBAN_CODE": "iban",
    "IP_ADDRESS": "ip_address",
    "PERSON": "person",
}

_TAXONOMY_SET = set(PII_TAXONOMY_LABELS)


def canonical_entity_type(entity_type: str) -> str:
    """Policy label; the original detector label is retained in audit output."""
    label = entity_type.strip().lower()
    aliases = {
        **{name: "PERSON" for name in ("person", "full_name", "first_name", "middle_name", "last_name")},
        **{name: "LOCATION" for name in ("location", "address", "street_address", "city", "state_or_region", "country")},
        "org": "ORGANIZATION", "organization": "ORGANIZATION",
        "email_address": "EMAIL", "email": "EMAIL",
        "phone": "PHONE_NUMBER", "phone_number": "PHONE_NUMBER",
        "date_time": "SENSITIVE_DATE", "sensitive_date": "SENSITIVE_DATE",
    }
    return aliases.get(label, label.upper())


def is_outside_taxonomy(entity_type: str) -> bool:
    """
    True  -> label is NOT in the 42-type taxonomy (directly or via the
              Presidio-builtin alias map) => route to GLiNER2 Vanilla
              fallback for a confidence check ("Yes" branch of the diagram).
    False -> label IS in the 42-type taxonomy => Resolved, no fallback
              needed ("No" branch of the diagram).
    """
    canonical = PRESIDIO_BUILTIN_TO_TAXONOMY.get(entity_type, entity_type)
    return canonical not in _TAXONOMY_SET
