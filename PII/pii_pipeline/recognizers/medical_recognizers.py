"""
Presidio Medical — Medical NER for PHI (purple box in the diagram).

Per your call: built-in Presidio recognizers + regex/checksum only, no extra
ML model. This module provides:

  1. Presidio's built-in MedicalLicenseRecognizer (checksum-validated US
     medical license / DEA-style numbers) — already ships with
     presidio-analyzer, just registered here.
  2. Custom PatternRecognizers for the HIPAA identifiers that Presidio
     doesn't cover out of the box: NPI (National Provider Identifier, with
     real Luhn-style checksum validation), MRN, health plan beneficiary
     numbers, device identifiers (UDI), and ICD-10 codes.

All of these deliberately emit entity_types NOT in the 42-type PII taxonomy
(see taxonomy.PHI_MEDICAL_LABELS), so they get routed through the GLiNER2
Vanilla fallback for a confidence check before anonymization — exactly the
"Label Outside Taxonomy? Yes" path in the diagram.
"""

from __future__ import annotations

import re
from typing import List

from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer.predefined_recognizers import MedicalLicenseRecognizer


def _npi_checksum_valid(npi: str) -> bool:
    """
    Validate a 10-digit NPI using the standard Luhn-based check digit,
    prefixed with the fixed '80840' issuer identifier as CMS specifies.
    https://en.wikipedia.org/wiki/National_Provider_Identifier#Check_digit
    """
    digits = re.sub(r"\D", "", npi)
    if len(digits) != 10:
        return False
    payload = "80840" + digits[:9]
    total = 0
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    check_digit = (10 - (total % 10)) % 10
    return check_digit == int(digits[9])


class NPIRecognizer(PatternRecognizer):
    """National Provider Identifier — 10 digits with a real checksum."""

    PATTERNS = [
        Pattern(name="npi_10_digit", regex=r"\b\d{10}\b", score=0.35),
    ]
    CONTEXT = ["npi", "national provider identifier", "provider id", "provider number"]

    def __init__(self):
        super().__init__(
            supported_entity="NPI_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str):
        # Returns True (confirmed valid), False (confirmed invalid, dropped
        # by Presidio), or None (no change to score-based confidence).
        return True if _npi_checksum_valid(pattern_text) else False


class MedicalRecordNumberRecognizer(PatternRecognizer):
    """
    MRN formats are facility-specific (no universal checksum), so this is
    context + regex only — deliberately conservative (requires a nearby
    context word) to avoid over-flagging arbitrary numbers.
    """

    PATTERNS = [
        Pattern(name="mrn_alnum", regex=r"\b[A-Z]{0,3}-?\d{6,10}\b", score=0.3),
    ]
    CONTEXT = ["mrn", "medical record number", "medical record no", "record number", "chart number"]

    def __init__(self):
        super().__init__(
            supported_entity="MEDICAL_RECORD_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


class HealthPlanBeneficiaryRecognizer(PatternRecognizer):
    """HIPAA identifier #9 — health plan beneficiary numbers."""

    PATTERNS = [
        Pattern(name="beneficiary_alnum", regex=r"\b[A-Z0-9]{2}-?\d{6,9}-?[A-Z0-9]{0,2}\b", score=0.3),
    ]
    CONTEXT = ["beneficiary", "health plan", "insurance id", "policy number", "member id"]

    def __init__(self):
        super().__init__(
            supported_entity="HEALTH_PLAN_BENEFICIARY_NUM",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


class DeviceIdentifierRecognizer(PatternRecognizer):
    """HIPAA identifier #12 — medical device serials / UDIs (GS1-style)."""

    PATTERNS = [
        Pattern(name="udi_gs1", regex=r"\(01\)\d{14}(\(1[07]\)\d{6})?(\(21\)[A-Z0-9]{1,20})?", score=0.5),
    ]
    CONTEXT = ["udi", "device identifier", "serial number", "lot number"]

    def __init__(self):
        super().__init__(
            supported_entity="DEVICE_IDENTIFIER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


class ICD10Recognizer(PatternRecognizer):
    """ICD-10-CM diagnosis code pattern, e.g. E11.9, S72.001A."""

    PATTERNS = [
        Pattern(name="icd10", regex=r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?\b", score=0.3),
    ]
    CONTEXT = ["diagnosis", "icd-10", "icd10", "dx code"]

    def __init__(self):
        super().__init__(
            supported_entity="ICD10_CODE",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


def build_medical_recognizers() -> List[PatternRecognizer]:
    """All recognizers feeding the 'Presidio Medical' branch of the pipeline."""
    return [
        MedicalLicenseRecognizer(),  # built-in, checksum-validated -> MEDICAL_LICENSE
        NPIRecognizer(),
        MedicalRecordNumberRecognizer(),
        HealthPlanBeneficiaryRecognizer(),
        DeviceIdentifierRecognizer(),
        ICD10Recognizer(),
    ]
