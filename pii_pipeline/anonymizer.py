"""Replace detected spans with configured safe placeholder tokens."""
from __future__ import annotations

from typing import List

from presidio_analyzer import RecognizerResult
from .policy.entity_policy import entity_from_result, resolve_entities


class PlaceholderAnonymizer:
    def __init__(self, prefix: str = "<", suffix: str = ">", indexed: bool = True):
        self.prefix = prefix
        self.suffix = suffix
        self.indexed = indexed

    def redact(self, text: str, results: List[RecognizerResult]) -> str:
        selected, _ = resolve_entities(text, [entity_from_result(r, "direct") for r in results])
        return self.redact_selected(text, selected)

    def redact_selected(self, text: str, entities: list[dict]) -> str:
        """Apply the shared policy's accepted spans without a second selection."""
        # Standardized Label Mapping
        label_map = {
            "phone_number": "PHONE",
            "email": "EMAIL",
            "person": "NAME",
            "full_name": "NAME",
            "first_name": "NAME",
            "middle_name": "NAME",
            "last_name": "NAME",
            "organization": "ORG",
            "org": "ORG",
            "location": "ADDRESS",
            "address": "ADDRESS",
            "street_address": "ADDRESS",
            "city": "ADDRESS",
            "state_or_region": "ADDRESS",
            "postal_code": "ADDRESS",
            "country": "ADDRESS",
            "government_id": "ID",
            "national_id_number": "ID",
            "passport_number": "ID",
            "drivers_license_number": "ID",
            "license_number": "ID",
            "tax_id": "ID",
            "tax_number": "ID",
            "bank_account": "BANK",
            "account_number": "BANK",
            "routing_number": "BANK",
            "iban": "BANK",
            "payment_card": "CARD",
            "card_number": "CARD",
            "card_expiry": "CARD",
            "card_cvv": "CARD",
            "username": "USER",
            "ip_address": "IP",
            "account_id": "USER",
            "sensitive_account_id": "USER",
            "password": "SECRET",
            "secret": "SECRET",
            "api_key": "SECRET",
            "access_token": "SECRET",
            "recovery_code": "SECRET",
            "sensitive_date": "DATE",
            "document_date": "DATE",
            "expiration_date": "DATE",
            "transaction_date": "DATE",
        }

        accepted = sorted(entities, key=lambda e: (e["start"], e["end"]))
        previous_end = 0
        for entity in accepted:
            if not previous_end <= entity["start"] < entity["end"] <= len(text):
                raise ValueError("redact_selected requires valid non-overlapping spans")
            previous_end = entity["end"]

        counters = {}
        pieces = []
        cursor = 0
        for res in accepted:
            raw_label = res["entity_type"]
            # Preserve the public uppercase PERSON placeholder contract. Hindi
            # organizations use the new ORG mapping for either label case.
            label = "PERSON" if raw_label == "PERSON" else label_map.get(raw_label.lower(), raw_label).upper()
            counters[label] = counters.get(label, 0) + 1
            index = f"_{counters[label]}" if self.indexed else ""
            pieces.extend((text[cursor:res["start"]], f"{self.prefix}{label}{index}{self.suffix}"))
            cursor = res["end"]
        pieces.append(text[cursor:])
        return "".join(pieces)
