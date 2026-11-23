"""Small, explicitly labelled Telugu values for the Presidio rules layer."""

from __future__ import annotations

import re
import unicodedata

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult


_PIN_DIGIT = r"[0-9\u0c66-\u0c6f]"
_PIN_VALUE = re.compile(rf"({_PIN_DIGIT}{{6}})$")


class LabelledIndianPinRecognizer(PatternRecognizer):
    """Recognize a six-digit PIN only after an explicit postal field label."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="postal_code",
            name="LabelledIndianPinRecognizer",
            supported_language="en",
            patterns=[Pattern(
                name="labelled_indian_pin",
                regex=(rf"(?<!\w)(?:పిన్\s*(?:కోడ్|నంబరు|నెంబర్)?|"
                       rf"పోస్టల్\s*కోడ్|pin\s*(?:code|number)|postal\s*code)"
                       rf"\s*[:#-]?\s*{_PIN_DIGIT}{{6}}(?!\w)"),
                score=0.85,
            )],
        )

    @staticmethod
    def valid_pin(value: str) -> bool:
        """Validate six decimal digits and India's nonzero leading digit."""
        if len(value) != 6:
            return False
        try:
            digits = [unicodedata.digit(char) for char in value]
        except (TypeError, ValueError):
            return False
        return digits[0] != 0

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and "postal_code" not in entities:
            return []
        results = []
        for found in super().analyze(text, entities, nlp_artifacts):
            match = _PIN_VALUE.search(text[found.start:found.end])
            if not match or not self.valid_pin(match.group(1)):
                continue
            start = found.start + match.start(1)
            end = found.start + match.end(1)
            results.append(RecognizerResult(
                entity_type="postal_code", start=start, end=end, score=found.score,
                recognition_metadata={
                    "recognizer_name": self.name,
                    "recognizer_identifier": self.id,
                    "boundary_evidence": "explicit_address",
                },
            ))
        return results
