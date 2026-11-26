"""Synthetic Hindi identity statements and explicit identifier fields."""
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult

from pii_pipeline.policy.entity_policy import entity_from_result, resolve_entities
from pii_pipeline.recognizers.indic_identifiers import (
    IndicNameContextRecognizer, SpokenPlaceRecognizer, build_indic_identifier_recognizers,
)
from tests.helpers import fake_pipeline


@pytest.mark.parametrize("name", ["अनिरुद्ध", "नवीन कुमार", "मेघना सेन"])
@pytest.mark.parametrize("copula", ["हूँ", "हूं", "है"])
def test_hindi_self_introduction_with_adjacent_age_field(name, copula):
    text = f"मैं {name} {copula}। मेरी उम्र 36 साल है।"
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == [name]


@pytest.mark.parametrize("text", [
    "मैं ठीक हूँ। मेरी उम्र 36 साल है।",
    "मैं बीमार हूँ। मेरी उम्र 36 साल है।",
    "मैं परेशान हूं। मेरी उम्र 36 साल है।",
    "मैं डॉक्टर हूँ। मेरी उम्र 36 साल है।",
    "मैं नर्स हूँ। मेरी उम्र 36 साल है।",
    "मैं बहुत खुश हूँ। मेरी उम्र 36 साल है।",
    "मैं नवचंद्रपुर से हूँ। मेरी उम्र 36 साल है।",
    "मैं अस्पताल में हूँ। मेरी उम्र 36 साल है।",
    "मैं कुछ कहना चाहता हूँ।",
])
def test_predicates_roles_and_origins_are_not_self_introduction_names(text):
    assert IndicNameContextRecognizer().analyze(text, ["PERSON"]) == []


@pytest.mark.parametrize("place", ["नवचंद्रपुर", "नया वसंत नगर", "New Haven"])
@pytest.mark.parametrize("relation", ["से हूँ", "से हूं", "में रहता हूँ", "में रहती हूं"])
def test_unlisted_hindi_personal_origins_and_residences(place, relation):
    rec = SpokenPlaceRecognizer()
    rec.places, rec.place_pattern = frozenset(), None
    text = f"मैं {place} {relation}।"
    results = rec.analyze(text, ["address"])
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in results])
    assert [e["text"] for e in selected] == [place]


@pytest.mark.parametrize("text", [
    "मैं घर से हूँ।", "मैं यहाँ से हूँ।", "मैं अस्पताल में रहता हूँ।",
    "दवा नवचंद्रपुर से आती है।", "मैं दर्द से परेशान हूँ।",
])
def test_generic_hindi_places_and_clinical_statements_are_not_addresses(text):
    rec = SpokenPlaceRecognizer()
    rec.places, rec.place_pattern = frozenset(), None
    assert rec.analyze(text, ["address"]) == []


@pytest.mark.parametrize("field", ["मेडिकल नंबर", "मेडिकल नम्बर", "हेल्थ आईडी", "medical number"])
@pytest.mark.parametrize("value", ["843-207-56192", "HC-582047", "8430 2075 6192"])
def test_medical_fields_override_phone_guesses(field, value):
    text = f"{field} {value} है।"
    results = [r for rec in build_indic_identifier_recognizers() if rec.name == "LabelledHealthIdRecognizer"
               for r in rec.analyze(text, ["MEDICAL_ID"])]
    start = text.index(value)
    phone = RecognizerResult("phone_number", start, start+len(value), .999,
                             recognition_metadata={"recognizer_name": "GLiNER2PIIPrimary"})
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in [phone, *results]])
    assert [(e["entity_type"], e["text"]) for e in selected] == [("MEDICAL_ID", value)]


@pytest.mark.parametrize("field", ["पैन नंबर", "पैन नम्बर", "PAN number", "PAN:"])
@pytest.mark.parametrize("value,valid", [("AZTPQ4821L", True), ("aztpq4821l", True), ("qwert12345", False)])
def test_pan_field_protects_opaque_values_and_reports_format_status(field, value, valid):
    text = f"{field} {value}।"
    results = [r for rec in build_indic_identifier_recognizers() if rec.name == "LabelledPanRecognizer"
               for r in rec.analyze(text, ["tax_id"])]
    assert len(results) == 1
    assert results[0].recognition_metadata["format_valid"] is valid
    start = text.index(value)
    wrong = RecognizerResult("PERSON", start, start+len(value), .99,
                             recognition_metadata={"recognizer_name": "SpacyRecognizer"})
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in [wrong, *results]])
    assert [(e["entity_type"], e["text"]) for e in selected] == [("tax_id", value)]
    pipe = fake_pipeline()
    pipe.orchestrator.analyze = lambda text, **kw: SimpleNamespace(resolved=results, needs_fallback=[])
    assert pipe.run(text)["redacted_text"] == f"{field} <ID_1>।"


@pytest.mark.parametrize("text", [
    "PAN number unavailable", "पैन नंबर उपलब्ध नहीं है।", "PAN: unknown",
    "PAN number 45 mg", "pan fry for 12345 seconds", "qwert12345",
])
def test_pan_context_requires_a_bounded_identifier_value(text):
    for rec in build_indic_identifier_recognizers():
        if rec.name == "LabelledPanRecognizer":
            assert rec.analyze(text, ["tax_id"]) == []
