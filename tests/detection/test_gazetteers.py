"""The closed name list that covers what neither model reaches.

GLiNER2 returns nothing for Indic-script names at any threshold, and IndicNER
misses names that double as ordinary words or sit at a code-switch boundary.
ಶೋಭನ is the case that motivated this: detected in a plain Kannada sentence,
missed in the Kannada-into-Telugu switch where it actually appears, and left in
the clear by every run until the gazetteer existed."""
from __future__ import annotations

import pytest

from pii_pipeline.recognizers.indic_identifiers import IndicPersonNameGazetteer, load_person_name_gazetteer


def test_place_gazetteer_is_verified_once_and_shared_by_recognizers(monkeypatch):
    from pii_pipeline.recognizers import indic_identifiers as module
    from unittest.mock import Mock
    verify = Mock(wraps=module.require_data_path)
    module._load_place_gazetteer.cache_clear()
    monkeypatch.setattr(module, "require_data_path", verify)
    try:
        first, second = module.SpokenPlaceRecognizer(), module.SpokenPlaceRecognizer()
        assert first.place_pattern is second.place_pattern
        assert first.places is second.places
        assert verify.call_count == 1
        for rec in (first, second):
            text = "I went to Hyderabad."
            assert "Hyderabad" in [text[r.start:r.end] for r in rec.analyze(text, ["address"])]
        assert verify.call_count == 1
    finally:
        module._load_place_gazetteer.cache_clear()


@pytest.mark.parametrize("place,expected", [("Hyderabad", ["Hyderabad"]), ("music", [])])
def test_asr_place_lookup_uses_the_cached_name_set(place, expected):
    from pii_pipeline.recognizers.indic_identifiers import SpokenPlaceRecognizer
    text = f"My interest is {place}."
    assert [text[r.start:r.end] for r in SpokenPlaceRecognizer().analyze(text, ["address"])] == expected


@pytest.fixture(scope="module")
def recognizer():
    return IndicPersonNameGazetteer()


def _spans(recognizer, text):
    return [text[r.start:r.end] for r in recognizer.analyze(text, ["PERSON"], None)]


def test_the_gazetteer_file_loads():
    assert len(load_person_name_gazetteer()) > 100


@pytest.mark.parametrize("text,expected", [
    ("ಹೇಳೋದು ಶೋಭನ ಗುರಿಂಚಿ", "ಶೋಭನ"),
    ("ఆ, విజయలక్ష్మి.", "విజయలక్ష్మి"),
    ("ಶ್ರೀವಲ್ಲಿ", "ಶ್ರೀವಲ್ಲಿ"),
    ("डॉक्टर: आरव परीक्षा", "आरव"),
])
def test_indic_names_are_found(recognizer, text, expected):
    assert expected in _spans(recognizer, text)


def test_the_code_mixed_sentence_that_defeated_both_models(recognizer):
    # Kannada name inside a Kannada-to-Telugu switch, as it appears in row 2.
    text = "ಹೇಳೋದು ಶೋಭನ గురించి సెట్ చేపించుకున్నామండి."
    assert "ಶೋಭನ" in _spans(recognizer, text)


def test_a_sentence_final_stop_is_not_swallowed(recognizer):
    # "Shobhana." must not be looked up with its trailing period attached.
    assert _spans(recognizer, "My name is Shobhana.") == ["Shobhana"]


def test_given_name_and_surname_report_as_one_span(recognizer):
    assert _spans(recognizer, "శారదా శర్మ గారు") == ["శారదా శర్మ"]
    assert _spans(recognizer, "రాహుల్ వర్మ") == ["రాహుల్ వర్మ"]


@pytest.mark.parametrize("text", [
    "ఒకసారి టెస్ట్ చేయించుకున్నాం",     # "once test we got done"
    "bowel and bladder normal, CBC done",
    "సో రథర్ షి గివెన్ ఓన్లీ",           # so / rather / she / given / only
])
def test_ordinary_speech_is_not_matched(recognizer, text):
    assert _spans(recognizer, text) == []


def test_it_outranks_spacy_so_presidio_keeps_the_precise_span(recognizer):
    # spaCy emits its free-text guesses at a flat 0.85 and Presidio resolves
    # overlaps by score, so anything at or below that gets swallowed.
    results = recognizer.analyze("ఆ, విజయలక్ష్మి.", ["PERSON"], None)
    assert results and all(r.score > 0.85 for r in results)


def test_other_entity_types_are_not_claimed(recognizer):
    assert recognizer.analyze("ఆ, విజయలక్ష్మి.", ["PHONE_NUMBER"], None) == []
