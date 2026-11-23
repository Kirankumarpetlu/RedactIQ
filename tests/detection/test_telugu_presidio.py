from types import SimpleNamespace

from presidio_analyzer import RecognizerResult

from pii_pipeline.orchestrator import run_recognizers
from pii_pipeline.recognizers.telugu_presidio import LabelledIndianPinRecognizer


def test_labelled_pin_returns_only_value_in_both_digit_scripts():
    recognizer = LabelledIndianPinRecognizer()
    for text, value in (("పిన్ కోడ్ 502319", "502319"),
                        ("పోస్టల్ కోడ్: ౫౦౨౩౧౯", "౫౦౨౩౧౯"),
                        ("postal code 502319", "502319")):
        results = recognizer.analyze(text, ["postal_code"])
        assert [text[result.start:result.end] for result in results] == [value]


def test_pin_needs_postal_label_and_valid_number():
    recognizer = LabelledIndianPinRecognizer()
    for text in ("రూమ్ నెం. 502319", "బి.పి 120/80", "పిన్ కోడ్ 002319",
                 "పిన్ కోడ్ 5023197", "ఇంటి నెం. 8-2-120"):
        assert recognizer.analyze(text, ["postal_code"]) == []


def test_english_spacy_indic_proposal_is_removed_before_context_enhancement():
    class FakeRecognizer:
        name = "SpacyRecognizer"
        id = "spacy"
        is_loaded = True
        supported_entities = ["PERSON"]
        score_thresholds = {}

        def analyze(self, text, entities, artifacts):
            return [RecognizerResult("PERSON", 0, len(text), .85)]

    enhanced = []
    engine = SimpleNamespace(
        default_score_threshold=0,
        _enhance_using_context=lambda text, results, artifacts, selected, context: enhanced.extend(results) or results,
    )
    text = "రవి"
    assert run_recognizers(engine, text, "en", [FakeRecognizer()], artifacts=object()) == []
    assert enhanced == []


def test_spacy_latin_proposal_remains_available():
    class FakeRecognizer:
        name = "SpacyRecognizer"
        id = "spacy"
        is_loaded = True
        supported_entities = ["PERSON"]
        score_thresholds = {}

        def analyze(self, text, entities, artifacts):
            return [RecognizerResult("PERSON", 0, len(text), .85)]

    engine = SimpleNamespace(default_score_threshold=0,
                             _enhance_using_context=lambda text, results, artifacts, selected, context: results)
    assert len(run_recognizers(engine, "Ravi", "en", [FakeRecognizer()], artifacts=object())) == 1
    assert len(run_recognizers(engine, "Ravi రవి", "en", [FakeRecognizer()], artifacts=object())) == 1
