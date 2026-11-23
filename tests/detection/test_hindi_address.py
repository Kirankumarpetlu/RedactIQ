"""Address rule checks use only the committed synthetic Hindi generator."""

from scripts.synthetic_hindi.generate import generate
from pii_pipeline.recognizers.hindi_address import HindiAddressRecognizer


def test_generated_address_and_place_boundaries():
    recognizer = HindiAddressRecognizer()
    cases = generate()
    for case in cases:
        predicted = recognizer.analyze(case.text, ["address"])
        for gold in case.gold:
            if gold.type not in {"ADDRESS", "LOCATION"}:
                continue
            assert any(
                result.start == gold.start and result.end == gold.end
                for result in predicted
            ), case.id


def test_generated_negative_cases_have_no_address():
    recognizer = HindiAddressRecognizer()
    for case in generate():
        if case.split == "C":
            assert not recognizer.analyze(case.text, ["address"]), case.id
