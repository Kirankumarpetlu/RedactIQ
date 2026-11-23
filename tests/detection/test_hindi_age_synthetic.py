"""Synthetic Hindi age checks; no repository data or fixtures are used."""
from pii_pipeline.hindi_utils import HINDI_NUMBER_WORDS, iter_hindi_number_spans, parse_hindi_number
from pii_pipeline.recognizers.hindi_age import iter_hindi_age_spans


def _surfaces(text):
    return [text[start:end] for start, end, _ in iter_hindi_age_spans(text)]


def test_every_cardinal_and_original_offsets():
    assert len(HINDI_NUMBER_WORDS) == 120
    for number, word in HINDI_NUMBER_WORDS.items():
        assert parse_hindi_number(word) == number
        assert parse_hindi_number(str(number)) == number
        devanagari = str(number).translate(str.maketrans("0123456789", "०१२३४५६७८९"))
        assert parse_hindi_number(devanagari) == number
        text = f"सूत्र: {word}।"
        assert [(text[start:end], value) for start, end, value in iter_hindi_number_spans(text)] == [(word, number)]


def test_personal_age_and_non_age_contexts():
    assert _surfaces("रोगी बयालीस साल की है।") == ["बयालीस साल की"]
    assert _surfaces("मैं ५८ का हूँ।") == ["५८"]
    assert _surfaces("मेरी आयु पैंसठ।") == ["पैंसठ"]
    assert _surfaces("बच्ची ८ महीने की है।") == ["८ महीने की"]
    for text in ("मैं 30 साल से लिख रहा हूँ।", "मैं 60 मिलीग्राम दवा लेता हूँ।",
                 "नब्ज ७२ है।", "कमरा ४५ खाली है।", "मैं 45 मिनट रुका।"):
        assert _surfaces(text) == []
