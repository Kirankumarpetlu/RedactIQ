"""Small, model-free checks for the shared Telugu morphology helpers."""
from pii_pipeline.telugu_utils import (
    is_akshara_boundary,
    parse_telugu_number,
    split_telugu_suffix,
    trim_telugu_honorifics,
)


def test_number_words_and_telugu_digits():
    assert parse_telugu_number("నలభై ఐదు") == 45
    assert parse_telugu_number("౪౫") == 45
    assert parse_telugu_number("నలభై ఏడు") == 47
    assert parse_telugu_number("నలభై వంద") is None


def test_suffixes_keep_original_codepoint_stem():
    assert split_telugu_suffix("రవికి") == ("రవి", "కి")
    assert split_telugu_suffix("రవితో") == ("రవి", "తో")
    assert split_telugu_suffix("గుంటూరుకు", "place") == ("గుంటూరు", "కు")
    # ZWNJ closes the preceding cluster, so the suffix is separable while the
    # joiner remains with the stem in the original code-point span.
    joined = "హైదరాబాద్\u200cలో"
    assert split_telugu_suffix(joined, "place") == ("హైదరాబాద్\u200c", "లో")


def test_akshara_and_honorific_boundaries():
    assert not is_akshara_boundary("రవి", 2)  # between వ and ి
    assert is_akshara_boundary("రవికి", len("రవి"))
    text = "శ్రీ రవి గారు"
    start, end = trim_telugu_honorifics(text, 0, len(text))
    assert text[start:end] == "రవి"
