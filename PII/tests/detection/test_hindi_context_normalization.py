"""Synthetic Hindi normalization and Presidio cue checks."""
from presidio_analyzer import RecognizerResult

from pii_pipeline.hindi_normalization import normalize_with_offsets
from pii_pipeline.recognizers.hindi_context import enhance_hindi_results


def test_normalized_offset_round_trip_preserves_original_clusters():
    original = "ज़िला १२-क\u200d्ष"
    normalized = normalize_with_offsets(original)
    assert normalized.text == "जिला 12-क्ष"
    for token in ("जि", "12", "क्ष"):
        start = normalized.text.index(token)
        begin, end = normalized.original_span(start, start + len(token))
        assert normalize_with_offsets(original[begin:end]).text == token
    assert normalized.original_span(normalized.text.index("क्ष"), len(normalized.text)) == (
        original.index("क\u200d्ष"), len(original)
    )


def test_hindi_context_only_lifts_hindi_recognizer_with_matching_cue():
    text = "मेरी उम्र ४५ साल है। नब्ज़ ४५ है।"
    age_start = text.index("४५")
    pulse_start = text.rindex("४५")
    age = RecognizerResult("AGE", age_start, age_start + 2, .5,
                           recognition_metadata={"recognizer_name": "HindiAgeRecognizer"})
    pulse = RecognizerResult("AGE", pulse_start, pulse_start + 2, .5,
                             recognition_metadata={"recognizer_name": "NumberRecognizer"})
    result = enhance_hindi_results(text, [age, pulse])
    assert result[0].score == .62
    assert result[0].recognition_metadata["hindi_context_cue"] is True
    assert result[1].score == .5


def test_non_hindi_is_strict_noop():
    original = "My age is 45"
    age = RecognizerResult("AGE", 10, 12, .5,
                           recognition_metadata={"recognizer_name": "HindiAgeRecognizer"})
    result = enhance_hindi_results(original, [age])
    assert result == [age]
    assert result[0].score == .5
    assert "hindi_context_cue" not in result[0].recognition_metadata
