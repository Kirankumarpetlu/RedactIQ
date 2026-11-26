"""Small, model-free checks for the IndicNER Telugu boundary layer."""

import pytest

from pii_pipeline.enrichment.telugu_enrichment import (
    merge_fragments,
    normalize_with_offsets,
    repair_boundaries,
)
from pii_pipeline.telugu_utils import is_akshara_boundary


@pytest.mark.parametrize("text", [
    "హైదరాబాద్‌లో ఉంటాను",
    "హైదరాబాద్\u200cలో ఉంటాను",
    "హైదరాబాద్\u200dలో ఉంటాను",
    "హైదరాబాద్ లో ఉంటాను",
    "నా పేరు శ్రీ రవి గారు",
    "నా పేరు రవి\u0301",
])
def test_normalization_maps_every_nonempty_subspan_to_original_slice(text):
    normalized = normalize_with_offsets(text)
    assert normalized.text == normalize_with_offsets(text).text
    for start in range(len(normalized.text)):
        for end in range(start + 1, len(normalized.text) + 1):
            original_start, original_end = normalized.original_span(start, end)
            assert 0 <= original_start < original_end <= len(text)
            assert text[original_start:original_end]


def test_joiner_mapping_keeps_original_slice():
    text = "హైదరాబాద్\u200cలో"
    normalized = normalize_with_offsets(text)
    assert "\u200c" not in normalized.text
    assert text[slice(*normalized.original_span(0, len(normalized.text)))] == text


@pytest.mark.parametrize("text,surface", [
    ("నా పేరు రవి", "రవి"),
    ("నేను హైదరాబాద్‌లో ఉంటాను", "హైదరాబాద్‌లో"),
    ("శ్రీమతి లక్ష్మి గారు", "లక్ష్మి"),
])
def test_repair_is_idempotent_and_akshara_safe(text, surface):
    start = text.index(surface)
    entity = {"start": start + 1, "end": start + len(surface) - 1,
              "entity_type": "PERSON", "score": 0.95}
    repaired = repair_boundaries(text, entity)
    assert repaired == repair_boundaries(text, repaired)
    assert text[repaired["start"]:repaired["end"]] == repaired["text"]
    assert is_akshara_boundary(text, repaired["start"])
    assert is_akshara_boundary(text, repaired["end"])


def test_fragment_merge_requires_initial_and_keeps_original_offsets():
    text = "ఆర్. రవి మరియు లక్ష్మి"
    initial_end = text.index(" ")
    first = {"start": 0, "end": initial_end, "text": text[:initial_end],
             "entity_type": "PERSON", "score": 0.9}
    second_start = text.index("రవి")
    second = {"start": second_start, "end": second_start + len("రవి"),
              "text": "రవి", "entity_type": "PERSON", "score": 0.8}
    merged = merge_fragments(text, [first, second])
    assert len(merged) == 1
    assert merged[0]["text"] == text[merged[0]["start"]:merged[0]["end"]]
    assert merged == merge_fragments(text, merged)
