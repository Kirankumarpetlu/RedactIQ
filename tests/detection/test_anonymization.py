"""Entity ordering, provenance and indexed redaction placeholders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
import unittest

from presidio_analyzer import RecognizerResult

from pii_pipeline.anonymizer import PlaceholderAnonymizer
import pytest


@pytest.mark.parametrize("indexed", [True, False])
def test_dense_unicode_redaction_preserves_offsets_and_placeholder_order(indexed):
    words = ["పేరు", "Ada", "नाम", "Jo"] * 100
    text = " ".join(words)
    offset, entities = 0, []
    for word in words:
        entities.append({"start": offset, "end": offset + len(word), "entity_type": "person"})
        offset += len(word) + 1
    expected = " ".join(f"[NAME_{i}]" if indexed else "[NAME]" for i in range(1, len(words) + 1))
    assert PlaceholderAnonymizer("[", "]", indexed).redact_selected(text, list(reversed(entities))) == expected


def test_linear_redaction_keeps_empty_inputs_and_rejects_overlapping_spans():
    anonymizer = PlaceholderAnonymizer()
    assert anonymizer.redact_selected("unchanged", []) == "unchanged"
    assert anonymizer.redact_selected("", []) == ""
    with pytest.raises(ValueError, match="non-overlapping"):
        anonymizer.redact_selected("abc", [{"start": 0, "end": 2}, {"start": 1, "end": 3}])


def test_anonymizer_honors_format_and_indexes_per_label():
    text = "Ada emailed ada@example.com and Ada called."
    results = [
        RecognizerResult(entity_type="person", start=0, end=3, score=1.0),
        RecognizerResult(entity_type="email", start=12, end=27, score=1.0),
        RecognizerResult(entity_type="person", start=32, end=35, score=1.0),
    ]

    assert PlaceholderAnonymizer().redact(text, results) == "<NAME_1> emailed <EMAIL_1> and <NAME_2> called."
    assert PlaceholderAnonymizer(prefix="[", suffix="]", indexed=False).redact(text, results) == "[NAME] emailed [EMAIL] and [NAME] called."


spec = importlib.util.spec_from_file_location(
    "entity_resolution_test_module",
    Path(__file__).parents[2] / "pii_pipeline" / "policy" / "entity_resolution.py",
)


entity_resolution = importlib.util.module_from_spec(spec)


sys.modules[spec.name] = entity_resolution


spec.loader.exec_module(entity_resolution)


@dataclass
class Candidate:
    entity_type: str
    start: int
    end: int
    score: float


class EntityResolutionTests(unittest.TestCase):
    def test_sort_keeps_entity_source_pairs_together(self) -> None:
        candidates = [
            (Candidate("PERSON", 12, 16, 0.7), "resolved"),
            (Candidate("PERSON", 0, 4, 0.8), "indic_ner"),
        ]

        resolved = entity_resolution.sort_and_deduplicate_entities(candidates)

        self.assertEqual([(item[0].start, item[1]) for item in resolved], [(0, "indic_ner"), (12, "resolved")])

    def test_exact_duplicate_uses_highest_score_source(self) -> None:
        candidates = [
            (Candidate("PERSON", 0, 4, 0.7), "resolved"),
            (Candidate("PERSON", 0, 4, 0.9), "indic_ner"),
            (Candidate("ORGANIZATION", 0, 4, 0.95), "fallback_routed"),
        ]

        resolved = entity_resolution.sort_and_deduplicate_entities(candidates)

        self.assertEqual(len(resolved), 2)
        self.assertEqual(resolved[1][1], "indic_ner")
        self.assertEqual(resolved[1][0].score, 0.9)
