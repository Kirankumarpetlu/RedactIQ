"""Unified dataset provenance and embedded synthetic seed integrity."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
from pii_pipeline import json_codec as json
import sys
import tempfile
import unittest

import pytest

from pii_pipeline.data_assets import ensure_data
from scripts.build_indic_pii_gold_dataset import build_rows, main
from scripts.build_unified_benchmark import ROOT, load_sources, merge_entries, restore_entries, source_annotations, validate_tags


def entry(path, kind, context, tags, number=1, **fields):
    row = {"row_index": str(number), "context": context,
           "verified_semantic_tags": json.dumps(tags), **fields}
    return {"path": path, "kind": kind, "record_number": number,
            "row": row, "original": row.copy(), "tags": tags, "issues": []}


def person(context, value):
    start = context.index(value)
    return {"entity_type": "PERSON", "text": value, "start": start, "end": start + len(value)}


def test_nbsp_variants_merge_reversibly_without_changing_offset_positions():
    primary = "Name: Mira Sen."
    legacy = "Name: Mira\u00a0Sen."
    rows = merge_entries([
        entry("legacy", "legacy", legacy, [person(legacy, "Mira\u00a0Sen")]),
        entry("draft", "real_draft", primary, [person(primary, "Mira Sen")]),
    ])
    assert len(rows) == 1
    row = rows[0]
    assert row["context"] == primary
    assert row["annotation_conflict"] == "false"
    assert row["verified_semantic_tags"] == ""
    for source in json.loads(row["source_records"]):
        restored = list(primary)
        for edit in source["context_edits"]:
            restored[edit["position"]] = edit["character"]
        assert "".join(restored) == (legacy if source["path"] == "legacy" else primary)
    for alternative in json.loads(row["annotation_variants"]):
        validate_tags(primary, alternative["tags"])


def test_annotation_conflicts_are_retained_without_union_or_false_signoff():
    context = "Name: Mira. Name: Tavi."
    first, second = person(context, "Mira"), person(context, "Tavi")
    row = merge_entries([
        entry("draft", "real_draft", context, [first]),
        entry("worksheet", "review_sheet", context, [first, second]),
    ])[0]
    assert json.loads(row["chosen_reference_tags"]) == [first]
    assert [len(source["tags"]) for source in json.loads(row["annotation_variants"])] == [1, 2]
    assert row["annotation_conflict"] == "true"
    assert row["review_status"] == "pending"
    assert row["annotation_scope"] == "incomplete"
    assert row["verified_semantic_tags"] == ""


def test_empty_pending_annotations_never_become_verified_negatives():
    row = merge_entries([entry("draft", "real_draft", "A consultation.", [])])[0]
    assert row["chosen_reference_tags"] == "[]"
    assert row["verified_semantic_tags"] == ""
    assert any("not confirmed" in issue for issue in json.loads(row["review_issues"]))


def test_stable_ids_do_not_reuse_conflicting_original_row_ids():
    sources = [entry("draft", "real_draft", "One consultation.", [], number=1),
               entry("draft", "real_draft", "Another consultation.", [], number=1)]
    forward = merge_entries(sources)
    reverse = merge_entries(list(reversed(sources)))
    assert len({row["row_index"] for row in forward}) == 2
    assert {row["context"]: row["row_index"] for row in forward} == {row["context"]: row["row_index"] for row in reverse}


def test_deduplication_does_not_collapse_other_whitespace_or_case():
    rows = merge_entries([entry("draft", "real_draft", text, [])
                          for text in ["Name: A.", "Name:  A.", "name: A."]])
    assert len(rows) == 3


def test_synthetic_reference_stays_separate_and_is_not_marked_human_reviewed():
    text = "Name: Mira."
    tag = person(text, "Mira")
    row = merge_entries([entry("gold", "synthetic_csv", text, [tag])])[0]
    assert row["benchmark_group"] == "synthetic_regression"
    assert json.loads(row["verified_semantic_tags"]) == [tag]
    assert row["review_status"] == "synthetic_reference"
    assert row["independent_test_set"] == "false"
    with pytest.raises(ValueError, match="Synthetic and real"):
        merge_entries([entry("gold", "synthetic_csv", text, [tag]),
                       entry("draft", "real_draft", text, [tag])])


def test_legacy_repeated_values_are_alternatives_not_adjudicated_truth():
    row = {"context": "Mira met Mira.", "pii_dictionary": json.dumps({"PERSON": ["Mira"]})}
    tags, issues = source_annotations("legacy", row)
    assert len(tags) == 2
    assert issues and "2 possible occurrences" in issues[0]


@pytest.mark.parametrize("tags", [None, {}, [{"entity_type": "PERSON", "text": "Mira", "start": 0, "end": 4}]])
def test_invalid_or_misaligned_labels_fail_closed(tags):
    with pytest.raises(ValueError):
        validate_tags("Name: Mira.", tags)


def test_duplicate_annotations_fail_closed():
    text = "Name: Mira."
    tag = person(text, "Mira")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_tags(text, [tag, tag])


def test_csv_is_not_an_inference_tag_input(tmp_path):
    from pii_pipeline.run_csv_through_stage9 import _load_input_records
    text = "Name: Mira."
    row = merge_entries([entry("gold", "synthetic_csv", text, [person(text, "Mira")])])[0]
    path = tmp_path / "unified.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    loaded, kind = _load_input_records(path, None)
    assert kind == "csv"
    assert loaded == [{"row_index": row["row_index"], "context": text, "semantic_tags": []}]


@pytest.mark.parametrize("group,expected_id", [("real_development_pending", "real_1"),
                                             ("synthetic_regression", "synthetic_1")])
def test_group_filter_keeps_ids_and_applies_before_row_limit(tmp_path, group, expected_id):
    from pii_pipeline.run_csv_through_stage9 import _load_input_records
    rows = [{"row_index": "synthetic_1", "context": "Synthetic one", "benchmark_group": "synthetic_regression"},
            {"row_index": "real_1", "context": "Real one", "benchmark_group": "real_development_pending"},
            {"row_index": "real_2", "context": "Real two", "benchmark_group": "real_development_pending"}]
    path = tmp_path / "mixed.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected, kind = _load_input_records(path, 1, benchmark_group=group)
    assert kind == "csv" and [row["row_index"] for row in selected] == [expected_id]
    assert selected[0]["semantic_tags"] == []


def test_real_group_contains_exactly_the_500_pinned_transcripts():
    from pii_pipeline.run_csv_through_stage9 import _load_input_records
    benchmark = ensure_data(check_only=True)["data/gold/unified_benchmark.csv"]
    selected, _ = _load_input_records(benchmark, None, benchmark_group="real_development_pending")
    with benchmark.open(encoding="utf-8-sig", newline="") as handle:
        expected = {row["row_index"] for row in csv.DictReader(handle)
                    if row["benchmark_group"] == "real_development_pending"}
    assert len(selected) == len(expected) == 500
    assert {row["row_index"] for row in selected} == expected
    assert all(row["semantic_tags"] == [] for row in selected)


def test_group_selection_rejects_missing_column_and_non_csv_inputs(tmp_path):
    from pii_pipeline.run_csv_through_stage9 import _load_input_records
    path = tmp_path / "unlabelled.csv"
    path.write_text("context\nSynthetic text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_group column"):
        _load_input_records(path, None, benchmark_group="real_development_pending")
    with pytest.raises(ValueError, match="CSV input"):
        _load_input_records(path.with_suffix(".json"), None, benchmark_group="real_development_pending")
    with pytest.raises(ValueError, match="Unknown benchmark group"):
        _load_input_records(path, None, benchmark_group="unknown")


def test_embedded_sources_rebuild_without_original_files():
    text = "Name: Mira."
    rows = merge_entries([entry("draft", "real_draft", text, [person(text, "Mira")])])
    inventory = [{"path": "draft", "kind": "real_draft", "records": 1}]
    assert merge_entries(restore_entries(rows, inventory)) == rows


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "context", "edit"])
def test_incomplete_or_corrupt_embedded_provenance_is_rejected(corruption):
    text = "Name: Mira."
    rows = merge_entries([entry("draft", "real_draft", text, [person(text, "Mira")])])
    inventory = [{"path": "draft", "kind": "real_draft", "records": 1}]
    records = json.loads(rows[0]["source_records"])
    if corruption == "missing":
        inventory[0]["records"] = 2
    elif corruption == "duplicate":
        records.append(deepcopy(records[0]))
    elif corruption == "context":
        records[0]["original_context_sha256"] = "0" * 64
    else:
        records[0]["context_edits"] = [{"position": len(text), "character": "X"}]
    rows[0]["source_records"] = json.dumps(records)
    with pytest.raises(ValueError):
        restore_entries(rows, inventory)


def synthetic_records():
    ensure_data(("benchmark",))
    entries, _ = load_sources(ROOT)
    return [entry["original"] for entry in entries if entry["kind"] == "synthetic_seed"]


class IndicPiiGoldDatasetTests(unittest.TestCase):
    def test_seed_has_valid_spans_and_language_coverage(self) -> None:
        rows = build_rows(synthetic_records())

        self.assertEqual(len(rows), 28)
        self.assertEqual(
            {row["language"] for row in rows},
            {"Hindi", "Hinglish", "Tamil", "Telugu", "Kannada", "Bengali", "Malayalam"},
        )
        for row in rows:
            for tag in json.loads(row["verified_semantic_tags"]):
                self.assertEqual(row["context"][tag["start"]:tag["end"]], tag["text"])

    def test_rejects_ambiguous_entity_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly once"):
            build_rows([{
                "id": "bad-001",
                "language": "Hindi",
                "script": "Devanagari",
                "context": "Demo Demo",
                "entities": [{"entity_type": "person", "text": "Demo"}],
            }])

    def test_cli_writes_evaluator_compatible_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gold.csv"
            seed = Path(directory) / "new-seed.json"
            seed.write_text(json.dumps({"records": synthetic_records()}), encoding="utf-8")
            previous_argv = sys.argv
            try:
                sys.argv = ["build_indic_pii_gold_dataset.py", "--seed", str(seed), "--output", str(output)]
                self.assertEqual(main(), 0)
            finally:
                sys.argv = previous_argv
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 28)
            self.assertIn("verified_semantic_tags", rows[0])
