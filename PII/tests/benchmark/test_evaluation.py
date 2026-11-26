"""Reviewed-reference evaluation and separate unified benchmark groups."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import hashlib
from pii_pipeline import json_codec as json
import tempfile
import unittest

from scripts.evaluate_pipeline import PipelineEvaluator
from scripts.evaluate_unified_benchmark import evaluate


def test_latency_benchmark_repeats_warms_up_and_reports_stages(tmp_path, monkeypatch):
    from scripts import benchmark_configured_pipeline as benchmark
    from pii_pipeline import data_assets, model_cache, pipeline
    import sys

    source = tmp_path / "samples.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["context"])
        writer.writeheader()
        writer.writerows([{"context": "Synthetic first."}, {"context": "Synthetic second."}])
    calls = []

    class FakePipeline:
        device = "cpu"
        inference_backends = {"primary": "fake"}

        def __init__(self, **kwargs):
            assert kwargs["config"].runtime.cpu_threads == 3

        def run(self, text, enable_profiling=False):
            calls.append((text, enable_profiling))
            result = {"entities_detected": [], "redacted_text": text}
            if enable_profiling:
                result["profiling"] = {"stages": [{"stage_name": "detector", "duration_ms": 10.0}]}
            return result

    monkeypatch.setattr(pipeline, "Stage9Pipeline", FakePipeline)
    monkeypatch.setattr(data_assets, "ensure_data", lambda groups: {})
    monkeypatch.setattr(model_cache, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(model_cache, "get_hardware_info", lambda *args: {"device": "cpu"})
    destination = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--input", str(source), "--repeat", "2",
                                     "--cpu-threads", "3", "--profile", "--output-dir", str(destination)])
    assert benchmark.main() == 0
    report = json.loads((destination / "benchmark_summary.json").read_text())
    assert report["rows"] == 4 and report["unique_input_rows"] == 2
    assert report["repeat"] == 2 and report["warmup_rows"] == 2
    assert report["cpu_threads"] == 3
    assert report["stage_timings"]["detector"] == {"calls": 4, "mean_ms": 10.0, "total_ms": 40.0}
    assert len(report["latencies_ms"]) == 4
    assert [enabled for _, enabled in calls] == [False, False, True, True, True, True]
    assert list(destination.iterdir()) == [destination / "benchmark_summary.json"]


class PipelineEvaluatorTests(unittest.TestCase):
    def write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def evaluate(
        self,
        ground_truth_rows: list[dict[str, str]],
        prediction_rows: list[dict[str, str]],
        scope: str = "strict",
        row_scope: str = "all",
        min_prediction_score: float = 0.0,
    ) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth_path = root / "ground_truth.csv"
            prediction_path = root / "predictions.csv"
            self.write_csv(
                ground_truth_path,
                ["row_index", "context", "verified_semantic_tags"],
                ground_truth_rows,
            )
            self.write_csv(
                prediction_path,
                ["row_index", "context", "pipeline_entities"],
                prediction_rows,
            )
            return PipelineEvaluator(
                scope=scope,
                row_scope=row_scope,
                min_prediction_score=min_prediction_score,
            ).evaluate(str(ground_truth_path), str(prediction_path))

    def test_strict_and_covered_label_scopes_are_distinct(self) -> None:
        source = "John visits Corp"
        ground_truth = [{
            "row_index": "17",
            "context": source,
            "verified_semantic_tags": json.dumps([
                {"entity_type": "PERSON", "text": "John", "start": 0, "end": 4},
            ]),
        }]
        predictions = [{
            "row_index": "17",
            "context": source,
            "pipeline_entities": json.dumps([
                {"entity_type": "full_name", "start": 0, "end": 4, "score": 0.9},
                {"entity_type": "organization", "start": 12, "end": 16, "score": 0.8},
            ]),
        }]

        strict = self.evaluate(ground_truth, predictions, scope="strict")["summary"]
        covered = self.evaluate(ground_truth, predictions, scope="covered-labels")["summary"]

        self.assertEqual(strict["overall"], {"tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0, "f1": 0.6667})
        self.assertEqual(covered["overall"], {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(covered["unscored_predictions"], {"ORGANIZATION": 1})

    def test_ambiguous_invalid_ground_truth_offset_is_not_silently_repaired(self) -> None:
        result = self.evaluate(
            [{
                "row_index": "4",
                "context": "Sam met Sam",
                "verified_semantic_tags": json.dumps([
                    {"entity_type": "PERSON", "text": "Sam", "start": 1, "end": 4},
                ]),
            }],
            [{"row_index": "4", "context": "Sam met Sam", "pipeline_entities": "[]"}],
        )

        self.assertEqual(result["summary"]["overall"], {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(result["summary"]["ground_truth_issues"], {"ambiguous_text_offset": 1})

    def test_union_of_row_identifiers_accounts_for_missing_output_and_reference_rows(self) -> None:
        result = self.evaluate(
            [{
                "row_index": "ground-truth-only",
                "context": "Alice",
                "verified_semantic_tags": json.dumps([
                    {"entity_type": "PERSON", "text": "Alice", "start": 0, "end": 5},
                ]),
            }],
            [{
                "row_index": "prediction-only",
                "context": "Bob",
                "pipeline_entities": json.dumps([
                    {"entity_type": "PERSON", "start": 0, "end": 3, "score": 0.9},
                ]),
            }],
        )

        self.assertEqual(result["summary"]["overall"], {"tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(result["summary"]["rows_scored"], 2)

    def test_prediction_row_scope_excludes_unprocessed_reference_rows(self) -> None:
        result = self.evaluate(
            [{
                "row_index": "ground-truth-only",
                "context": "Alice",
                "verified_semantic_tags": json.dumps([
                    {"entity_type": "PERSON", "text": "Alice", "start": 0, "end": 5},
                ]),
            }],
            [{
                "row_index": "prediction-only",
                "context": "Bob",
                "pipeline_entities": json.dumps([
                    {"entity_type": "PERSON", "start": 0, "end": 3, "score": 0.9},
                ]),
            }],
            row_scope="predictions",
        )

        self.assertEqual(result["summary"]["overall"], {"tp": 0, "fp": 1, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(result["summary"]["rows_scored"], 1)

    def test_source_mismatch_is_skipped_by_default(self) -> None:
        result = self.evaluate(
            [{
                "row_index": "8",
                "context": "Alice",
                "verified_semantic_tags": json.dumps([
                    {"entity_type": "PERSON", "text": "Alice", "start": 0, "end": 5},
                ]),
            }],
            [{
                "row_index": "8",
                "context": "Bob",
                "pipeline_entities": json.dumps([
                    {"entity_type": "PERSON", "start": 0, "end": 3, "score": 0.9},
                ]),
            }],
        )

        self.assertEqual(result["summary"]["rows_scored"], 0)
        self.assertEqual(result["source_mismatch_row_indices"], ["8"])

    def test_minimum_score_filters_predictions_before_matching(self) -> None:
        source = "John visits Corp"
        result = self.evaluate(
            [{
                "row_index": "10",
                "context": source,
                "verified_semantic_tags": json.dumps([
                    {"entity_type": "PERSON", "text": "John", "start": 0, "end": 4},
                ]),
            }],
            [{
                "row_index": "10",
                "context": source,
                "pipeline_entities": json.dumps([
                    {"entity_type": "full_name", "start": 0, "end": 4, "score": 0.9},
                    {"entity_type": "organization", "start": 12, "end": 16, "score": 0.8},
                ]),
            }],
            min_prediction_score=0.9,
        )

        self.assertEqual(result["summary"]["overall"], {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0})
        self.assertEqual(result["summary"]["prediction_issues"], {"below_min_score": 1})


class UnifiedBenchmarkEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.benchmark = self.root / "benchmark.csv"
        self.predictions = self.root / "predictions.csv"
        self.output = self.root / "metrics.json"
        tag = {"entity_type": "PERSON", "text": "Sam", "start": 0, "end": 3}
        self.references = []
        self.saved_predictions = []
        for row_id, group, status, scope in (
            ("synthetic_1", "synthetic_regression", "synthetic_reference", "synthetic_exhaustive"),
            ("real_1", "real_development_pending", "pending", "incomplete"),
        ):
            self.references.append({
                "row_index": row_id, "context": "Sam called.",
                "source_fingerprint": hashlib.sha256(b"Sam called.").hexdigest(),
                "benchmark_group": group, "review_status": status, "annotation_scope": scope,
                "chosen_reference_tags": json.dumps([tag]),
                "verified_semantic_tags": json.dumps([tag]) if group == "synthetic_regression" else "",
            })
            self.saved_predictions.append({"row_index": row_id, "context": "Sam called.",
                                           "pipeline_entities": json.dumps([tag]), "total_entities_count": "1"})

    def write(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def score(self):
        self.write(self.benchmark, self.references)
        self.write(self.predictions, self.saved_predictions)
        return evaluate(self.benchmark, self.predictions, self.output)

    def test_groups_stay_separate_and_inputs_stay_unchanged(self):
        self.saved_predictions[1]["pipeline_entities"] = "[]"
        self.saved_predictions[1]["total_entities_count"] = "0"
        result = self.score()
        synthetic = result["groups"]["synthetic_regression"]
        pending = result["groups"]["real_development_pending"]
        self.assertNotIn("overall", result)
        self.assertEqual(synthetic["typed_span_metrics"]["f1"], 1.0)
        self.assertEqual(pending["typed_span_metrics"]["fn"], 1)
        self.assertEqual(pending["comparison_status"], "provisional_pending_reference_agreement")
        self.assertEqual(pending["annotation_scope"], "incomplete")
        self.assertFalse(pending["verified_human_real_reference"])
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), result)
        with self.benchmark.open(encoding="utf-8-sig", newline="") as handle:
            pending_source = list(csv.DictReader(handle))[1]
        self.assertEqual(pending_source["verified_semantic_tags"], "")
        self.assertEqual(pending_source["review_status"], "pending")
        self.assertEqual(result["source_files"][str(self.benchmark.resolve())],
                         hashlib.sha256(self.benchmark.read_bytes()).hexdigest())

    def test_exact_coverage_ignores_type_but_typed_metrics_do_not(self):
        wrong_type = {"entity_type": "ORGANIZATION", "text": "Sam", "start": 0, "end": 3}
        self.saved_predictions[1]["pipeline_entities"] = json.dumps([wrong_type])
        pending = self.score()["groups"]["real_development_pending"]
        self.assertEqual(pending["typed_span_metrics"]["tp"], 0)
        self.assertEqual(pending["typed_span_metrics"]["fp"], 1)
        self.assertEqual(pending["typed_span_metrics"]["fn"], 1)
        self.assertEqual(pending["exact_character_coverage"]["coverage_fraction"], 1.0)
        self.assertEqual(pending["exact_character_coverage"]["fully_hidden_reference_spans"], 1)

    def test_explicit_negative_is_valid(self):
        self.references[1]["chosen_reference_tags"] = "[]"
        self.saved_predictions[1]["pipeline_entities"] = "[]"
        self.saved_predictions[1]["total_entities_count"] = "0"
        coverage = self.score()["groups"]["real_development_pending"]["exact_character_coverage"]
        self.assertIsNone(coverage["coverage_fraction"])
        self.assertEqual(coverage["reference_characters"], 0)

    def test_selected_group_accepts_partial_or_complete_known_predictions(self):
        self.write(self.benchmark, self.references)
        for records in (self.saved_predictions[:1], self.saved_predictions):
            with self.subTest(count=len(records)):
                self.write(self.predictions, records)
                result = evaluate(self.benchmark, self.predictions, selected_groups=["synthetic_regression"])
                self.assertEqual(set(result["groups"]), {"synthetic_regression"})
                self.assertEqual(result["groups"]["synthetic_regression"]["rows_scored"], 1)

    def test_selected_group_rejects_unknown_or_missing_ids(self):
        self.write(self.benchmark, self.references)
        for records in (self.saved_predictions[1:],
                        self.saved_predictions + [{**self.saved_predictions[0], "row_index": "unknown"}]):
            with self.subTest(ids=[r["row_index"] for r in records]):
                self.write(self.predictions, records)
                with self.assertRaises(ValueError):
                    evaluate(self.benchmark, self.predictions, selected_groups=["synthetic_regression"])

    def test_blank_or_invalid_references_fail_without_output(self):
        for raw in ("", "not json", "null", "{}"):
            with self.subTest(raw=raw):
                self.references[1]["chosen_reference_tags"] = raw
                with self.assertRaises(ValueError):
                    self.score()
                self.assertFalse(self.output.exists())

    def test_missing_duplicate_or_mismatched_prediction_ids_fail(self):
        original = deepcopy(self.saved_predictions)
        for records in (original[:1], original + [original[0]],
                        [original[0], {**original[1], "row_index": "other"}]):
            with self.subTest(ids=[r["row_index"] for r in records]):
                self.saved_predictions = records
                with self.assertRaises(ValueError):
                    self.score()

    def test_duplicate_reference_ids_fail(self):
        self.references.append(deepcopy(self.references[0]))
        with self.assertRaisesRegex(ValueError, "duplicate row_index"):
            self.score()

    def test_changed_context_or_fingerprint_fails(self):
        for field in ("context", "source_fingerprint"):
            with self.subTest(field=field):
                old = self.references[1][field]
                self.references[1][field] = "changed"
                with self.assertRaises(ValueError):
                    self.score()
                self.references[1][field] = old

    def test_invalid_offsets_text_and_duplicate_spans_fail(self):
        tag = {"entity_type": "PERSON", "text": "Sam", "start": 0, "end": 3}
        cases = ([{**tag, "start": True}], [{**tag, "end": 99}],
                 [{**tag, "text": "wrong"}], [tag, {**tag, "entity_type": "full_name"}])
        for column in ("chosen_reference_tags", "pipeline_entities"):
            records = self.references if column == "chosen_reference_tags" else self.saved_predictions
            for spans in cases:
                with self.subTest(column=column, spans=spans):
                    records[1][column] = json.dumps(spans)
                    with self.assertRaises(ValueError):
                        self.score()
            records[1][column] = json.dumps([tag])

    def test_prediction_count_is_required_and_must_match(self):
        for count in ("", "two", "0", "2", "-1"):
            with self.subTest(count=count):
                self.saved_predictions[1]["total_entities_count"] = count
                with self.assertRaises(ValueError):
                    self.score()

    def test_pending_cannot_be_promoted_to_verified(self):
        self.references[1]["verified_semantic_tags"] = self.references[1]["chosen_reference_tags"]
        with self.assertRaisesRegex(ValueError, "must not be marked verified"):
            self.score()

    def test_review_group_metadata_is_enforced(self):
        self.references[1]["review_status"] = "reviewed"
        with self.assertRaisesRegex(ValueError, "unexpected review status"):
            self.score()

    def test_output_cannot_overwrite_input_or_existing_artifact(self):
        self.write(self.benchmark, self.references)
        self.write(self.predictions, self.saved_predictions)
        original = self.benchmark.read_bytes()
        with self.assertRaises(ValueError):
            evaluate(self.benchmark, self.predictions, self.benchmark)
        self.assertEqual(self.benchmark.read_bytes(), original)
        self.output.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            evaluate(self.benchmark, self.predictions, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing")
