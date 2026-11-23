#!/usr/bin/env python3
"""Evaluate Stage 9 entities against an annotated reference CSV.

The evaluator supports two explicit scopes:

* ``strict``: every unmatched prediction is a false positive. Use only with
  exhaustively annotated reference data.
* ``covered-labels``: predictions for labels never annotated anywhere in the
  reference are reported as unscored. Predictions for covered labels remain
  subject to normal false-positive accounting.

Neither scope mutates ground truth. Invalid or ambiguous annotations are
reported and excluded rather than silently repaired to an arbitrary span.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence, Set, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entity:
    """A typed character span in a source record."""

    entity_type: str
    start: int
    end: int
    text: str = ""
    score: float = 1.0

    def iou(self, other: "Entity") -> float:
        overlap = max(0, min(self.end, other.end) - max(self.start, other.start))
        union = (self.end - self.start) + (other.end - other.start) - overlap
        return overlap / union if union else 0.0


@dataclass
class Metrics:
    """Confusion-matrix counts and derived metrics."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


ENTITY_TYPE_MAP = {
    "PERSON": {"person", "full_name", "first_name", "last_name", "middle_name"},
    "MEDICAL_ID": {"medical_id"},
    "PHONE_NUMBER": {"phone_number"},
    "DATE_OF_BIRTH": {"date_of_birth"},
    "ADDRESS": {"address", "street_address"},
    "US_DRIVER_LICENSE": {"us_driver_license", "drivers_license_number"},
    "EMAIL": {"email", "email_address"},
    "ORGANIZATION": {"organization", "org"},
}


def normalize_entity_type(entity_type: str) -> str:
    """Return a canonical, case-insensitive label for scoring."""

    normalized = entity_type.strip().lower()
    for canonical, variants in ENTITY_TYPE_MAP.items():
        if normalized == canonical.lower() or normalized in variants:
            return canonical
    return entity_type.strip().upper()


class PipelineEvaluator:
    """Evaluate predictions with validated source alignment and span matching."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        scope: str = "strict",
        row_scope: str = "all",
        min_prediction_score: float = 0.0,
        require_source_match: bool = True,
        require_reviewed: bool = False,
    ):
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be greater than 0 and no greater than 1")
        if scope not in {"strict", "covered-labels"}:
            raise ValueError("scope must be 'strict' or 'covered-labels'")
        if row_scope not in {"all", "predictions"}:
            raise ValueError("row_scope must be 'all' or 'predictions'")
        if not 0 <= min_prediction_score <= 1:
            raise ValueError("min_prediction_score must be between 0 and 1")
        self.iou_threshold = iou_threshold
        self.scope = scope
        self.row_scope = row_scope
        self.min_prediction_score = min_prediction_score
        self.require_source_match = require_source_match
        self.require_reviewed = require_reviewed
        self._reviewed_reference_rows: Set[str] = set()
        self._ground_truth_sources: Dict[str, str] = {}
        self._prediction_sources: Dict[str, str] = {}
        self._ground_truth_issues: List[Dict[str, str]] = []
        self._prediction_issues: List[Dict[str, str]] = []

    @staticmethod
    def _row_id(row: Mapping[str, str], fallback_index: int) -> str:
        value = row.get("row_index")
        return value.strip() if value and value.strip() else str(fallback_index)

    @staticmethod
    def _all_occurrences(text: str, value: str) -> List[int]:
        offsets: List[int] = []
        start = 0
        while True:
            offset = text.find(value, start)
            if offset < 0:
                return offsets
            offsets.append(offset)
            start = offset + 1

    def _ground_truth_entity(
        self,
        row_id: str,
        source: str,
        raw_entity: Mapping[str, Any],
    ) -> Optional[Entity]:
        entity_text = str(raw_entity.get("text", ""))
        entity_type = str(raw_entity.get("entity_type", ""))
        if not entity_text or not entity_type:
            self._ground_truth_issues.append({"row_index": row_id, "reason": "missing_text_or_type"})
            return None

        try:
            start = int(raw_entity.get("start"))
            end = int(raw_entity.get("end"))
        except (TypeError, ValueError):
            start = end = -1

        if 0 <= start < end <= len(source) and source[start:end] == entity_text:
            return Entity(normalize_entity_type(entity_type), start, end, entity_text)

        occurrences = self._all_occurrences(source, entity_text)
        if len(occurrences) == 1:
            repaired_start = occurrences[0]
            self._ground_truth_issues.append({"row_index": row_id, "reason": "repaired_unique_offset"})
            return Entity(
                normalize_entity_type(entity_type),
                repaired_start,
                repaired_start + len(entity_text),
                entity_text,
            )

        reason = "ambiguous_text_offset" if occurrences else "text_not_in_source"
        self._ground_truth_issues.append({"row_index": row_id, "reason": reason})
        return None

    def load_ground_truth(self, gt_csv: str) -> Dict[str, List[Entity]]:
        """Load validated ground-truth entities keyed by stable row identifier."""

        manifest_path = Path(gt_csv).with_suffix(Path(gt_csv).suffix + ".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("mode") == "freeze" and manifest.get("dataset_sha256") != hashlib.sha256(Path(gt_csv).read_bytes()).hexdigest():
                raise ValueError("Locked reference dataset changed after review")

        entities_by_row: DefaultDict[str, List[Entity]] = defaultdict(list)
        self._ground_truth_sources = {}
        self._ground_truth_issues = []
        self._reviewed_reference_rows = set()
        with open(gt_csv, newline="", encoding="utf-8-sig") as handle:
            reference_rows=list(csv.DictReader(handle))
            from scripts.prepare_precision_dataset import has_human_review_signoff
            if self.require_reviewed:
                from scripts.prepare_precision_dataset import validate_reviewed
                validate_reviewed(reference_rows)
            for fallback_index, row in enumerate(reference_rows):
                row_id = self._row_id(row, fallback_index)
                if row_id in self._ground_truth_sources:
                    raise ValueError(f"Duplicate ground-truth row_index: {row_id}")
                if "review_status" in row and not has_human_review_signoff(row):
                    raise ValueError(f"Row {row_id} has not been exhaustively reviewed")
                if has_human_review_signoff(row):
                    self._reviewed_reference_rows.add(row_id)
                source = row.get("context", "")
                self._ground_truth_sources[row_id] = source
                raw_tags = row.get("verified_semantic_tags", "[]")
                try:
                    tags = json.loads(raw_tags)
                except json.JSONDecodeError:
                    self._ground_truth_issues.append({"row_index": row_id, "reason": "invalid_tag_json"})
                    continue
                if not isinstance(tags, list):
                    self._ground_truth_issues.append({"row_index": row_id, "reason": "tags_not_a_list"})
                    continue
                for raw_entity in tags:
                    if not isinstance(raw_entity, Mapping):
                        self._ground_truth_issues.append({"row_index": row_id, "reason": "tag_not_an_object"})
                        continue
                    entity = self._ground_truth_entity(row_id, source, raw_entity)
                    if entity is not None:
                        entities_by_row[row_id].append(entity)
        logger.info("Loaded ground truth for %d rows", len(self._ground_truth_sources))
        return dict(entities_by_row)

    def load_predictions(self, prediction_csv: str) -> Dict[str, List[Entity]]:
        """Load valid prediction spans keyed by stable row identifier."""

        entities_by_row: DefaultDict[str, List[Entity]] = defaultdict(list)
        self._prediction_sources = {}
        self._prediction_issues = []
        with open(prediction_csv, newline="", encoding="utf-8-sig") as handle:
            for fallback_index, row in enumerate(csv.DictReader(handle)):
                row_id = self._row_id(row, fallback_index)
                if row_id in self._prediction_sources:
                    raise ValueError(f"Duplicate prediction row_index: {row_id}")
                source = row.get("context", "")
                self._prediction_sources[row_id] = source
                try:
                    raw_entities = json.loads(row.get("pipeline_entities", "[]"))
                except json.JSONDecodeError:
                    self._prediction_issues.append({"row_index": row_id, "reason": "invalid_entity_json"})
                    continue
                if not isinstance(raw_entities, list):
                    self._prediction_issues.append({"row_index": row_id, "reason": "entities_not_a_list"})
                    continue
                for raw_entity in raw_entities:
                    try:
                        entity_type = normalize_entity_type(str(raw_entity["entity_type"]))
                        start = int(raw_entity["start"])
                        end = int(raw_entity["end"])
                        score = float(raw_entity.get("score", 1.0))
                    except (KeyError, TypeError, ValueError):
                        self._prediction_issues.append({"row_index": row_id, "reason": "invalid_entity"})
                        continue
                    if not 0 <= start < end <= len(source):
                        self._prediction_issues.append({"row_index": row_id, "reason": "span_out_of_bounds"})
                        continue
                    if score < self.min_prediction_score:
                        self._prediction_issues.append({"row_index": row_id, "reason": "below_min_score"})
                        continue
                    entities_by_row[row_id].append(
                        Entity(entity_type, start, end, source[start:end], score)
                    )
        logger.info("Loaded predictions for %d rows", len(self._prediction_sources))
        return dict(entities_by_row)

    def _match_entities(
        self,
        ground_truth: Sequence[Entity],
        predictions: Sequence[Entity],
    ) -> Dict[int, Tuple[int, float]]:
        """Return a maximum-cardinality, deterministic one-to-one span matching."""

        edges: Dict[int, List[Tuple[int, float]]] = {}
        for prediction_index, prediction in enumerate(predictions):
            candidates = [
                (ground_truth_index, prediction.iou(expected))
                for ground_truth_index, expected in enumerate(ground_truth)
                if prediction.entity_type == expected.entity_type
                and prediction.iou(expected) >= self.iou_threshold
            ]
            edges[prediction_index] = sorted(candidates, key=lambda item: (-item[1], item[0]))

        ground_truth_to_prediction: Dict[int, int] = {}

        def assign(prediction_index: int, visited: Set[int]) -> bool:
            for ground_truth_index, _ in edges[prediction_index]:
                if ground_truth_index in visited:
                    continue
                visited.add(ground_truth_index)
                current = ground_truth_to_prediction.get(ground_truth_index)
                if current is None or assign(current, visited):
                    ground_truth_to_prediction[ground_truth_index] = prediction_index
                    return True
            return False

        for prediction_index in sorted(edges, key=lambda index: (len(edges[index]), index)):
            assign(prediction_index, set())

        matches: Dict[int, Tuple[int, float]] = {}
        for ground_truth_index, prediction_index in ground_truth_to_prediction.items():
            iou = next(
                value for candidate, value in edges[prediction_index] if candidate == ground_truth_index
            )
            matches[prediction_index] = (ground_truth_index, iou)
        return matches

    @staticmethod
    def _fingerprint(source: str) -> str:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _source_matches(self, row_id: str) -> bool:
        ground_truth_source = self._ground_truth_sources.get(row_id)
        prediction_source = self._prediction_sources.get(row_id)
        if ground_truth_source is None or prediction_source is None:
            return True
        return self._fingerprint(ground_truth_source) == self._fingerprint(prediction_source)

    def evaluate(
        self,
        gt_csv: str,
        prediction_csv: str,
        output_json: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score the complete union of reference and prediction row identifiers."""

        ground_truth_by_row = self.load_ground_truth(gt_csv)
        predictions_by_row = self.load_predictions(prediction_csv)
        if self.require_reviewed:
            missing=set(self._prediction_sources)-set(self._ground_truth_sources)
            mismatched={r for r in self._prediction_sources if not self._source_matches(r)}
            if missing or mismatched:
                raise ValueError("Reviewed evaluation requires matching ground truth for every prediction row")
        covered_labels = {
            entity.entity_type
            for entities in ground_truth_by_row.values()
            for entity in entities
        }
        all_row_ids = set(self._ground_truth_sources) | set(self._prediction_sources)
        if self.row_scope == "predictions":
            all_row_ids &= set(self._prediction_sources)
        all_row_ids = sorted(all_row_ids)
        overall = Metrics()
        per_type: DefaultDict[str, Metrics] = defaultdict(Metrics)
        row_details: List[Dict[str, Any]] = []
        skipped_source_rows: List[str] = []
        unscored_predictions: Counter[str] = Counter()
        coverage_details=[]
        character_counts=Counter()

        for row_id in all_row_ids:
            if self.require_source_match and not self._source_matches(row_id):
                skipped_source_rows.append(row_id)
                continue

            expected = ground_truth_by_row.get(row_id, [])
            predictions = predictions_by_row.get(row_id, [])
            selected_characters={i for e in predictions for i in range(e.start,e.end)}
            reference_characters={i for e in expected for i in range(e.start,e.end)}
            character_counts.update(reference=len(reference_characters),selected=len(selected_characters),
                                    intersection=len(reference_characters&selected_characters),
                                    union=len(reference_characters|selected_characters))
            source=self._ground_truth_sources.get(row_id,"")
            for e in expected:
                uncovered=[i for i in range(e.start,e.end) if i not in selected_characters]
                coverage_details.append({"row_index":row_id,"entity_type":e.entity_type,"start":e.start,"end":e.end,
                    "fully_hidden":not uncovered,
                    "non_whitespace_fully_hidden":not any(not source[i].isspace() for i in uncovered),
                    "uncovered_characters":len(uncovered)})
            scored_indexes = [
                index
                for index, prediction in enumerate(predictions)
                if self.scope == "strict" or prediction.entity_type in covered_labels
            ]
            scored_predictions = [predictions[index] for index in scored_indexes]
            unscored_indexes = set(range(len(predictions))) - set(scored_indexes)
            for prediction_index in unscored_indexes:
                unscored_predictions[predictions[prediction_index].entity_type] += 1

            matches = self._match_entities(expected, scored_predictions)
            matched_ground_truth = {ground_truth_index for ground_truth_index, _ in matches.values()}
            row_result: Dict[str, Any] = {
                "row_index": row_id,
                "true_positives": [],
                "false_positives": [],
                "false_negatives": [],
                "unscored_predictions": [
                    {
                        "entity_type": predictions[prediction_index].entity_type,
                        "span": f"{predictions[prediction_index].start}-{predictions[prediction_index].end}",
                    }
                    for prediction_index in sorted(unscored_indexes)
                ],
            }

            for prediction_index, prediction in enumerate(scored_predictions):
                match = matches.get(prediction_index)
                metric = per_type[prediction.entity_type]
                if match is None:
                    overall.fp += 1
                    metric.fp += 1
                    row_result["false_positives"].append(
                        {
                            "entity_type": prediction.entity_type,
                            "span": f"{prediction.start}-{prediction.end}",
                            "score": prediction.score,
                        }
                    )
                    continue
                _, iou = match
                overall.tp += 1
                metric.tp += 1
                row_result["true_positives"].append(
                    {
                        "entity_type": prediction.entity_type,
                        "span": f"{prediction.start}-{prediction.end}",
                        "iou": round(iou, 4),
                    }
                )

            for ground_truth_index, expected_entity in enumerate(expected):
                if ground_truth_index in matched_ground_truth:
                    continue
                overall.fn += 1
                per_type[expected_entity.entity_type].fn += 1
                row_result["false_negatives"].append(
                    {
                        "entity_type": expected_entity.entity_type,
                        "span": f"{expected_entity.start}-{expected_entity.end}",
                    }
                )
            row_details.append(row_result)

        summary = {
            "overall": overall.to_dict(),
            "per_entity_type": {label: metric.to_dict() for label, metric in sorted(per_type.items())},
            "evaluation_scope": self.scope,
            "row_scope": self.row_scope,
            "iou_threshold": self.iou_threshold,
            "min_prediction_score": self.min_prediction_score,
            "covered_labels": sorted(covered_labels),
            "ground_truth_rows": len(self._ground_truth_sources),
            "prediction_rows": len(self._prediction_sources),
            "rows_scored": len(row_details),
            "source_mismatch_rows_skipped": len(skipped_source_rows),
            "unscored_predictions": dict(sorted(unscored_predictions.items())),
            "ground_truth_issues": dict(Counter(issue["reason"] for issue in self._ground_truth_issues)),
            "prediction_issues": dict(Counter(issue["reason"] for issue in self._prediction_issues)),
            "reference_review_status":"human_reviewed" if self._ground_truth_sources and len(self._reviewed_reference_rows)==len(self._ground_truth_sources) else "unreviewed_or_synthetic",
            "verified_human_reference":bool(self._ground_truth_sources) and len(self._reviewed_reference_rows)==len(self._ground_truth_sources),
            "matched_source_rows":sum(r in self._prediction_sources and self._source_matches(r) for r in self._ground_truth_sources),
            "prediction_rows_without_reference":len(set(self._prediction_sources)-set(self._ground_truth_sources)),
            "complete_span_coverage":{"reference_spans":len(coverage_details),
                "fully_hidden":sum(e["fully_hidden"] for e in coverage_details),
                "non_whitespace_fully_hidden":sum(e["non_whitespace_fully_hidden"] for e in coverage_details)},
            "character_coverage":{**dict(character_counts),
                "recall":character_counts['intersection']/character_counts['reference'] if character_counts['reference'] else None,
                "jaccard":character_counts['intersection']/character_counts['union'] if character_counts['union'] else None},
        }
        result = {
            "summary": summary,
            "row_details": row_details,
            "source_mismatch_row_indices": skipped_source_rows,
            "ground_truth_issues": self._ground_truth_issues,
            "prediction_issues": self._prediction_issues,
            "span_coverage_details":coverage_details,
        }
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            logger.info("Detailed results saved to %s", output_path)
        return result

    @staticmethod
    def print_summary(summary: Mapping[str, Any]) -> None:
        """Print the score and its evaluation scope."""

        overall = summary["overall"]
        print("PIPELINE EVALUATION RESULTS")
        print(f"Scope: {summary['evaluation_scope']}")
        print(f"Row scope: {summary['row_scope']}")
        print(f"Rows scored: {summary['rows_scored']}")
        print(f"Reference status: {summary['reference_review_status']}")
        print(f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}")
        print(f"Precision={overall['precision']:.4f} Recall={overall['recall']:.4f} F1={overall['f1']:.4f}")
        if summary["unscored_predictions"]:
            print(f"Unscored predictions: {summary['unscored_predictions']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PII pipeline output against reference annotations")
    parser.add_argument("--ground-truth", required=True, help="Reference CSV with context and verified_semantic_tags")
    parser.add_argument("--predictions", required=True, help="Pipeline output CSV with context and pipeline_entities")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--iou", type=float, default=0.5, help="Character-span IoU match threshold")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum prediction confidence included in scoring")
    parser.add_argument("--require-reviewed",action="store_true",help="Require exhaustive human review and reference coverage of every output row")
    parser.add_argument(
        "--scope",
        choices=("strict", "covered-labels"),
        default="strict",
        help="strict requires exhaustive labels; covered-labels excludes never-annotated label types from scoring",
    )
    parser.add_argument(
        "--row-scope",
        choices=("all", "predictions"),
        default="all",
        help="all scores the complete reference/prediction union; predictions scores only output row identifiers",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="Score rows even when ground-truth and prediction context fingerprints differ",
    )
    parser.add_argument("--fail-under-f1", type=float, help="Optional minimum F1 release gate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluator = PipelineEvaluator(
        iou_threshold=args.iou,
        scope=args.scope,
        row_scope=args.row_scope,
        min_prediction_score=args.min_score,
        require_source_match=not args.allow_source_mismatch,
        require_reviewed=args.require_reviewed,
    )
    result = evaluator.evaluate(args.ground_truth, args.predictions, args.output)
    evaluator.print_summary(result["summary"])
    if args.fail_under_f1 is not None and result["summary"]["overall"]["f1"] < args.fail_under_f1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
