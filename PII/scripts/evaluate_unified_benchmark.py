"""Compare saved predictions with a unified benchmark, reporting each group separately."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pii_pipeline import json_codec as json
from scripts.evaluate_pipeline import Entity, Metrics, PipelineEvaluator, normalize_entity_type


GROUPS = {
    "synthetic_regression": ("synthetic_reference", "synthetic_exhaustive"),
    "real_development_pending": ("pending", "incomplete"),
}


def _snapshot(path: Path):
    csv.field_size_limit(10**9)
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    indexed = {}
    for number, row in enumerate(rows, 1):
        row_id = (row.get("row_index") or "").strip()
        if not row_id or row_id in indexed or not isinstance(row.get("context"), str):
            raise ValueError(f"{path.name} row {number}: missing context or missing/duplicate row_index")
        indexed[row_id] = row
    return raw, indexed


def _spans(row, column):
    raw = row.get(column)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Row {row['row_index']}: missing {column}; use [] only for explicit negatives")
    try:
        tags = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row['row_index']}: invalid JSON in {column}") from exc
    if not isinstance(tags, list):
        raise ValueError(f"Row {row['row_index']}: {column} must be a JSON list")
    text, spans, seen = row["context"], [], set()
    for tag in tags:
        if not isinstance(tag, dict):
            raise ValueError(f"Row {row['row_index']}: annotation must be an object")
        start, end, label = tag.get("start"), tag.get("end"), tag.get("entity_type")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)
                or tag.get("text") != text[start:end]
                or not isinstance(label, str) or not label.strip()):
            raise ValueError(f"Row {row['row_index']}: invalid exact span in {column}")
        entity_type = normalize_entity_type(label)
        key = (entity_type, start, end)
        if key in seen:
            raise ValueError(f"Row {row['row_index']}: duplicate annotation in {column}")
        seen.add(key)
        spans.append(Entity(entity_type, start, end, text[start:end]))
    return spans


def evaluate(benchmark: Path, predictions: Path, output: Path | None = None, iou: float = 0.5,
             selected_groups=None):
    """Validate complete source alignment and score groups without a mixed aggregate."""
    benchmark, predictions = Path(benchmark), Path(predictions)
    if output is not None:
        output = Path(output)
        if output.resolve() in {benchmark.resolve(), predictions.resolve()}:
            raise ValueError("Output must differ from both input files")
        if output.exists():
            raise FileExistsError("Choose a fresh output file")
    matcher = PipelineEvaluator(iou_threshold=iou)
    reference_raw, references = _snapshot(benchmark)
    prediction_raw, predicted_rows = _snapshot(predictions)
    selected = set(selected_groups) if selected_groups is not None else set(GROUPS)
    if not selected or not selected <= set(GROUPS):
        raise ValueError("Select one or more supported benchmark groups")
    if any(row.get("benchmark_group") not in GROUPS for row in references.values()):
        raise ValueError("Benchmark contains an unknown benchmark_group")
    all_reference_ids = set(references)
    references = {row_id: row for row_id, row in references.items() if row["benchmark_group"] in selected}
    if not references:
        raise ValueError("No benchmark rows belong to the selected groups")
    if set(predicted_rows) - all_reference_ids or set(references) - set(predicted_rows):
        raise ValueError("Benchmark and predictions must contain the same complete row IDs")
    predicted_rows = {row_id: predicted_rows[row_id] for row_id in references}

    groups = {}
    for row_id, row in references.items():
        predicted = predicted_rows[row_id]
        if row["context"] != predicted["context"]:
            raise ValueError(f"Row {row_id}: original contexts differ")
        fingerprint = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
        if row.get("source_fingerprint") != fingerprint:
            raise ValueError(f"Row {row_id}: source_fingerprint does not match original context")
        group = row.get("benchmark_group")
        if group not in GROUPS:
            raise ValueError(f"Row {row_id}: unknown benchmark_group")
        if (row.get("review_status"), row.get("annotation_scope")) != GROUPS[group]:
            raise ValueError(f"Row {row_id}: unexpected review status or annotation scope for {group}")
        expected = _spans(row, "chosen_reference_tags")
        if group == "real_development_pending":
            verified_raw = row.get("verified_semantic_tags")
            if not isinstance(verified_raw, str):
                raise ValueError(f"Row {row_id}: missing verified_semantic_tags column")
            if verified_raw.strip():
                raise ValueError(f"Row {row_id}: pending annotations must not be marked verified")
        else:
            verified = _spans(row, "verified_semantic_tags")
            if set(expected) != set(verified):
                raise ValueError(f"Row {row_id}: synthetic chosen reference differs from verified reference")
        found = _spans(predicted, "pipeline_entities")
        try:
            prediction_count = int(predicted["total_entities_count"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Row {row_id}: missing or invalid total_entities_count") from exc
        if prediction_count != len(found):
            raise ValueError(f"Row {row_id}: prediction entity count differs")

        state = groups.setdefault(group, {"metrics": Metrics(), "per_entity_type": defaultdict(Metrics),
                                         "counts": Counter(), "rows": []})
        matches = matcher._match_entities(expected, found)
        matched_reference = {pair[0] for pair in matches.values()}
        tp, fp, fn = len(matches), len(found) - len(matches), len(expected) - len(matches)
        state["metrics"].tp += tp
        state["metrics"].fp += fp
        state["metrics"].fn += fn
        for index, span in enumerate(found):
            metric = state["per_entity_type"][span.entity_type]
            if index in matches:
                metric.tp += 1
            else:
                metric.fp += 1
        for index, span in enumerate(expected):
            if index not in matched_reference:
                state["per_entity_type"][span.entity_type].fn += 1
        reference_chars = {i for span in expected for i in range(span.start, span.end)}
        selected_chars = {i for span in found for i in range(span.start, span.end)}
        fully_hidden = sum(set(range(span.start, span.end)) <= selected_chars for span in expected)
        state["counts"].update(rows=1, reference_spans=len(expected), prediction_spans=len(found),
                               reference_characters=len(reference_chars),
                               masked_reference_characters=len(reference_chars & selected_chars),
                               fully_hidden_reference_spans=fully_hidden)
        state["rows"].append({"row_index": row_id, "tp": tp, "fp": fp, "fn": fn,
                              "fully_hidden_reference_spans": fully_hidden})

    results = {}
    for group, state in sorted(groups.items()):
        counts = state["counts"]
        total = counts["reference_characters"]
        masked = counts["masked_reference_characters"]
        provisional = group == "real_development_pending"
        results[group] = {
            "comparison_status": ("provisional_pending_reference_agreement" if provisional
                                  else "synthetic_regression"),
            "verified_human_real_reference": False,
            "review_status": GROUPS[group][0], "annotation_scope": GROUPS[group][1],
            "rows_scored": counts["rows"], "reference_spans": counts["reference_spans"],
            "prediction_spans": counts["prediction_spans"],
            "typed_span_metrics": state["metrics"].to_dict(),
            "per_entity_type": {label: metric.to_dict() for label, metric in sorted(state["per_entity_type"].items())},
            "exact_character_coverage": {
                "reference_characters": total, "masked_reference_characters": masked,
                "unmasked_reference_characters": total - masked,
                "coverage_fraction": masked / total if total else None,
                "fully_hidden_reference_spans": counts["fully_hidden_reference_spans"],
                "ignores_entity_type": True,
            },
            "row_details": state["rows"],
            "interpretation": ("Agreement with incomplete pending annotations; missing labels can hide misses and create apparent false positives."
                               if provisional else "Synthetic regression coverage; does not establish human-reviewed real-world accuracy."),
        }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_groups": sorted(selected),
        "reference_column": "chosen_reference_tags", "iou_threshold": iou,
        "confidence_filter_applied": False,
        "evaluation_scope": "strict_typed_span_comparison_per_group",
        "groups": results,
        "source_files": {str(path.resolve()): hashlib.sha256(raw).hexdigest()
                         for path, raw in ((benchmark, reference_raw), (predictions, prediction_raw))},
        "limitations": ["No combined accuracy score is calculated across synthetic and pending real groups.",
                        "Character coverage measures annotated source positions covered by prediction spans, regardless of type.",
                        "Saved predictions were compared without model inference or changes to reference review status."],
    }
    for path, raw in ((benchmark, reference_raw), (predictions, prediction_raw)):
        if path.read_bytes() != raw:
            raise ValueError(f"Input changed while scoring: {path}")
    if output is not None:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, help="Default: pull the pinned private benchmark")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--group", action="append", choices=sorted(GROUPS),
                        help="Repeat to select groups; default scores all groups separately")
    args = parser.parse_args()
    benchmark = args.benchmark
    if benchmark is None:
        from pii_pipeline.data_assets import ensure_data
        benchmark = ensure_data(("benchmark",))["data/gold/unified_benchmark.csv"]
    evaluate(benchmark, args.predictions, args.output, args.iou, args.group)


if __name__ == "__main__":
    main()
