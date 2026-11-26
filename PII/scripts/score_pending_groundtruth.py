"""Score an aligned prediction CSV against provisional offset annotations.

Pending/incomplete annotations support recall on marked spans. Unmatched
predictions are reported separately: apparent precision assumes annotations
are exhaustive and must not be presented as verified precision.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pii_pipeline import json_codec as json
from pii_pipeline.taxonomy import canonical_entity_type


def _label(value: str) -> str:
    label = canonical_entity_type(value)
    return "MEDICAL_ID" if label == "MEDICAL_RECORD_NUMBER" else label


def _spans(row: dict[str, str], column: str) -> list[dict]:
    text = row["context"]
    spans = json.loads(row[column] or "[]")
    if not isinstance(spans, list):
        raise ValueError(f"Row {row['row_index']}: {column} must be a list")
    for item in spans:
        a, b = item.get("start"), item.get("end")
        if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text)
                or item.get("text", text[a:b]) != text[a:b]):
            raise ValueError(f"Row {row['row_index']}: invalid {column} offset/text")
    return spans


def _iou(a: dict, b: dict) -> float:
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - overlap
    return overlap / union if union else 0.0


def _match(gold: list[dict], predicted: list[dict], *, exact: bool) -> tuple[set[int], set[int]]:
    options = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(predicted):
            if _label(g["entity_type"]) != _label(p["entity_type"]):
                continue
            overlap = _iou(g, p)
            if ((g["start"], g["end"]) == (p["start"], p["end"]) if exact else overlap >= .5):
                options.append((overlap, gi, pi))
    matched_g, matched_p = set(), set()
    for _, gi, pi in sorted(options, reverse=True):
        if gi not in matched_g and pi not in matched_p:
            matched_g.add(gi)
            matched_p.add(pi)
    return matched_g, matched_p


def score(reference: Path, predictions: Path) -> dict:
    with reference.open(encoding="utf-8-sig", newline="") as stream:
        gold_rows = list(csv.DictReader(stream))
    with predictions.open(encoding="utf-8-sig", newline="") as stream:
        predicted_rows = list(csv.DictReader(stream))
    if len(gold_rows) != len(predicted_rows):
        raise ValueError("Reference and prediction row counts differ")
    for g, p in zip(gold_rows, predicted_rows):
        if g["row_index"] != p["row_index"] or g["context"] != p["context"]:
            raise ValueError(f"Row alignment differs at {g['row_index']}")
    totals = {"exact": Counter(), "iou_50": Counter()}
    by_type = {"exact": {}, "iou_50": {}}
    missed = []
    unmatched_predictions = []
    latencies = []
    for g, p in zip(gold_rows, predicted_rows):
        gold = _spans(g, "verified_semantic_tags")
        predicted = _spans(p, "pipeline_entities")
        latencies.append(float(p["latency_ms"]))
        for key, exact in (("exact", True), ("iou_50", False)):
            matched_g, matched_p = _match(gold, predicted, exact=exact)
            totals[key].update(tp=len(matched_g), fn=len(gold) - len(matched_g),
                               unmatched_predictions=len(predicted) - len(matched_p))
            types = by_type[key]
            for i, item in enumerate(gold):
                label = _label(item["entity_type"])
                types.setdefault(label, Counter()).update(tp=1 if i in matched_g else 0,
                                                            fn=0 if i in matched_g else 1)
            for i, item in enumerate(predicted):
                if i not in matched_p:
                    types.setdefault(_label(item["entity_type"]), Counter()).update(unmatched_predictions=1)
            if not exact:
                missed.extend({"row_index": g["row_index"], "entity_type": item["entity_type"],
                               "start": item["start"], "end": item["end"]}
                              for i, item in enumerate(gold) if i not in matched_g)
                unmatched_predictions.extend({"row_index": g["row_index"], "entity_type": item["entity_type"],
                                               "start": item["start"], "end": item["end"]}
                                              for i, item in enumerate(predicted) if i not in matched_p)

    def metrics(counts: Counter) -> dict:
        tp, fn, unverified = (counts[k] for k in ("tp", "fn", "unmatched_predictions"))
        apparent_precision = tp / (tp + unverified) if tp + unverified else 0
        recall = tp / (tp + fn) if tp + fn else 0
        return {"matched": tp, "missed": fn, "unmatched_predictions": unverified,
                "marked_span_recall": round(recall, 4),
                "apparent_precision_if_exhaustive": round(apparent_precision, 4),
                "apparent_f1_if_exhaustive": round(2 * recall * apparent_precision / (recall + apparent_precision), 4)
                if recall + apparent_precision else 0}

    ordered = sorted(latencies)
    percentile = lambda q: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]
    return {
        "reference_status": "provisional_pending_incomplete_ai_review",
        "rows": len(gold_rows), "marked_spans": sum(len(_spans(r, "verified_semantic_tags")) for r in gold_rows),
        "predictions": sum(len(_spans(r, "pipeline_entities")) for r in predicted_rows),
        "latency_ms": {"mean": round(statistics.mean(latencies), 2),
                       "median": round(statistics.median(latencies), 2),
                       "p90": round(percentile(.90), 2), "p95": round(percentile(.95), 2),
                       "p99": round(percentile(.99), 2), "max": round(max(latencies), 2)},
        "exact": {**metrics(totals["exact"]), "per_type": {k: metrics(v) for k, v in by_type["exact"].items()}},
        "iou_50": {**metrics(totals["iou_50"]), "per_type": {k: metrics(v) for k, v in by_type["iou_50"].items()}},
        "missed_marked_spans": missed,
        "unmatched_predictions_for_review": unmatched_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.reference, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "marked_spans", "predictions", "latency_ms", "exact", "iou_50")},
                     ensure_ascii=False, indent=2))
    print("Report:", args.output)


if __name__ == "__main__":
    main()
