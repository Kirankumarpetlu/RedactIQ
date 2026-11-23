"""Build one readable Markdown report from a timed pipeline benchmark.

The report keeps the full source/model × language × class matrix in the
existing CSV artifact, and presents every non-empty cell in collapsible
sections.  This avoids hiding model-specific false positives behind a single
aggregate score.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pii_pipeline import json_codec as json


def _metric(value: Any) -> str:
    return "—" if value in (None, "") else f"{float(value):.2f}%"


def _int(value: Any) -> str:
    return str(int(value or 0))


def _table(rows: Iterable[dict[str, Any]], fields: list[tuple[str, str]], lines: list[str]) -> None:
    lines.append("| " + " | ".join(label for _, label in fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = []
        for field, _ in fields:
            value = row[field]
            values.append(_metric(value) if field.endswith("_pct") else str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


def _nonempty(row: dict[str, str]) -> bool:
    return any(int(row[key] or 0) for key in ("reference_spans", "predicted_spans", "tp", "fp", "fn"))


def _source_title(row: dict[str, str]) -> str:
    return f"{row['source']} ({row['source_kind']})"


def build_report(score_path: Path, latency_path: Path, matrix_path: Path, output_path: Path) -> None:
    score = json.loads(score_path.read_bytes())
    with latency_path.open(encoding="utf-8", newline="") as handle:
        latency = list(csv.DictReader(handle))
    with matrix_path.open(encoding="utf-8", newline="") as handle:
        matrix = list(csv.DictReader(handle))

    exact = score["exact"]
    iou = score["iou_50"]
    lines = [
        "# 500-row PII pipeline benchmark report",
        "",
        "## Scope and interpretation",
        "",
        f"- Rows: **{score['rows']}**; marked reference spans: **{score['marked_spans']}**; final predictions: **{score['predictions']}**.",
        "- Scores use entity type plus span matching. `Exact` requires identical start/end offsets; the model matrix below uses exact matching.",
        "- The reference status is `provisional_pending_incomplete_ai_review`; precision is therefore **apparent precision** against currently marked spans, not a final human-verified precision value.",
        "- A dash for recall or precision means its denominator is zero for that language/class cell.",
        "",
        "## Final pipeline quality",
        "",
    ]
    summary_rows = []
    for name, result in (("Exact span", exact), ("IoU ≥ 0.50", iou)):
        tp = int(result["matched"])
        fp = int(result["unmatched_predictions"])
        fn = int(result["missed"])
        summary_rows.append({
            "match": name, "tp": tp, "fp": fp, "fn": fn,
            "precision_pct": result["apparent_precision_if_exhaustive"] * 100,
            "recall_pct": result["marked_span_recall"] * 100,
            "f1_pct": result["apparent_f1_if_exhaustive"] * 100,
        })
    _table(summary_rows, [("match", "Match"), ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
                          ("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")], lines)

    lines.extend(["### Final pipeline: exact score by entity class", ""])
    final_class_rows = []
    for label, result in sorted(exact["per_type"].items()):
        tp, fp, fn = int(result["matched"]), int(result["unmatched_predictions"]), int(result["missed"])
        final_class_rows.append({"entity_class": label, "tp": tp, "fp": fp, "fn": fn,
                                 "precision_pct": result["apparent_precision_if_exhaustive"] * 100,
                                 "recall_pct": result["marked_span_recall"] * 100,
                                 "f1_pct": result["apparent_f1_if_exhaustive"] * 100})
    _table(final_class_rows, [("entity_class", "Class"), ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
                              ("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")], lines)

    lines.extend(["## Latency by pipeline stage", "", "All times are milliseconds. Rows are sorted by total measured time.", ""])
    latency.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    _table(latency, [("stage_name", "Stage"), ("depth", "Depth"), ("calls", "Calls"), ("total_ms", "Total ms"),
                     ("mean_ms", "Mean ms"), ("median_ms", "Median ms"), ("p90_ms", "P90 ms"),
                     ("p95_ms", "P95 ms"), ("p99_ms", "P99 ms"), ("max_ms", "Max ms")], lines)

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matrix:
        by_source[row["source"]].append(row)
    source_order = sorted(by_source, key=lambda name: name.lower())

    lines.extend(["## Exact P/R/F1 by recognizer or model: global", "",
                  "Each row is the model's raw candidate output before shared conflict resolution, except `final_pipeline`, which is the accepted output.", ""])
    global_rows = []
    for source in source_order:
        row = next(item for item in by_source[source] if item["language"] == "ALL" and item["entity_class"] == "ALL")
        global_rows.append({"source": _source_title(row), "tp": _int(row["tp"]), "fp": _int(row["fp"]), "fn": _int(row["fn"]),
                            "precision_pct": row["precision_pct"], "recall_pct": row["recall_pct"], "f1_pct": row["f1_pct"]})
    _table(global_rows, [("source", "Source"), ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
                         ("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")], lines)

    lines.extend(["## Exact P/R/F1 by source, entity class, and language", "",
                  "Only cells with a reference span or a prediction are shown. The complete zero-inclusive matrix is retained in `exact_prf_by_source_language_class.csv`.", ""])
    fields = [("language", "Language"), ("entity_class", "Class"), ("reference_spans", "Gold"),
              ("predicted_spans", "Predicted"), ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
              ("precision_pct", "Precision"), ("recall_pct", "Recall"), ("f1_pct", "F1")]
    for source in source_order:
        rows = [row for row in by_source[source] if row["entity_class"] != "ALL" and _nonempty(row)]
        if not rows:
            continue
        rows.sort(key=lambda row: (row["language"] != "ALL", row["language"], row["entity_class"]))
        lines.append(f"<details><summary>{_source_title(rows[0])}</summary>")
        lines.append("")
        _table(rows, fields, lines)
        lines.extend(["</details>", ""])

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.score, args.latency, args.matrix, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
