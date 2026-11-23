"""Align a completed prediction CSV to a reviewed sheet by source text.

The original prediction file is never changed. Matching requires unique,
identical transcripts after the benchmark's whitespace normalization.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.score_against_benchmark import normalise

csv.field_size_limit(10**9)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def align(reference: list[dict[str, str]], predictions: list[dict[str, str]]) -> list[dict[str, str]]:
    if not reference or not predictions or "context" not in reference[0] or "context" not in predictions[0]:
        raise ValueError("Both CSV files need nonempty context columns")
    by_context: dict[str, dict[str, str]] = {}
    for row in predictions:
        key = normalise(row["context"])
        if key in by_context:
            raise ValueError("Prediction CSV contains duplicate transcript context")
        by_context[key] = row
    ids = set()
    aligned = []
    for row in reference:
        row_id = row.get("row_index", "")
        if not row_id or row_id in ids:
            raise ValueError("Reference has missing or duplicate row IDs")
        ids.add(row_id)
        prediction = by_context.get(normalise(row["context"]))
        if prediction is None:
            raise ValueError(f"No matching prediction transcript for {row_id}")
        aligned.append({**prediction, "row_index": row_id})
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {args.review_sheet.resolve(), args.predictions.resolve()}:
        raise ValueError("Output must not overwrite either input CSV")
    result = align(read_rows(args.review_sheet), read_rows(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Aligned {len(result)} prediction rows to reviewed-sheet IDs: {args.output}")


if __name__ == "__main__":
    main()
