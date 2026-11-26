"""Score a completed prefix of a stopped first-rows benchmark run."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from scripts.run_first_500_benchmark import copy_first_rows
from scripts.evaluate_unified_benchmark import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=int)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    run_dir = args.run_dir.resolve()
    benchmark = run_dir / "benchmark_first500.csv"
    predictions = run_dir / "predictions.csv"
    if not benchmark.is_file() or not predictions.is_file():
        parser.error("Run directory must contain benchmark_first500.csv and predictions.csv")

    scored_dir = run_dir / f"evaluation_first_{args.rows}_{uuid.uuid4().hex}"
    scored_dir.mkdir()
    reference_prefix = scored_dir / "benchmark.csv"
    prediction_prefix = scored_dir / "predictions.csv"
    copy_first_rows(benchmark, reference_prefix, args.rows)
    copy_first_rows(predictions, prediction_prefix, args.rows)
    metrics = scored_dir / "metrics.json"
    result = evaluate(reference_prefix, prediction_prefix, metrics)
    print(f"Metrics: {metrics}")
    for group, scores in result["groups"].items():
        print(f"{group}: {scores['rows_scored']} rows scored")


if __name__ == "__main__":
    main()
