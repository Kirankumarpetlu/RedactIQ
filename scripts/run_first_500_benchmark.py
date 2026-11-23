"""Run and score the first 500 rows of the pinned unified benchmark in CSV order.

CUDA uses the measured balanced-latency profile by default; --config overrides it.
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
import os
import subprocess
import sys
import uuid
from itertools import islice
from pathlib import Path

from pii_pipeline.data_assets import ROOT, ensure_data


def default_config_for_device(device: str) -> Path:
    """Use the measured CUDA profile for benchmark runs on a CUDA device."""
    name = "pipeline_config.balanced_latency.yaml" if device in {"cuda", "cuda:0"} else "pipeline_config.yaml"
    return ROOT / "config" / name


def copy_first_rows(source: Path, destination: Path, count: int = 500) -> None:
    """Copy a CSV prefix without changing text, row IDs, or reference columns."""
    csv.field_size_limit(10**9)
    with source.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("Benchmark CSV has no header")
        rows = list(islice(reader, count))
    if len(rows) != count:
        raise ValueError(f"Benchmark has only {len(rows)} rows; expected {count}")
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "mps"], default="cpu")
    parser.add_argument("--indic-device", choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
                        help="Optional IndicNER placement, for example cpu on a small GPU")
    parser.add_argument("--config", type=Path, help="Override the selected CPU/CUDA profile")
    parser.add_argument("--output-dir", type=Path, help="New directory; defaults to output/first500_<ID>")
    args = parser.parse_args()
    config = args.config or default_config_for_device(args.device)

    if args.device in {"cuda", "cuda:0"}:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            parser.error("CUDA was requested but no usable CUDA device is available; refusing CPU fallback")

    benchmark = ensure_data()["data/gold/unified_benchmark.csv"]
    output = (args.output_dir or ROOT / "output" / f"first500_{uuid.uuid4().hex}").resolve()
    if output.exists():
        parser.error("Choose a new output directory to avoid overwriting a run")
    output.mkdir(parents=True)
    selected = output / "benchmark_first500.csv"
    predictions = output / "predictions.csv"
    metrics = output / "metrics.json"
    copy_first_rows(benchmark, selected)
    (output / "selection.json").write_text(json.dumps({
        "selection": "first 500 CSV data rows in file order",
        "source_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
        "selected_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "rows": 500,
        "config": str(config.resolve()),
    }, indent=2) + "\n", encoding="utf-8")

    command = [sys.executable, "-X", "utf8", "-m", "pii_pipeline.run_csv_through_stage9",
               "--config", str(config.resolve()), "--input", str(selected),
               "--output", str(predictions), "--device", args.device]
    if args.indic_device is not None:
        command.extend(["--indic-device", args.indic_device])
    # Some Windows terminals default to cp1252; model startup logs include
    # Unicode and can otherwise fail before the first row is processed.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(command, cwd=ROOT, check=True, env=child_env)

    from scripts.evaluate_unified_benchmark import evaluate

    result = evaluate(selected, predictions, metrics)
    print(f"Metrics: {metrics}")
    for group, scores in result["groups"].items():
        print(f"{group}: {scores['rows_scored']} rows scored")


if __name__ == "__main__":
    main()
