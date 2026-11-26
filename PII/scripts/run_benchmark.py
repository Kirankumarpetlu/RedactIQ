"""Pull the pinned private benchmark, run inference, and score separate groups."""
from __future__ import annotations

import argparse
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
import subprocess
import sys
import uuid
from pathlib import Path

from pii_pipeline.data_assets import ROOT, ensure_data, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New run directory; default: output/benchmark_<unique ID>")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--synthetic-only", action="store_true", help="Run the 28 synthetic cases only")
    selection.add_argument("--real-only", action="store_true", help="Run and evaluate only the 500 real transcripts")
    parser.add_argument("--predictions", type=Path, help="Score existing predictions instead of running inference")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0", "mps"], default="auto")
    parser.add_argument("--indic-device", choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
                        help="Override IndicNER placement; use cpu for GPUs with limited VRAM")
    parser.add_argument("--config", type=Path, default=ROOT / "config/pipeline_config.yaml")
    args = parser.parse_args()
    group = ("synthetic_regression" if args.synthetic_only else
             "real_development_pending" if args.real_only else None)
    output = (args.output_dir or ROOT / "output" / f"benchmark_{uuid.uuid4().hex}").resolve()
    if output.exists():
        parser.error("Choose a new output directory to avoid overwriting a benchmark run")
    if args.predictions and not args.predictions.is_file():
        parser.error("Predictions file does not exist")
    paths = ensure_data()
    benchmark = paths["data/gold/unified_benchmark.csv"]
    output.mkdir(parents=True, exist_ok=False)
    predictions = args.predictions.resolve() if args.predictions else output / "predictions.csv"
    if not args.predictions:
        command = [sys.executable, "-X", "utf8", "-m", "pii_pipeline.run_csv_through_stage9",
                   "--config", str(args.config.resolve()), "--input", str(benchmark),
                   "--output", str(predictions), "--device", args.device,
                   "--save-diagnostics", "--enable-profiling"]
        if group is not None:
            command.extend(["--benchmark-group", group])
        if args.indic_device is not None:
            command.extend(["--indic-device", args.indic_device])
        # A failed/incomplete inference run must never proceed to scoring.
        subprocess.run(command, cwd=ROOT, check=True)
    from scripts.evaluate_unified_benchmark import evaluate
    result = evaluate(benchmark, predictions, output / "metrics.json",
                      selected_groups=[group] if group is not None else None)
    manifest = load_manifest()
    (output / "data_provenance.json").write_text(json.dumps({
        "repo_id": manifest["repo_id"], "repo_type": "dataset", "revision": manifest["revision"],
        "assets": {item["path"]: item["sha256"] for item in manifest["files"] if item["path"] in paths},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {output / 'metrics.json'}")
    for group, metrics in result["groups"].items():
        print(f"{group}: {metrics['rows_scored']} rows scored")


if __name__ == "__main__":
    main()
