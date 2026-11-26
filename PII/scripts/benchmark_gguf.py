#!/usr/bin/env python3
"""Benchmark the isolated portable GGUF GLiNER2 model on a CSV cohort."""
from __future__ import annotations

import argparse
import csv
import importlib.util
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "pii_pipeline" / "models" / "gliner_gguf.py"


def load_adapter() -> Any:
    """Load the GGUF adapter without initializing the main pipeline package."""

    spec = importlib.util.spec_from_file_location("gguf_benchmark_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load GGUF adapter: {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def run_benchmark(
    input_csv: Path,
    output_csv: Path,
    model_dir: Path,
    threshold: float,
    max_rows: int,
) -> Dict[str, Any]:
    adapter = load_adapter()
    model = adapter.PortableGGUFGLiNER(model_dir=model_dir, threshold=threshold)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    latencies: List[float] = []
    entity_count = 0
    with input_csv.open(newline="", encoding="utf-8-sig") as input_handle, output_csv.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=["row_index", "context", "pipeline_entities", "latency_ms"])
        writer.writeheader()
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break
            text = row.get("context", "")
            started = time.perf_counter()
            entities = model.extract(text)
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)
            entity_count += len(entities)
            writer.writerow(
                {
                    "row_index": row_index,
                    "context": text,
                    "pipeline_entities": json.dumps(entities, ensure_ascii=False),
                    "latency_ms": round(latency_ms, 2),
                }
            )

    total_seconds = sum(latencies) / 1000
    return {
        "rows": len(latencies),
        "entities": entity_count,
        "entities_per_row": entity_count / len(latencies) if latencies else 0.0,
        "threshold": threshold,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency_ms": percentile(latencies, 0.5) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95) if latencies else 0.0,
        "min_latency_ms": min(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "throughput_rows_per_sec": len(latencies) / total_seconds if total_seconds else 0.0,
        "model_labels": list(model.labels),
        "model_dir": str(model_dir.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark portable GGUF GLiNER2")
    parser.add_argument("--input", type=Path, required=True, help="Explicit input CSV, such as the cached unified benchmark")
    parser.add_argument("--output", type=Path, default=Path("output/gguf_benchmark/predictions.csv"))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "gliner2-pii-portable")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max-rows", type=int, default=50)
    args = parser.parse_args()
    summary = run_benchmark(args.input, args.output, args.model_dir, args.threshold, args.max_rows)
    summary_path = args.output.parent / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
