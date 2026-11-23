#!/usr/bin/env python3
"""Measure the configured Stage 9 pipeline using repeated input passes.

Defaults to the pinned unified benchmark. Warmup is excluded from latency
statistics; optional profiling includes its measurement overhead. Only the
backends selected in the configuration are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pii_pipeline import json_codec as json


def _percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * quantile
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def _entity_signature(result: Dict[str, Any]) -> set[tuple[str, int, int]]:
    """Stable model-independent entity contract used for experiment gating."""
    return {
        (str(entity["entity_type"]), int(entity["start"]), int(entity["end"]))
        for entity in result.get("entities_detected", [])
        if all(key in entity for key in ("entity_type", "start", "end"))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Pipeline YAML/JSON configuration path")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input .csv, .json, or .txt file; defaults to the pinned unified benchmark",
    )
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=None, help="Optional GLiNER threshold override")
    parser.add_argument("--device", default=None, help="Optional device override")
    parser.add_argument("--cpu-threads", type=int, default=None, help="Override runtime.cpu_threads for a measured tuning run")
    parser.add_argument("--repeat", type=int, default=1, help="Measured passes over the selected rows")
    parser.add_argument("--warmup-rows", type=int, default=2, help="Initial input rows to warm up before timing")
    parser.add_argument("--profile", action="store_true", help="Include per-stage timings (adds measurement overhead)")
    parser.add_argument(
        "--baseline-config",
        default=None,
        help="Optional approved configuration. Compares normalized entities and redacted output without benchmarking alternative backends.",
    )
    parser.add_argument(
        "--min-entity-f1",
        type=float,
        default=1.0,
        help="Minimum entity-span F1 against --baseline-config (default: exact agreement).",
    )
    parser.add_argument(
        "--require-identical-redaction",
        action="store_true",
        help="Fail the experiment gate unless every candidate redacted output matches the approved baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / f"configured_pipeline_benchmark_{time.time_ns()}",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup_rows < 0 or (args.cpu_threads is not None and args.cpu_threads < 1):
        parser.error("repeat and cpu-threads must be positive; warmup-rows must be nonnegative")

    from pii_pipeline.config import load_pipeline_config
    from pii_pipeline.data_assets import ensure_data
    from pii_pipeline.model_cache import configure_runtime, get_hardware_info
    from pii_pipeline.pipeline import Stage9Pipeline
    from pii_pipeline.run_csv_through_stage9 import _load_input_records

    cfg = load_pipeline_config(args.config)
    if args.cpu_threads is not None:
        cfg.runtime.cpu_threads = args.cpu_threads
    assets = ensure_data(("runtime",) if args.input is not None else ("benchmark", "runtime"))
    if args.input is None:
        args.input = assets["data/gold/unified_benchmark.csv"]
    configure_runtime(cfg.runtime.cpu_threads, cfg.runtime.enable_mps)
    threshold = args.threshold if args.threshold is not None else cfg.models.gliner_primary.threshold
    rows, input_format = _load_input_records(args.input, args.max_rows)
    contexts = [str(row.get("context", "")) for row in rows]
    if not contexts:
        raise ValueError("The benchmark input contains no processable records.")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    requested_device = args.device or cfg.runtime.device
    print("Benchmarking one configured pipeline only (no ONNX/GGUF comparison).", flush=True)
    print(f"Configured device: {requested_device} | rows: {len(contexts)}", flush=True)

    started = time.perf_counter()
    pipeline = Stage9Pipeline(
        config=cfg,
        device=requested_device,
        gliner_threshold=threshold,
    )
    load_time_ms = round((time.perf_counter() - started) * 1000, 2)
    hardware = get_hardware_info(pipeline.device)

    baseline_pipeline = None
    if args.baseline_config:
        baseline_cfg = load_pipeline_config(args.baseline_config)
        if baseline_cfg.runtime.cpu_threads != cfg.runtime.cpu_threads:
            raise ValueError("Compare thread settings in separate processes; PyTorch thread count is process-wide.")
        baseline_pipeline = Stage9Pipeline(
            config=baseline_cfg,
            device=args.device or baseline_cfg.runtime.device,
            gliner_threshold=baseline_cfg.models.gliner_primary.threshold,
        )

    # Warmup is excluded from measured rows.
    for context in contexts[:args.warmup_rows]:
        pipeline.run(context)
        if baseline_pipeline is not None:
            baseline_pipeline.run(context)
    latencies_ms: List[float] = []
    stage_timings: Dict[str, List[float]] = {}
    entity_count = 0
    matched_entities = candidate_entities = baseline_entities = identical_redactions = exact_entity_rows = 0
    measured_contexts = contexts * args.repeat
    for index, context in enumerate(measured_contexts, start=1):
        row_started = time.perf_counter()
        result = pipeline.run(context, enable_profiling=args.profile)
        latency_ms = (time.perf_counter() - row_started) * 1000
        latencies_ms.append(latency_ms)
        entity_count += len(result.get("entities_detected", []))
        for stage in result.get("profiling", {}).get("stages", []):
            stage_timings.setdefault(stage["stage_name"], []).append(stage["duration_ms"])
        if baseline_pipeline is not None:
            baseline_result = baseline_pipeline.run(context)
            candidate_signature = _entity_signature(result)
            baseline_signature = _entity_signature(baseline_result)
            matched_entities += len(candidate_signature & baseline_signature)
            candidate_entities += len(candidate_signature)
            baseline_entities += len(baseline_signature)
            exact_entity_rows += int(candidate_signature == baseline_signature)
            identical_redactions += int(result.get("redacted_text") == baseline_result.get("redacted_text"))
        print(f"[{index}/{len(measured_contexts)}] {latency_ms:.2f} ms", flush=True)

    total_seconds = sum(latencies_ms) / 1000
    precision = matched_entities / candidate_entities if candidate_entities else float(baseline_entities == 0)
    recall = matched_entities / baseline_entities if baseline_entities else float(candidate_entities == 0)
    entity_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    quality_gate = None
    if baseline_pipeline is not None:
        quality_gate = {
            "baseline_config": str(args.baseline_config),
            "entity_span_precision": round(precision, 6),
            "entity_span_recall": round(recall, 6),
            "entity_span_f1": round(entity_f1, 6),
            "exact_entity_rows": exact_entity_rows,
            "identical_redaction_rows": identical_redactions,
            "min_entity_f1": args.min_entity_f1,
            "require_identical_redaction": args.require_identical_redaction,
            "passed": entity_f1 >= args.min_entity_f1
            and (not args.require_identical_redaction or identical_redactions == len(measured_contexts)),
        }
    summary: Dict[str, Any] = {
        "benchmark_type": "configured_pipeline_single_backend",
        "input_format": input_format,
        "rows": len(latencies_ms),
        "unique_input_rows": len(contexts),
        "repeat": args.repeat,
        "warmup_rows": min(args.warmup_rows, len(contexts)),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "cpu_threads": cfg.runtime.cpu_threads,
        "profiling_enabled": args.profile,
        "latencies_ms": latencies_ms,
        "stage_timings": {name: {"calls": len(values), "mean_ms": mean(values), "total_ms": sum(values)}
                          for name, values in stage_timings.items()},
        "load_time_ms": load_time_ms,
        "mean_latency_ms": round(mean(latencies_ms), 2),
        "p50_latency_ms": round(median(latencies_ms), 2),
        "p95_latency_ms": round(_percentile(latencies_ms, 0.95), 2),
        "throughput_rows_per_sec": round(len(latencies_ms) / total_seconds, 3) if total_seconds else 0.0,
        "entity_count": entity_count,
        "device": pipeline.device,
        "hardware": hardware,
        "models": {
            "gliner_primary": cfg.models.gliner_primary.model_name_or_path,
            "gliner_primary_adapter": cfg.models.gliner_primary.adapter,
            "gliner_vanilla_fallback": cfg.models.gliner_vanilla_fallback.model_name_or_path,
            "gliner_vanilla_fallback_adapter": cfg.models.gliner_vanilla_fallback.adapter,
            "spacy": cfg.models.spacy.model_name,
            "indic_ner_path": cfg.models.indic_ner.model_path,
            "indic_ner_weights_path": cfg.models.indic_ner.model_weights_path,
        },
        "inference_backends": pipeline.inference_backends,
        "quality_gate": quality_gate,
    }
    json_path = args.output_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark complete: {json_path}", flush=True)
    if quality_gate and not quality_gate["passed"]:
        print("Quality gate failed; candidate configuration must not replace the approved baseline.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
