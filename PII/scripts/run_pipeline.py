#!/usr/bin/env python3
"""
CLI entry point: run Stage 9 (PII Detection & Redaction) against real
transcripts in Plain Text or JSON format.

Features:
- Supports both Plain Text (.txt) and Structured JSON (.json) transcript inputs.
- For JSON transcripts, outputs .redacted.json preserving the exact original structure with PII/PHI redacted.
- Loads model weights ONCE at startup into memory cache (zero per-file reload overhead).
- Auto-detects ANY GPU (CUDA 0..N, Apple Silicon MPS, or CPU fallback).

Usage:
    python scripts/run_pipeline.py --input sample_transcript.json --output-dir out/
    python scripts/run_pipeline.py --input sample_transcript.txt --output-dir out/
    python scripts/run_pipeline.py --input sample_transcript.json --output-dir GPU_performed_outputs/ --device cuda --profile

For a directory of transcripts:
    python scripts/run_pipeline.py --input transcripts/ --output-dir out/
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pii_pipeline import json_codec as json
from pii_pipeline import ModelCache, Stage9Pipeline, auto_detect_device, get_hardware_info, get_profiler
from pii_pipeline.config import load_pipeline_config
from pii_pipeline.model_cache import configure_runtime


def process_file(pipeline: Stage9Pipeline, in_path: Path, out_dir: Path, profile: bool = False, enable_profiling: bool = False) -> None:
    is_json = in_path.suffix.lower() == ".json"
    t0 = time.perf_counter()

    if is_json:
        raw_content = in_path.read_text(encoding="utf-8")
        json_data = json.loads(raw_content)
        result = pipeline.run_json(json_data, enable_profiling=enable_profiling)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        redacted_path = out_dir / f"{in_path.stem}.redacted.json"
        audit_path = out_dir / f"{in_path.stem}.audit.json"

        redacted_path.write_text(json.dumps(result["redacted_json"], indent=2, ensure_ascii=False), encoding="utf-8")
        audit_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        n_entities = len(result["entities_detected"])
        fields = result.get("fields_processed", 0)
        profile_info = f" in {latency_ms} ms (device: {pipeline.device})" if profile else ""
        print(f"[{in_path.name}] (JSON, {fields} fields) {n_entities} entities detected{profile_info} -> {redacted_path.name}, {audit_path.name}")
    else:
        text = in_path.read_text(encoding="utf-8")
        result = pipeline.run(text, enable_profiling=enable_profiling)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        redacted_path = out_dir / f"{in_path.stem}.redacted.txt"
        audit_path = out_dir / f"{in_path.stem}.audit.json"

        redacted_path.write_text(result["redacted_text"], encoding="utf-8")
        audit_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        n_entities = len(result["entities_detected"])
        profile_info = f" in {latency_ms} ms (device: {pipeline.device})" if profile else ""
        print(f"[{in_path.name}] {n_entities} entities detected{profile_info} -> {redacted_path.name}, {audit_path.name}")

    if enable_profiling and "profiling" in result:
        from pii_pipeline.profiling import print_report
        print_report(result["profiling"])
        
        # Save profiling data to audit
        profiling_path = out_dir / f"{in_path.stem}.profiling.json"
        profiling_path.write_text(json.dumps(result["profiling"], indent=2), encoding="utf-8")
        print(f"  Profiling data saved to {profiling_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 9 PII/PHI detection & redaction (Text or JSON).")
    parser.add_argument("--input", required=True, help="Transcript file (.txt or .json) or directory of files")
    parser.add_argument("--output-dir", default="output", help="Directory to write redacted output + audit JSON (default: ./output/)")
    parser.add_argument("--config", default=None, help="Pipeline YAML/JSON configuration path")
    parser.add_argument("--threshold", type=float, default=None, help="Optional GLiNER2 confidence override")
    parser.add_argument("--device", default=None, help="Optional device override: cuda, cuda:0, mps, cpu, or auto")
    parser.add_argument("--profile", action="store_true", help="Print execution timing and device profiling")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable detailed per-stage CPU/core profiling")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_pipeline_config(args.config)
    configure_runtime(cfg.runtime.cpu_threads, cfg.runtime.enable_mps)
    threshold = args.threshold if args.threshold is not None else cfg.models.gliner_primary.threshold
    resolved_device = auto_detect_device(args.device or cfg.runtime.device)
    hw = get_hardware_info(resolved_device)
    if args.verbose or args.profile:
        print(f"Hardware: {hw.get('current_gpu_name')} | Device: {resolved_device}")

    # Preload all model weights once at startup into memory cache
    t_load = time.perf_counter()
    resolved_device = ModelCache.preload_all(
        device=resolved_device,
        language=cfg.models.spacy.language,
        gliner_model_name=cfg.models.gliner_primary.model_name_or_path,
        fallback_gliner_model_name=cfg.models.gliner_vanilla_fallback.model_name_or_path,
        spacy_model_name=cfg.models.spacy.model_name,
        fallback_to_cpu=cfg.runtime.fallback_to_cpu,
    )
    load_time_ms = round((time.perf_counter() - t_load) * 1000, 2)
    if args.verbose or args.profile:
        print(f"Model weights loaded into memory cache in {load_time_ms} ms (resident).")

    pipeline = Stage9Pipeline(config=cfg, gliner_threshold=threshold, device=resolved_device)

    if in_path.is_dir():
        files = sorted(list(in_path.glob("*.json")) + list(in_path.glob("*.txt")))
        if not files:
            print(f"No .json or .txt files found in {in_path}", file=sys.stderr)
            return 1
        for f in files:
            process_file(pipeline, f, out_dir, profile=args.profile, enable_profiling=args.enable_profiling)
    else:
        process_file(pipeline, in_path, out_dir, profile=args.profile, enable_profiling=args.enable_profiling)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
