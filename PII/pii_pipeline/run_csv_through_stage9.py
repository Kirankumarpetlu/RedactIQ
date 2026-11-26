"""
run_csv_through_stage9.py
=========================
Runs CSV, JSON, or TXT input through the Stage 9 PII/PHI Pipeline.

Guarantees:
- 100% LOCAL execution on Apple Silicon GPU (MPS), CUDA, and CPU.
- Zero network/cloud data transfer - no data ever leaves the machine.
- Fuses Gemma semantic tags with Stage 9 (GLiNER + Presidio Medical PHI + IndicNER) entities.
- Redacts all detected PII/PHI using Presidio PlaceholderAnonymizer.
- Supports tqdm progress tracking, batch checkpointing, and centralized YAML configuration.
"""
from __future__ import annotations

import argparse
import csv
from . import json_codec as json
import logging
import os
import time
import hashlib
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from pii_pipeline.pipeline import Stage9Pipeline
from pii_pipeline.config import load_pipeline_config
from pii_pipeline.record_identity import row_identity, validate_row_identities
from pii_pipeline.data_assets import load_manifest as load_data_manifest, require_data_path
from pii_pipeline.model_cache import auto_detect_device, configure_runtime, get_hardware_info
from pii_pipeline.profiling import PipelineProfiler, profile_context
from pii_pipeline.stage_timing_report import write_stage_timing_reports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _write_pipeline_progress(
    output_dir: Path,
    *,
    status: str,
    total_rows: int,
    processed_rows: int,
    latencies_ms: List[float],
    total_start: float,
    device: str,
    hardware: Optional[str],
    gliner_threshold: float,
    min_entity_score: float,
    entity_count: int,
    input_format: str,
) -> None:
    """Atomically publish safe aggregate telemetry for the API metrics endpoint."""
    elapsed_seconds = round(time.perf_counter() - total_start, 2)
    average_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0
    throughput = round(processed_rows / elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0
    progress = {
        "status": status,
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "progress_percent": round((processed_rows / total_rows) * 100, 2) if total_rows else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "average_latency_ms": average_latency_ms,
        "throughput_rows_per_sec": throughput,
        "device": device,
        "hardware": hardware,
        "gliner_threshold": gliner_threshold,
        "min_entity_score": min_entity_score,
        "entity_count": entity_count,
        "input_format": input_format,
    }
    progress_path = output_dir / "pipeline_progress.json"
    temporary_path = progress_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    # On Windows, a metrics reader can briefly hold the destination open and
    # make os.replace fail with WinError 5. Progress publication is telemetry;
    # retry the atomic replace rather than aborting a completed inference row.
    for attempt in range(8):
        try:
            temporary_path.replace(progress_path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (2 ** attempt))


def _normalise_input_record(record: Any) -> Dict[str, Any]:
    """Convert a supported input item into the runner's internal row format."""
    if isinstance(record, str):
        return {"context": record, "semantic_tags": []}
    if isinstance(record, dict):
        normalized = {
            "context": record.get("context", record.get("text", record.get("content", ""))),
            "semantic_tags": record.get("semantic_tags", record.get("gemma_semantic_tags", [])),
        }
        if "row_index" in record or "row_id" in record:
            normalized["row_index"] = row_identity(record, "")
        return normalized
    return {"context": "", "semantic_tags": []}


def _load_input_records(in_path: Path, max_rows: Optional[int], *, benchmark_group: Optional[str] = None) -> tuple[List[Dict[str, Any]], str]:
    """Load CSV rows, JSON records, or non-empty TXT lines without exposing them in metrics."""
    suffix = in_path.suffix.lower()
    if benchmark_group not in {None, "synthetic_regression", "real_development_pending"}:
        raise ValueError("Unknown benchmark group")
    if benchmark_group is not None and suffix != ".csv":
        raise ValueError("Benchmark group selection requires a CSV input")
    if suffix == ".csv":
        with open(in_path, "r", encoding="utf-8-sig") as f_in:
            reader = csv.DictReader(f_in)
            if benchmark_group is not None and "benchmark_group" not in (reader.fieldnames or []):
                raise ValueError("Input CSV has no benchmark_group column")
            selected = (row for row in reader if benchmark_group is None or row.get("benchmark_group") == benchmark_group)
            rows = list(islice(selected, max_rows)) if max_rows is not None else list(selected)
        if benchmark_group is not None and not rows:
            raise ValueError("No input rows match the selected benchmark group")
        return [_normalise_input_record(row) for row in rows], "csv"

    if suffix == ".json":
        try:
            document = json.loads(in_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {in_path}") from exc
        if isinstance(document, dict):
            document = document.get("records", document.get("items", document.get("data", document)))
        records = document if isinstance(document, list) else [document]
        if max_rows is not None:
            records = records[:max_rows]
        return [_normalise_input_record(record) for record in records], "json"

    if suffix == ".txt":
        lines = [line.strip() for line in in_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if max_rows is not None:
            lines = lines[:max_rows]
        return [_normalise_input_record(line) for line in lines], "txt"

    raise ValueError("Unsupported input type. Use a .csv, .json, or .txt file.")


def _inference_code_hashes(code_root: Path, indic_package_root: Path) -> dict[str, str]:
    hashes = {path.relative_to(code_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(code_root.rglob("*.py"))}
    vendor_files = [indic_package_root / "entity_reconstruction.py",
                    *sorted((indic_package_root / "indicner").glob("*.py"))]
    for path in vendor_files:
        if path.is_file():
            key = "indicner_package/" + path.relative_to(indic_package_root).as_posix()
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_pipeline_on_file(
    input_file: Optional[str] = None,
    output_csv: str = "output/pipeline_redacted_output.csv",
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    max_rows: Optional[int] = None,
    enable_profiling: bool = False,
    gliner_threshold: Optional[float] = None,
    min_entity_score: float = 0.0,
    save_diagnostics: bool = False,
    benchmark_group: Optional[str] = None,
    indic_device: Optional[str] = None,
    decider: Any = None,
) -> Dict[str, Any]:
    in_path = Path(input_file) if input_file else require_data_path("data/input/csv/context_and_semantic_tags.csv")
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_pipeline_config(config_path)
    if indic_device is not None:
        cfg.models.indic_ner.device = indic_device
    configure_runtime(cfg.runtime.cpu_threads, cfg.runtime.enable_mps)
    resolved_device_str = device or cfg.runtime.device
    resolved_device = auto_detect_device(resolved_device_str)
    hw = get_hardware_info(resolved_device)
    threshold = gliner_threshold if gliner_threshold is not None else cfg.models.gliner_primary.threshold

    print("\n" + "=" * 65)
    hosted = decider is not None and decider.settings.get("synthetic_only", True)
    print(" STAGE 9 PIPELINE EXECUTION" + (" (HOSTED, SYNTHETIC ONLY)" if hosted else " (100% LOCAL & PRIVATE)"))
    print("=" * 65)
    print(f" Input File        : {in_path.resolve()}")
    print(f" Output CSV        : {out_path.resolve()}")
    print(f" Compute Device    : {resolved_device.upper()} ({hw.get('current_gpu_name')})")
    if decider is not None:
        threshold = decider.settings["acceptance_threshold"]
        print(f" Decider Model     : {decider.settings['model']}")
    else:
        print(f" GLiNER Model      : {cfg.models.gliner_primary.model_name_or_path}")
    print(f" Threshold         : {threshold}")
    if not hosted:
        print(" Data Security     : STRICT LOCAL (Zero bytes leave device)")
    print("=" * 65 + "\n")

    reader, input_format = _load_input_records(in_path, max_rows, benchmark_group=benchmark_group)
    validate_row_identities(reader)
    if decider is not None:
        for record in reader:
            decider.verify_input(record["context"])
    total_rows = len(reader)
    code_root = Path(__file__).resolve().parent
    provenance = {
        "input_sha256":hashlib.sha256(in_path.read_bytes()).hexdigest(),
        "config":asdict(cfg), "gliner_threshold":threshold,"min_entity_score":min_entity_score,
        "requested_device":resolved_device_str,
        "benchmark_group": benchmark_group,
        "code_sha256":_inference_code_hashes(code_root, Path(cfg.models.indic_ner.model_path)),
        "recognizer_resources_sha256":{Path(relative).name:hashlib.sha256(require_data_path(relative).read_bytes()).hexdigest()
                                       for relative in load_data_manifest()['groups']['runtime']},
        "model_metadata_sha256":{},
    }
    if decider is not None:
        provenance["decider"] = decider.settings
    model_paths = () if decider is not None else (cfg.models.gliner_primary.model_name_or_path, cfg.models.indic_ner.model_path)
    for model_path in model_paths:
        model_root=Path(model_path)
        if model_root.is_dir():
            for filename in ("config.json","manifest.json","model_manifest.json","provisioning_manifest.json"):
                file=model_root/filename
                if file.is_file():
                    provenance["model_metadata_sha256"][str(file.resolve())]=hashlib.sha256(file.read_bytes()).hexdigest()
    run_manifest_path=out_path.with_suffix(out_path.suffix+".run_manifest.json")
    run_key=hashlib.sha256(json.dumps(provenance,sort_keys=True,default=str).encode("utf-8")).hexdigest()
    if out_path.is_file() and out_path.stat().st_size>0:
        if not run_manifest_path.is_file() or json.loads(run_manifest_path.read_text(encoding="utf-8")).get("run_key")!=run_key:
            raise ValueError("Existing output has different or missing run provenance; choose a fresh output path")
    run_manifest_path.write_text(json.dumps({**provenance,"run_key":run_key,
        "created_utc":datetime.now(timezone.utc).isoformat(),"weights_hash_note":"Model metadata hashes are recorded; immutable provisioned weights are required."},indent=2,default=str),encoding="utf-8")
    diagnostics_path=out_path.with_suffix(".diagnostics.jsonl")
    logger.info(f"Loaded {total_rows} {input_format.upper()} item(s) from {in_path.name}.")

    # Publish immediately so the API shows progress while models are loading.
    total_start = time.perf_counter()
    _write_pipeline_progress(
        out_path.parent,
        status="initializing",
        total_rows=total_rows,
        processed_rows=0,
        latencies_ms=[],
        total_start=total_start,
        device=resolved_device,
        hardware=hw.get("current_gpu_name"),
        gliner_threshold=threshold,
        min_entity_score=min_entity_score,
        entity_count=0,
        input_format=input_format,
    )

    # Initialize Stage 9 Pipeline
    logger.info("Initializing Stage 9 Pipeline (%s)...",
                f"{decider.settings['model']} + Presidio" if decider is not None else "GLiNER + Presidio Medical PHI + IndicNER")
    pipeline = Stage9Pipeline(
        config=cfg,
        device=resolved_device,
        gliner_threshold=threshold,
        **({"decider": decider} if decider is not None else {}),
    )
    initialization_ms = round((time.perf_counter() - total_start) * 1000, 2)
    # An accelerator may have been detected but rejected by a model; record
    # the effective fallback device from this point onward.
    resolved_device = pipeline.device
    hw = get_hardware_info(resolved_device)
    fieldnames = [
        "row_index",
        "context",
        "gemma_semantic_tags",
        "pipeline_entities",
        "total_entities_count",
        "redacted_text",
        "latency_ms"
    ]

    checkpoints_dir = out_path.parent / "stage9_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    existing_row_indices = set()
    write_header = True
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8-sig") as f_exist:
            exist_reader = csv.DictReader(f_exist)
            for r in exist_reader:
                existing_row_indices.add(r["row_index"])
        if existing_row_indices:
            logger.info(f"Resuming from output file: {len(existing_row_indices)} rows already processed.")
            write_header = False

    results: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []
    row_profiles: List[Dict[str, Any]] = []
    total_entity_count = 0
    decider_totals = dict.fromkeys(("decider_calls", "decider_cache_hits", "decider_failed"), 0)

    _write_pipeline_progress(
        out_path.parent,
        status="running",
        total_rows=total_rows,
        processed_rows=len(existing_row_indices),
        latencies_ms=latencies_ms,
        total_start=total_start,
        device=resolved_device,
        hardware=hw.get("current_gpu_name"),
        gliner_threshold=threshold,
        min_entity_score=min_entity_score,
        entity_count=total_entity_count,
        input_format=input_format,
    )

    with open(out_path, "a" if not write_header else "w", newline="", encoding="utf-8-sig") as f_out, \
            (diagnostics_path.open("a" if not write_header else "w",encoding="utf-8") if save_diagnostics else nullcontext()) as f_diag:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        pbar = tqdm(reader, total=total_rows, desc="Processing Stage 9 Rows", unit="row")
        for i, row in enumerate(pbar, start=1):
            row_id = row_identity(row, i)
            if row_id in existing_row_indices:
                continue

            text = row.get("context", "")
            raw_tags = row.get("semantic_tags", "[]")

            try:
                gemma_tags = json.loads(raw_tags) if isinstance(raw_tags, str) else (raw_tags or [])
            except Exception:
                gemma_tags = []

            t0 = time.perf_counter()
            fusion_profiler = PipelineProfiler() if enable_profiling else None

            # Fuse and redact once through the same policy used by API/UI.
            diagnostic_kwargs={"include_diagnostics":True} if save_diagnostics else {}
            res = pipeline.run(text, enable_profiling=enable_profiling,
                               semantic_tags=raw_tags, min_entity_score=min_entity_score,**diagnostic_kwargs)
            final_redacted_text = res["redacted_text"]
            if decider is not None:
                for key in decider_totals:
                    decider_totals[key] += res["routing_summary"][key]
            all_detected_list = res["entities_detected"]
            if res.get("entity_rejections"):
                logger.debug("Row %s candidate rejections: %s", i, res["entity_rejections"])
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            latencies_ms.append(latency_ms)
            if enable_profiling:
                row_profiles.append({"row_index": row_id, "latency_ms": latency_ms, "pipeline": res.get("profiling"),
                                     "model_status": res.get("model_status"), "routing_summary": res.get("routing_summary", {})})

            out_row = {
                "row_index": row_id,
                "context": text,
                "gemma_semantic_tags": json.dumps(gemma_tags, ensure_ascii=False),
                "pipeline_entities": json.dumps(all_detected_list, ensure_ascii=False),
                "total_entities_count": len(all_detected_list),
                "redacted_text": final_redacted_text,
                "latency_ms": latency_ms
            }

            writer.writerow(out_row)
            f_out.flush()
            if f_diag is not None:
                f_diag.write(json.dumps({"row_index":row_id,"source_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "run_key":run_key,"latency_ms":latency_ms,"routing_summary":res.get("routing_summary",{}),
                    "decision_trace":res.get("decision_trace",{}),"entity_rejections":res.get("entity_rejections",[])},ensure_ascii=False,default=str)+"\n")
                f_diag.flush()
            results.append(out_row)
            total_entity_count += len(all_detected_list)

            _write_pipeline_progress(
                out_path.parent,
                status="running",
                total_rows=total_rows,
                processed_rows=len(existing_row_indices) + len(results),
                latencies_ms=latencies_ms,
                total_start=total_start,
                device=resolved_device,
                hardware=hw.get("current_gpu_name"),
                gliner_threshold=threshold,
                min_entity_score=min_entity_score,
                entity_count=total_entity_count,
                input_format=input_format,
            )

            # Checkpoint every 1000 rows
            processed_so_far = len(existing_row_indices) + len(results)
            if processed_so_far % 1000 == 0 or processed_so_far == total_rows:
                ckpt_csv = checkpoints_dir / f"pipeline_checkpoint_row_{processed_so_far:04d}.csv"
                with open(ckpt_csv, "w", newline="", encoding="utf-8") as f_ckpt:
                    ckpt_writer = csv.DictWriter(f_ckpt, fieldnames=fieldnames)
                    ckpt_writer.writeheader()
                    ckpt_writer.writerows(results)
                logger.info(f"🚩 Checkpoint saved at row {processed_so_far} -> {ckpt_csv.name}")

            # Update tqdm postfix
            avg_l = sum(latencies_ms[-10:]) / len(latencies_ms[-10:]) if latencies_ms else 0.0
            pbar.set_postfix({"avg_ms": f"{avg_l:.1f}", "entities": len(all_detected_list)})

    total_time_sec = round(time.perf_counter() - total_start, 2)
    avg_latency_ms = round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0
    throughput = round(len(results) / total_time_sec, 2) if total_time_sec > 0 else 0.0
    # Per-row profiling excludes CSV writes, checkpoints and progress updates.
    # Report their combined run overhead separately from model initialization.
    other_run_overhead_ms = round(max(0.0, total_time_sec * 1000 - initialization_ms
                                      - sum(latencies_ms)), 2)

    summary = {
        "total_rows_processed": len(results),
        "total_time_seconds": total_time_sec,
        "average_latency_ms": avg_latency_ms,
        "throughput_rows_per_sec": throughput,
        "pipeline_initialization_ms": initialization_ms,
        "other_run_overhead_ms": other_run_overhead_ms,
        "device": resolved_device,
        "hardware": hw.get("current_gpu_name"),
        "gliner_threshold": threshold,
        "min_entity_score": min_entity_score,
        "input_file": str(in_path.resolve()),
        "input_format": input_format,
        "output_csv": str(out_path.resolve())
    }
    summary.update(config=asdict(cfg),inference_backends=getattr(pipeline,"inference_backends",{}),
                   run_manifest=str(run_manifest_path.resolve()),run_key=run_key,
                   diagnostics_file=str(diagnostics_path.resolve()) if save_diagnostics else None)
    if decider is not None:
        summary["decider"] = {**decider.settings, **decider_totals}

    if enable_profiling:
        (out_path.parent / "stage_metrics.json").write_text(json.dumps(row_profiles, indent=2), encoding="utf-8")
        call_path, stage_summary_path = write_stage_timing_reports(row_profiles, out_path.parent)
        summary["stage_timings_per_row_csv"] = str(call_path.resolve())
        summary["stage_latency_summary_csv"] = str(stage_summary_path.resolve())

    summary_json_path = out_path.parent / "pipeline_run_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    _write_pipeline_progress(
        out_path.parent,
        status="completed",
        total_rows=total_rows,
        processed_rows=len(existing_row_indices) + len(results),
        latencies_ms=latencies_ms,
        total_start=total_start,
        device=resolved_device,
        hardware=hw.get("current_gpu_name"),
        gliner_threshold=threshold,
        min_entity_score=min_entity_score,
        entity_count=total_entity_count,
        input_format=input_format,
    )

    print("\n" + "=" * 65)
    print(" STAGE 9 PIPELINE EXECUTION COMPLETE!")
    print("=" * 65)
    print(f" Total Rows Processed : {len(results)}")
    print(f" Execution Time       : {total_time_sec} sec")
    print(f" Average Latency      : {avg_latency_ms} ms / row")
    print(f" Throughput           : {throughput} rows / sec")
    print(f" Output CSV           : {out_path.resolve()}")
    print(f" Summary JSON         : {summary_json_path.resolve()}")
    if enable_profiling:
        print(f" Stage Calls CSV      : {call_path.resolve()}")
        print(f" Stage Summary CSV    : {stage_summary_path.resolve()}")
    print("=" * 65 + "\n")

    return summary


def run_pipeline_on_csv(
    input_csv: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Backward-compatible CSV entry point; use :func:`run_pipeline_on_file` for any input."""
    return run_pipeline_on_file(input_file=input_csv, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CSV, JSON, or TXT input through the Stage 9 Pipeline.")
    parser.add_argument("--config", default=None, help="Path to pipeline YAML/JSON configuration file")
    parser.add_argument("--input", default=None, help="Input .csv, .json, or .txt file; default: cached private corpus")
    parser.add_argument("--output", default="output/pipeline_redacted_output.csv", help="Output CSV file")
    parser.add_argument("--device", default=None, help="Compute device ('cuda:0', 'mps', 'cpu', 'auto')")
    parser.add_argument("--indic-device", choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
                        help="Override IndicNER placement; cpu leaves GPU memory for GLiNER")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of rows to process")
    parser.add_argument("--benchmark-group", choices=["synthetic_regression", "real_development_pending"],
                        help="Select a unified CSV benchmark group before applying max-rows")
    parser.add_argument("--enable-profiling", action="store_true")
    parser.add_argument("--save-diagnostics",action="store_true",help="Save local candidate/gate/overlap traces and run provenance")
    parser.add_argument("--gliner-threshold", type=float, default=None, help="Minimum GLiNER confidence")
    parser.add_argument("--min-entity-score", type=float, default=0.0, help="Minimum merged entity confidence")
    args = parser.parse_args()

    run_pipeline_on_file(
        input_file=args.input,
        output_csv=args.output,
        config_path=args.config,
        device=args.device,
        max_rows=args.max_rows,
        enable_profiling=args.enable_profiling,
        gliner_threshold=args.gliner_threshold,
        min_entity_score=args.min_entity_score,
        save_diagnostics=args.save_diagnostics,
        benchmark_group=args.benchmark_group,
        indic_device=args.indic_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
