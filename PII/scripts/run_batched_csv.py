#!/usr/bin/env python3
"""Chunked CSV runner for Stage 9 pipeline.

Rows are checkpointed in groups, but model inference remains sequential until a
parity-tested backend batch implementation is available.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Permit documented direct execution (``python scripts/run_batched_csv.py``)
# without requiring callers to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pii_pipeline import json_codec as json
from pii_pipeline.pipeline import Stage9Pipeline
from pii_pipeline.record_identity import row_identity, validate_row_identities
from pii_pipeline.model_cache import auto_detect_device, get_hardware_info
from pii_pipeline.data_assets import require_data_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BatchedPipelineRunner:
    """Run sequential inference while checkpointing CSV rows in groups."""
    
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 8,
        max_workers: int = 4,
        gliner_threshold: Optional[float] = None,
        gliner_window: int = 512,
        gliner_stride: int = 384,
    ):
        self.device = auto_detect_device(device)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        logger.info(f"Initializing batched pipeline: batch_size={batch_size}, workers={max_workers}")
        
        # Initialize single pipeline (models loaded once)
        self.pipeline = Stage9Pipeline(
            device=self.device,
            gliner_threshold=gliner_threshold,
            gliner_window=gliner_window,
            gliner_stride=gliner_stride,
        )
    def process_batch(self, texts: List[str], gemma_tags_list: List[List]) -> List[Dict]:
        """Process a batch of texts sequentially (for now)."""
        results = []
        
        for text, gemma_tags in zip(texts, gemma_tags_list):
            if not text:
                results.append({
                    "entities": [],
                    "redacted_text": "",
                    "latency_ms": 0,
                })
                continue
            
            t0 = time.perf_counter()
            
            res = self.pipeline.run(text, enable_profiling=False, semantic_tags=gemma_tags)
            redacted = res["redacted_text"]
            all_detected = res["entities_detected"]
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            results.append({
                "entities": all_detected,
                "redacted_text": redacted,
                "latency_ms": latency_ms,
            })
        
        return results
    
    def run_csv(
        self,
        input_csv: str,
        output_csv: str,
        max_rows: Optional[int] = None,
        enable_profiling: bool = False,
    ) -> Dict[str, Any]:
        """Run pipeline on CSV with batching."""
        in_path = Path(input_csv)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(in_path, "r", encoding="utf-8-sig") as f:
            reader = list(islice(csv.DictReader(f), max_rows)) if max_rows else list(csv.DictReader(f))
        validate_row_identities(reader, start=0)
        
        total_rows = len(reader)
        logger.info(f"Loaded {total_rows} rows")
        
        fieldnames = [
            "row_index", "context", "gemma_semantic_tags",
            "pipeline_entities", "total_entities_count",
            "redacted_text", "latency_ms"
        ]
        
        checkpoints_dir = out_path.parent / "batch_checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume support
        existing = set()
        write_header = True
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "row_index" in row:
                        existing.add(row["row_index"])
            if existing:
                write_header = False
                logger.info(f"Resuming: {len(existing)} rows already processed")
        
        all_results = []
        latencies = []
        
        with open(out_path, "a" if not write_header else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            
            # Process in batches
            for batch_start in tqdm(range(0, total_rows, self.batch_size), desc="Batched Pipeline"):
                batch_end = min(batch_start + self.batch_size, total_rows)
                batch_rows = reader[batch_start:batch_end]
                
                # Filter already processed
                batch_to_process = [(i, row) for i, row in enumerate(batch_rows, batch_start) 
                                   if row_identity(row, i) not in existing]
                
                if not batch_to_process:
                    continue
                
                texts = [row.get("context", "") for _, row in batch_to_process]
                gemma_tags_list = []
                for _, row in batch_to_process:
                    gemma_tags_list.append(row.get("semantic_tags") or "[]")
                
                # Process batch
                batch_results = self.process_batch(texts, gemma_tags_list)
                
                # Write results
                for (orig_idx, row), res in zip(batch_to_process, batch_results):
                    all_detected = res["entities"]
                    
                    out_row = {
                        "row_index": row_identity(row, orig_idx),
                        "context": row.get("context", ""),
                        "gemma_semantic_tags": row.get("semantic_tags") or "[]",
                        "pipeline_entities": json.dumps(all_detected, ensure_ascii=False),
                        "total_entities_count": len(all_detected),
                        "redacted_text": res["redacted_text"],
                        "latency_ms": res["latency_ms"],
                    }
                    
                    writer.writerow(out_row)
                    all_results.append(out_row)
                    latencies.append(res["latency_ms"])
                
                f.flush()
                
                # Checkpoint
                if (len(all_results) + len(existing)) % 1000 == 0:
                    ckpt = checkpoints_dir / f"checkpoint_row_{len(all_results) + len(existing):04d}.csv"
                    with open(ckpt, "w", newline="", encoding="utf-8") as ckpt_f:
                        ckpt_w = csv.DictWriter(ckpt_f, fieldnames=fieldnames)
                        ckpt_w.writeheader()
                        ckpt_w.writerows(all_results)
        
        total_time = sum(latencies) / 1000 if latencies else 0
        
        return {
            "total_rows_processed": len(all_results),
            "total_time_seconds": total_time,
            "average_latency_ms": np.mean(latencies) if latencies else 0,
            "throughput_rows_per_sec": len(all_results) / total_time if total_time > 0 else 0,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }


def main():
    parser = argparse.ArgumentParser(description="Batched CSV Pipeline Runner")
    parser.add_argument("--input", help="Default: cached unified benchmark")
    parser.add_argument("--output", default="output/batched/predictions.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    input_csv = args.input or str(require_data_path("data/gold/unified_benchmark.csv"))
    
    runner = BatchedPipelineRunner(
        device=args.device,
        batch_size=args.batch_size,
        max_workers=args.workers,
    )
    
    summary = runner.run_csv(
        input_csv=input_csv,
        output_csv=args.output,
        max_rows=args.max_rows,
        enable_profiling=args.profile,
    )
    
    print("\n" + "=" * 60)
    print(" BATCHED PIPELINE COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
