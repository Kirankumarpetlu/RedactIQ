"""Export every profiled stage call and an inclusive latency summary.

The profiler measures nested inclusive wall time. Stage totals therefore
overlap and must not be added to obtain end-to-end latency.
"""
from __future__ import annotations

import argparse
import csv
from . import json_codec as json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_CALL_FIELDS = ("row_index", "stage_name", "depth", "call_index", "duration_ms", "status")
_SUMMARY_FIELDS = ("stage_name", "depth", "calls", "rows", "total_ms", "mean_ms",
                   "median_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms")


def _percentile(sorted_values: list[float], fraction: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int((len(sorted_values) - 1) * fraction))]


def stage_timing_rows(row_profiles: Iterable[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Return one row per stage call plus grouped summaries by name and depth."""
    calls: list[dict] = []
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in row_profiles:
        row_index = row["row_index"]
        stage_counts: dict[tuple[str, int], int] = defaultdict(int)
        for stage in (row.get("pipeline") or {}).get("stages", []):
            key = (str(stage["stage_name"]), int(stage["depth"]))
            stage_counts[key] += 1
            call = {
                "row_index": row_index,
                "stage_name": key[0],
                "depth": key[1],
                "call_index": stage_counts[key],
                "duration_ms": round(float(stage["duration_ms"]), 4),
                "status": stage.get("status", "ok"),
            }
            calls.append(call)
            grouped[key].append(call)

    summaries: list[dict] = []
    for (name, depth), items in grouped.items():
        durations = sorted(item["duration_ms"] for item in items)
        summaries.append({
            "stage_name": name,
            "depth": depth,
            "calls": len(items),
            "rows": len({item["row_index"] for item in items}),
            "total_ms": round(sum(durations), 2),
            "mean_ms": round(statistics.mean(durations), 2),
            "median_ms": round(statistics.median(durations), 2),
            "p90_ms": round(_percentile(durations, .90), 2),
            "p95_ms": round(_percentile(durations, .95), 2),
            "p99_ms": round(_percentile(durations, .99), 2),
            "max_ms": round(durations[-1], 2),
        })
    summaries.sort(key=lambda item: (-item["total_ms"], item["stage_name"], item["depth"]))
    return calls, summaries


def write_stage_timing_reports(row_profiles: Iterable[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write complete per-call and per-stage CSV reports beside stage_metrics.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    calls, summaries = stage_timing_rows(row_profiles)
    call_path = output_dir / "stage_timings_per_row.csv"
    summary_path = output_dir / "stage_latency_summary.csv"
    for path, fields, values in ((call_path, _CALL_FIELDS, calls),
                                 (summary_path, _SUMMARY_FIELDS, summaries)):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    return call_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="stage_metrics.json")
    parser.add_argument("--output-dir", type=Path, help="defaults to the input directory")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as stream:
        profiles = json.load(stream)
    calls, summaries = write_stage_timing_reports(profiles, args.output_dir or args.input.parent)
    print(f"Exported {len(profiles)} rows and {len(stage_timing_rows(profiles)[0])} stage calls")
    print(f"Per-call timings: {calls}")
    print(f"Per-stage summary: {summaries}")
    print("Timings are inclusive; nested stages overlap and must not be summed.")


if __name__ == "__main__":
    main()
