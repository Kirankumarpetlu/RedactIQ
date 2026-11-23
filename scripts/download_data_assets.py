"""Download pinned private data into an ignored cache and verify SHA256 hashes."""
from __future__ import annotations

import argparse
from pathlib import Path

from pii_pipeline.data_assets import ensure_data, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=["all", "benchmark", "runtime"],
                        help="Repeat to combine groups; default: benchmark and runtime")
    parser.add_argument("--check-only", action="store_true", help="Verify cached files without network access")
    parser.add_argument("--cache-dir", type=Path, help="Cache base directory (or set PII_DATA_CACHE)")
    printing = parser.add_mutually_exclusive_group()
    printing.add_argument("--print-benchmark-path", action="store_true")
    printing.add_argument("--print-path", metavar="ASSET", help="Print one selected asset's local cache path")
    args = parser.parse_args()
    try:
        paths = ensure_data(args.group or ("benchmark", "runtime"), check_only=args.check_only,
                            cache_root=args.cache_dir)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")
    if args.print_benchmark_path or args.print_path:
        relative = "data/gold/unified_benchmark.csv" if args.print_benchmark_path else args.print_path
        if relative not in paths:
            parser.error("The requested asset is not in the selected data groups")
        print(paths[relative])
    else:
        manifest = load_manifest()
        print(f"Verified {len(paths)} private data assets at {manifest['revision']}.")


if __name__ == "__main__":
    main()
