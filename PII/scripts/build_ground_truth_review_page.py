"""Export a general offline annotation page from the unified benchmark or a review CSV."""
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
from pathlib import Path

from pii_pipeline.data_assets import ensure_data

csv.field_size_limit(10**9)
DEFAULT_TEMPLATE = Path(__file__).with_name("ground_truth_review_template.html")


def review_payload(rows):
    """Keep original text and draft suggestions; completion requires human review."""
    payload, seen = [], set()
    for row in rows:
        row_id, context = row.get("row_index", ""), row.get("context", "")
        if not row_id or row_id in seen or not context:
            raise ValueError("Review rows need unique IDs and nonempty original contexts")
        seen.add(row_id)
        fingerprint = hashlib.sha256(context.encode("utf-8")).hexdigest()
        if row.get("source_fingerprint") and row["source_fingerprint"] != fingerprint:
            raise ValueError("Review source fingerprint differs from original text")
        raw = (row.get("chosen_reference_tags") or row.get("proposed_tags")
               or row.get("assistant_semantic_tags") or "[]")
        tags = json.loads(raw)
        if not isinstance(tags, list):
            raise ValueError("Draft suggestions must be a list")
        for tag in tags:
            if not isinstance(tag, dict):
                raise ValueError("A draft suggestion must be an object")
            start, end = tag.get("start"), tag.get("end")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(context)
                    or tag.get("text") != context[start:end]
                    or not isinstance(tag.get("entity_type"), str) or not tag["entity_type"].strip()):
                raise ValueError("Draft suggestion does not match original text and offsets")
        issues = json.loads(row.get("review_issues") or "[]")
        if not isinstance(issues, list):
            raise ValueError("Review issues must be a list")
        payload.append({"row_index": row_id, "context": context,
                        "source_fingerprint": fingerprint, "language": row.get("language", ""),
                        "script": row.get("script", ""), "case_type": row.get("case_type", ""),
                        "source": row.get("reference_origin") or row.get("source", ""),
                        "suggestions": tags, "uncertainties": issues})
    if not payload:
        raise ValueError("No transcripts selected for review")
    return payload


def render_page(payload, template):
    if template.count("__TRANSCRIPT_DATA__") != 1:
        raise ValueError("Template must contain exactly one data marker")
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return template.replace("__TRANSCRIPT_DATA__", serialized)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Default: fetch the pinned unified benchmark")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, required=True, help="New private HTML review workspace")
    parser.add_argument("--group", choices=["real_development_pending", "synthetic_regression", "all"],
                        default="real_development_pending")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file; existing review work must be preserved")
    source = args.input or ensure_data(("benchmark",))["data/gold/unified_benchmark.csv"]
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and "benchmark_group" in rows[0] and args.group != "all":
        rows = [row for row in rows if row["benchmark_group"] == args.group]
    payload = review_payload(rows)
    html = render_page(payload, args.template.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(html)
    print(f"Created private review workspace for {len(payload)} transcripts: {args.output}")


if __name__ == "__main__":
    main()
