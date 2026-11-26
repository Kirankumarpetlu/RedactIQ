"""Turn a prediction-copied real-transcript sheet back into a review draft.

This does not annotate PII or certify any row. The original bytes are backed
up before replacing the source file; all model tags remain in predicted_tags.
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
from pathlib import Path

from scripts.prepare_precision_dataset import benchmark_draft, fingerprint
from scripts.score_against_benchmark import normalise

csv.field_size_limit(10**9)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_pending(rows: list[dict[str, str]], legacy_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    if not rows or not {"row_index", "context", "verified_semantic_tags", "predicted_tags"} <= rows[0].keys():
        raise ValueError("Expected a real-transcript labelling sheet with predictions and verified tags")
    if any(row.get("review_status") == "reviewed" for row in rows):
        raise ValueError("Refusing to clear a row already marked reviewed")
    if any(row["verified_semantic_tags"].strip() and
           json.loads(row["verified_semantic_tags"]) != json.loads(row["predicted_tags"] or "[]") for row in rows):
        raise ValueError("Refusing to clear annotations that differ from model predictions")
    legacy = {normalise(row["context"]): row for row in legacy_rows}
    if len(legacy) != len(legacy_rows):
        raise ValueError("Duplicate legacy reference contexts")
    ids = [row["row_index"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate review sheet row IDs")

    prepared = []
    legacy_matches = suggestions = 0
    for row in rows:
        source = legacy.get(normalise(row["context"]))
        proposed, issues = benchmark_draft(source) if source else ([], [])
        legacy_matches += source is not None
        suggestions += len(proposed)
        prepared.append({
            **row,
            "verified_semantic_tags": "",
            "source_fingerprint": fingerprint(row["context"]),
            "legacy_reference_row_id": source.get("row_id", "") if source else "",
            "legacy_reference_suggestions": json.dumps(proposed, ensure_ascii=False),
            "review_status": "pending",
            "annotation_scope": "",
            "reviewer": "",
            "review_notes": "Unreviewed model and legacy suggestions; inspect the entire transcript. " + "; ".join(issues),
        })
    return prepared, {"rows": len(rows), "legacy_matching_rows": legacy_matches,
                      "legacy_suggested_spans": suggestions,
                      "model_suggested_spans": sum(len(json.loads(row["predicted_tags"] or "[]")) for row in rows),
                      "review_status": "pending"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args()
    original = args.input.read_bytes()
    repaired, summary = prepare_pending(read_rows(args.input), read_rows(args.legacy))
    if args.backup.exists():
        raise FileExistsError(f"Backup already exists: {args.backup}")
    args.backup.parent.mkdir(parents=True, exist_ok=True)
    args.backup.write_bytes(original)
    fields = list(repaired[0])
    temporary = args.input.with_suffix(args.input.suffix + ".repair_tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(repaired)
        os.replace(temporary, args.input)
    finally:
        temporary.unlink(missing_ok=True)
    summary.update(backup=str(args.backup), original_sha256=hashlib.sha256(original).hexdigest(),
                   repaired_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
