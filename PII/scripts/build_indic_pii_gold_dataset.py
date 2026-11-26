#!/usr/bin/env python3
"""Build a validated, synthetic Indic clinical PII gold dataset.

The seed contains only fictional patient--clinician dialogues.  Reviewers can
append confirmed misses or false positives to the JSONL regression ledger only
after replacing every real identifier with a synthetic equivalent.  Offsets
are derived here instead of being hand-maintained.
"""
from __future__ import annotations

import argparse
import csv
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each regression must be a JSON object")
        records.append(value)
    return records


def _occurrences(text: str, value: str) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        offset = text.find(value, start)
        if offset < 0:
            return offsets
        offsets.append(offset)
        start = offset + len(value)


def build_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Validate records and produce evaluator-compatible rows with exact spans."""
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for record in records:
        row_id = str(record.get("id", "")).strip()
        language = str(record.get("language", "")).strip()
        script = str(record.get("script", "")).strip()
        case_type = str(record.get("case_type", "synthetic_seed")).strip()
        source = str(record.get("source", "synthetic_seed")).strip()
        context = record.get("context")
        entities = record.get("entities")
        if not row_id or row_id in seen_ids:
            raise ValueError(f"record id must be unique and non-empty: {row_id!r}")
        if not language or not script or not isinstance(context, str) or not context:
            raise ValueError(f"{row_id}: language, script, and non-empty context are required")
        if not isinstance(entities, list):
            raise ValueError(f"{row_id}: entities must be a list")

        seen_ids.add(row_id)
        tags: list[dict[str, Any]] = []
        used_spans: set[tuple[int, int]] = set()
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError(f"{row_id}: each entity must be an object")
            entity_type = str(entity.get("entity_type", "")).strip()
            value = str(entity.get("text", ""))
            if not entity_type or not value:
                raise ValueError(f"{row_id}: entity_type and text are required")
            matches = _occurrences(context, value)
            if len(matches) != 1:
                raise ValueError(f"{row_id}: entity {value!r} must occur exactly once, found {len(matches)}")
            start = matches[0]
            span = (start, start + len(value))
            if span in used_spans:
                raise ValueError(f"{row_id}: duplicate annotated span for {value!r}")
            used_spans.add(span)
            tags.append({"entity_type": entity_type, "text": value, "start": start, "end": span[1]})

        rows.append({
            "row_index": row_id,
            "language": language,
            "script": script,
            "case_type": case_type,
            "source": source,
            "context": context,
            "verified_semantic_tags": json.dumps(tags, ensure_ascii=False, separators=(",", ":")),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the synthetic Indic clinical PII gold CSV")
    parser.add_argument("--seed", type=Path, required=True, help="Explicit new synthetic seed; bundled seeds were merged into the unified benchmark")
    parser.add_argument("--regressions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed_payload = json.loads(args.seed.read_text(encoding="utf-8"))
    if not isinstance(seed_payload, dict) or not isinstance(seed_payload.get("records"), list):
        raise ValueError(f"{args.seed}: expected an object with a records list")
    rows = build_rows([*seed_payload["records"], *_read_jsonl(args.regressions)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else ["row_index", "context", "verified_semantic_tags"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["language"] for row in rows)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "by_language": dict(sorted(counts.items()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
