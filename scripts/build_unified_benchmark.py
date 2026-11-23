"""Validate or rebuild the unified benchmark from its embedded source records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
from pathlib import Path

from scripts.build_indic_pii_gold_dataset import build_rows
from scripts.evaluate_pipeline import normalize_entity_type
from pii_pipeline.data_assets import ensure_data, require_data_path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(100_000_000)

BENCHMARK = "data/gold/unified_benchmark.csv"
PRIORITY = {"synthetic_csv": 0, "synthetic_seed": 1, "synthetic_regressions": 2,
            "real_draft": 3, "review_sheet": 4, "legacy": 5}
FIELDS = ["row_index", "context", "benchmark_group", "reference_origin",
          "chosen_reference_tags", "verified_semantic_tags", "review_status",
          "annotation_scope", "reviewer", "language", "script", "case_type",
          "source_fingerprint", "normalized_context_fingerprint",
          "annotation_conflict", "annotation_variants", "source_records",
          "review_issues", "independent_test_set"]


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_key(text):
    # NBSP replacement preserves character positions; never collapse whitespace.
    return text.replace("\u00a0", " ")


def validate_tags(context, tags):
    if not isinstance(tags, list):
        raise ValueError("Annotations must be a JSON list")
    seen = set()
    for tag in tags:
        if not isinstance(tag, dict):
            raise ValueError("An annotation must be an object")
        start, end = tag.get("start"), tag.get("end")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(context)
                or tag.get("text") != context[start:end]
                or not isinstance(tag.get("entity_type"), str)
                or not tag["entity_type"].strip()):
            raise ValueError("An annotation does not match its original context")
        key = (normalize_entity_type(tag["entity_type"]), start, end)
        if key in seen:
            raise ValueError("Duplicate annotation within a source")
        seen.add(key)
    return tags


def source_annotations(kind, row):
    issues = []
    if kind in {"synthetic_csv", "synthetic_seed", "synthetic_regressions", "real_draft"}:
        raw = row.get("verified_semantic_tags", "")
        if not raw.strip():
            raise ValueError("Source reference labels are missing")
        tags = json.loads(raw)
    elif kind == "review_sheet":
        # A blank verified field is not a reviewed negative.
        raw = row.get("verified_semantic_tags", "").strip()
        if raw and row.get("review_status") == "reviewed":
            tags = json.loads(raw)
        else:
            raw = row.get("proposed_tags", "").strip() or row.get("assistant_semantic_tags", "").strip()
            tags = json.loads(raw) if raw else []
            issues.append("Worksheet labels are proposals; verified labels were not promoted")
    elif kind == "legacy":
        tags = []
        context = row["context"]
        normalized = context_key(context)
        dictionary = json.loads(row.get("pii_dictionary") or "{}")
        for label, values in dictionary.items():
            if not isinstance(values, list):
                raise ValueError("Legacy dictionary values must be lists")
            for value in values:
                if not isinstance(value, str) or not value:
                    raise ValueError("Legacy entity text must be nonempty")
                starts, offset = [], 0
                while True:
                    start = normalized.find(context_key(value), offset)
                    if start < 0:
                        break
                    starts.append(start)
                    offset = start + 1
                if len(starts) != 1:
                    issues.append(f"Legacy {label}: {len(starts)} possible occurrences require review")
                for start in starts:
                    end = start + len(value)
                    tags.append({"entity_type": label, "start": start, "end": end,
                                 "text": context[start:end]})
    else:
        raise ValueError("Unknown source kind")
    return validate_tags(row["context"], tags), issues


def restore_entries(rows, inventory):
    """Reconstruct all original records in memory; historical files are unnecessary."""
    sources = {source["path"]: source for source in inventory}
    entries, seen = [], set()
    for master in rows:
        context = master["context"]
        if sha(context) != master["source_fingerprint"]:
            raise ValueError("Unified context fingerprint does not match")
        records = json.loads(master["source_records"])
        if not isinstance(records, list) or not records:
            raise ValueError("Unified row has no embedded source records")
        for record in records:
            relative, kind, number = record["path"], record["kind"], record["record_number"]
            if (relative not in sources or sources[relative]["kind"] != kind
                    or type(number) is not int or number < 1 or (relative, number) in seen):
                raise ValueError("Invalid or duplicate embedded source identity")
            seen.add((relative, number))
            restored = list(context)
            positions = set()
            for edit in record["context_edits"]:
                position, character = edit["position"], edit["character"]
                if (type(position) is not int or not 0 <= position < len(restored)
                        or position in positions or not isinstance(character, str) or len(character) != 1):
                    raise ValueError("Invalid embedded context edit")
                positions.add(position)
                restored[position] = character
            original_context = "".join(restored)
            if (sha(original_context) != record["original_context_sha256"]
                    or context_key(original_context) != context_key(context)):
                raise ValueError("Embedded source context does not match its fingerprint")
            if "context" in record["original_fields"]:
                raise ValueError("Embedded source fields cannot override context")
            original = {**record["original_fields"], "context": original_context}
            row = build_rows([original])[0] if kind in {"synthetic_seed", "synthetic_regressions"} else original.copy()
            tags, issues = source_annotations(kind, row)
            entries.append({"path": relative, "kind": kind, "record_number": number,
                            "row": row, "original": original, "tags": tags, "issues": issues})
    expected = {(source["path"], number) for source in inventory for number in range(1, source["records"] + 1)}
    if seen != expected:
        raise ValueError("Unified benchmark does not contain every inventoried source record")
    return entries


def load_sources(root):
    root = Path(root).resolve()
    path = require_data_path(BENCHMARK) if root == ROOT else root / BENCHMARK
    metadata = require_data_path(BENCHMARK + ".manifest.json") if root == ROOT else path.with_suffix(".csv.manifest.json")
    raw = path.read_bytes()
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Unified benchmark does not match its manifest checksum")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    if len(rows) != manifest["records"]:
        raise ValueError("Unified benchmark row count differs from its manifest")
    return restore_entries(rows, manifest["sources"]), manifest["sources"]


def merge_entries(entries):
    groups = {}
    for entry in entries:
        text = entry["row"]["context"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Source context is empty")
        groups.setdefault(context_key(text), []).append(entry)
    result = []
    for normalized, members in groups.items():
        members = sorted(members, key=lambda m: (PRIORITY[m["kind"]], m["path"], m["record_number"]))
        chosen = members[0]
        context = chosen["row"]["context"]
        synthetic = chosen["kind"].startswith("synthetic_")
        if any(m["kind"].startswith("synthetic_") != synthetic for m in members):
            raise ValueError("Synthetic and real source contexts collide; adjudication required")
        provenance, variants, issues = [], [], []
        signatures = set()
        for member in members:
            source_context = member["row"]["context"]
            assert len(source_context) == len(context)
            edits = [{"position": i, "character": old}
                     for i, (old, preferred) in enumerate(zip(source_context, context)) if old != preferred]
            restored = list(context)
            for edit in edits:
                restored[edit["position"]] = edit["character"]
            assert "".join(restored) == source_context
            source_fields = {key: value for key, value in member["original"].items() if key != "context"}
            provenance.append({"path": member["path"], "kind": member["kind"],
                               "record_number": member["record_number"],
                               "original_context_sha256": sha(source_context),
                               "context_edits": edits, "original_fields": source_fields})
            tags = []
            for original_tag in member["tags"]:
                tag = dict(original_tag)
                assert context_key(original_tag["text"]) == context_key(context[tag["start"]:tag["end"]])
                tag["text"] = context[tag["start"]:tag["end"]]
                tags.append(tag)
            validate_tags(context, tags)
            signatures.add(tuple(sorted((normalize_entity_type(t["entity_type"]), t["start"], t["end"]) for t in tags)))
            variants.append({"path": member["path"], "kind": member["kind"],
                             "record_number": member["record_number"], "tags": tags,
                             "review_status": member["row"].get("review_status", "synthetic_generated" if synthetic else "legacy_unreviewed")})
            issues.extend(member["issues"])
        conflict = len(signatures) > 1
        if conflict:
            issues.append("Source annotations differ; all variants are retained for review")
        if not synthetic:
            issues.append("Pending/incomplete development annotations; no human-verified accuracy claim")
            if not chosen["tags"]:
                issues.append("Empty draft annotations are not confirmed no-PII negatives")
        preferred_tags = variants[0]["tags"]
        def metadata(name):
            return next((m["row"].get(name, "") for m in members if m["row"].get(name, "")), "")
        result.append({
            "row_index": "benchmark-" + sha(normalized)[:20], "context": context,
            "benchmark_group": "synthetic_regression" if synthetic else "real_development_pending",
            "reference_origin": chosen["path"], "chosen_reference_tags": compact(preferred_tags),
            "verified_semantic_tags": compact(preferred_tags) if synthetic else "",
            "review_status": "synthetic_reference" if synthetic else "pending",
            "annotation_scope": "synthetic_exhaustive" if synthetic else "incomplete",
            "reviewer": "", "language": metadata("language"), "script": metadata("script"),
            "case_type": metadata("case_type"), "source_fingerprint": sha(context),
            "normalized_context_fingerprint": sha(normalized),
            "annotation_conflict": str(conflict).lower(), "annotation_variants": compact(variants),
            "source_records": compact(provenance), "review_issues": compact(sorted(set(issues))),
            "independent_test_set": "false",
        })
    # The first 28 rows of the current file form the quick synthetic regression set.
    result.sort(key=lambda row: (row["benchmark_group"] != "synthetic_regression",
                                 row["reference_origin"],
                                 json.loads(row["source_records"])[0]["record_number"]))
    assert len({row["row_index"] for row in result}) == len(result)
    return result


def build_benchmark(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Choose a new output path inside the project")
    entries, sources = load_sources(root)
    rows = merge_entries(entries)
    assert sum(len(json.loads(row["source_records"])) for row in rows) == len(entries)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    manifest = {
        "schema_version": 1, "mode": "unified_benchmark_draft", "dataset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "records": len(rows), "source_records_preserved": len(entries), "sources": sources,
        "groups": {group: {"rows": sum(r["benchmark_group"] == group for r in rows),
                           "chosen_spans": sum(len(json.loads(r["chosen_reference_tags"])) for r in rows if r["benchmark_group"] == group)}
                   for group in sorted({r["benchmark_group"] for r in rows})},
        "rows_with_annotation_conflicts": sum(row["annotation_conflict"] == "true" for row in rows),
        "deduplication": "Exact original context after replacing only NBSP with ASCII space; chosen original context is preserved; source edits are reversible",
        "source_priority": list(PRIORITY),
        "limitations": ["Real references are pending/incomplete development drafts, not human-verified held-out truth.",
                        "Synthetic and real provisional scores must be reported separately.",
                        "No semantic_tags or gemma_semantic_tags input column is emitted; reference labels are not fed into inference.",
                        "Raw production inputs, model-generated production predictions, gazetteers and audit logs are not merged as ground truth."],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional explicit new file; default validates without creating files")
    args = parser.parse_args()
    ensure_data(("benchmark",))
    if args.output is None:
        entries, _ = load_sources(ROOT)
        rows = merge_entries(entries)
        with require_data_path(BENCHMARK).open(encoding="utf-8-sig", newline="") as stream:
            if list(csv.DictReader(stream)) != rows:
                raise ValueError("Unified benchmark differs from its reconstructed source records")
        print(f"Verified {len(rows)} unified rows and {len(entries)} embedded source records; no files created.")
        return
    manifest = build_benchmark(ROOT, args.output)
    print(json.dumps({"output": str(args.output), "rows": manifest["records"],
                      "groups": manifest["groups"], "annotation_conflict_rows": manifest["rows_with_annotation_conflicts"]}, indent=2))


if __name__ == "__main__":
    main()
