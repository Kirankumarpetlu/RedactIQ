"""Prepare human-review sheets and freeze exhaustively reviewed held-out data.

No model predictions are presented as verified annotations. A held-out sample
requires real consultation/patient group identifiers and exclusion files for
every dataset used to tune the pipeline. Duplicate groups stay together.
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
import random
import re
import unicodedata
from pathlib import Path

csv.field_size_limit(10**9)


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text):
    value = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    return {value[i:i + 5] for i in range(max(1, len(value) - 4))}


def similar(a, b, threshold):
    if min(len(a), len(b)) / max(1, max(len(a), len(b))) < threshold:
        return False
    return len(a & b) / max(1, len(a | b)) >= threshold


def select_heldout(rows, excluded, group_column, count=50, seed=42, similarity=0.85):
    """Return complete randomly selected groups with no known dataset leakage."""
    if not group_column or count <= 0 or not 0 < similarity <= 1:
        raise ValueError("A group column, positive row count and valid similarity are required")
    groups = {}
    for row in rows:
        group = row.get(group_column, "").strip()
        if not group:
            raise ValueError(f"Missing consultation/patient group in column {group_column}")
        groups.setdefault(group, []).append(row)
    excluded_groups = {r.get(group_column, "").strip() for r in excluded} - {""}
    excluded_texts = [shingles(r.get("context", "")) for r in excluded if r.get("context", "").strip()]
    signatures = {g: [shingles(r["context"]) for r in records] for g, records in groups.items()}
    # Connected components keep near-duplicate consultations together, even
    # when an intermediary connects two examples that are less similar.
    parent = {g: g for g in groups}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    ids = sorted(groups)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if any(similar(x, y, similarity) for x in signatures[a] for y in signatures[b]):
                parent[find(b)] = find(a)
    blocked = set()
    for group in ids:
        if group in excluded_groups or any(similar(x, y, similarity) for x in signatures[group] for y in excluded_texts):
            blocked.add(find(group))
    components = {}
    for group in ids:
        root = find(group)
        if root not in blocked:
            components.setdefault(root, []).extend(groups[group])
    choices = list(components.values())
    random.Random(seed).shuffle(choices)
    selected = []
    for component in choices:
        if len(selected) >= count:
            break
        selected.extend(component)
    return selected


def has_human_review_signoff(row):
    """Check recorded sign-off; this cannot verify that review actually occurred.

    A known AI reviewer is not human sign-off, even if someone changes status
    strings to reviewed/exhaustive. AI drafts use the provisional evaluator.
    """
    reviewer = row.get("reviewer", "").strip()
    ai_only = re.match(
        r"^(?:chatgpt|codex|claude|gemini|gpt[-0-9.]*)(?:$|[\s:/-])|"
        r"^(?:ai|assistant|automated|model)$", reviewer, re.I)
    return bool(row.get("review_status") == "reviewed" and
                row.get("annotation_scope") == "exhaustive" and reviewer and not ai_only)


def validate_reviewed(rows):
    """Require explicit human sign-off and exact original-text offsets."""
    if not rows:
        raise ValueError("Cannot freeze an empty test set")
    ids = set()
    for i, row in enumerate(rows, 1):
        row_id = row.get("row_index", "")
        if not row_id or row_id in ids:
            raise ValueError(f"Row {i}: missing or duplicate row_index")
        ids.add(row_id)
        if not has_human_review_signoff(row):
            raise ValueError(f"Row {row_id}: human exhaustive review is required")
        text = row["context"]
        if row.get("source_fingerprint") and row["source_fingerprint"] != fingerprint(text):
            raise ValueError(f"Row {row_id}: original source text changed during review")
        tags = json.loads(row["verified_semantic_tags"])
        if not isinstance(tags, list):
            raise ValueError(f"Row {row_id}: tags must be a list (including [] for no PII)")
        seen = set()
        for tag in tags:
            if not isinstance(tag, dict):
                raise ValueError(f"Row {row_id}: invalid annotation")
            start, end = tag.get("start"), tag.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f"Row {row_id}: invalid exact offsets")
            if not isinstance(tag.get("entity_type"), str) or not tag["entity_type"].strip() or tag.get("text") != text[start:end]:
                raise ValueError(f"Row {row_id}: annotation does not match original text")
            key = (tag["entity_type"].casefold(), start, end)
            if key in seen:
                raise ValueError(f"Row {row_id}: duplicate annotation")
            seen.add(key)


def benchmark_draft(row):
    """List possible occurrences for review; never disambiguate repeated text."""
    candidates, issues = [], []
    for label, values in json.loads(row.get("pii_dictionary") or "{}").items():
        for value in values:
            offsets = [m.start() for m in re.finditer(re.escape(value), row["context"])] if value else []
            if len(offsets) != 1:
                issues.append(f"{label}: {len(offsets)} candidate occurrences; review required")
            for start in offsets:
                candidates.append({"entity_type": label, "text": value, "start": start, "end": start + len(value)})
    return candidates, issues


def run_review_draft(rows, benchmark, previous_rows=()):
    """Attach source-aligned predictions and differences for exhaustive review.

    Prediction agreement never counts as human sign-off. Include every run
    row, including negative examples and rows absent from the old reference.
    """
    def context_key(text):
        return text.replace("\u00a0", " ").replace("\u200b", "")

    references, previous = {}, {}
    for row in benchmark:
        key = context_key(row["context"])
        if key in references:
            raise ValueError("Duplicate benchmark context; adjudicate references before preparing review")
        references[key] = row
    for row in previous_rows:
        key = context_key(row["context"])
        if key in previous:
            raise ValueError("Duplicate previous-run context; comparison would be ambiguous")
        previous[key] = row

    def predictions(row):
        entities = json.loads(row.get("pipeline_entities") or "[]")
        if not isinstance(entities, list):
            raise ValueError("pipeline_entities must be a list")
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("Prediction must be an object with original source offsets")
            a, b = entity.get("start"), entity.get("end")
            if (type(a) is not int or type(b) is not int or not 0 <= a < b <= len(row["context"])
                    or not isinstance(entity.get("entity_type"), str) or not entity["entity_type"].strip()
                    or entity.get("text", row["context"][a:b]) != row["context"][a:b]):
                raise ValueError("Prediction does not match original source text")
        return entities

    def entity_key(entity):
        return entity["start"], entity["end"], entity["entity_type"].casefold()

    prepared, ids = [], set()
    for i, row in enumerate(rows, 1):
        row_id = str(row.get("row_index") or row.get("row_id") or i)
        if row_id in ids:
            raise ValueError("Duplicate output row ID")
        ids.add(row_id)
        text = row["context"]
        key = context_key(text)
        reference = references.get(key)
        draft, issues = benchmark_draft({"context": text, "pii_dictionary": reference.get("pii_dictionary")}) if reference else ([], [])
        for tag in draft:
            tag["proposal_source"] = "legacy_benchmark"
        if not reference:
            issues.append("No legacy reference: review the entire transcript, including negative cases")
        current = predictions(row)
        previous_row = previous.get(key)
        if previous_row and previous_row["context"] != text:
            raise ValueError("Previous run source differs; offsets cannot be compared safely")
        old = predictions(previous_row) if previous_row else []
        old_keys, current_keys = {entity_key(e) for e in old}, {entity_key(e) for e in current}
        added = [e for e in current if entity_key(e) not in old_keys]
        removed = [e for e in old if entity_key(e) not in current_keys]
        gold_chars = {k for tag in draft for k in range(tag["start"], tag["end"])}
        current_chars = {k for e in current for k in range(e["start"], e["end"])}
        uncovered = [tag for tag in draft if any(k not in current_chars for k in range(tag["start"], tag["end"]))]
        if uncovered:
            issues.append("Legacy PII not fully covered by the new run; inspect uncovered_legacy_tags")
        for entity in current:
            a, b = entity["start"], entity["end"]
            if not any(k in gold_chars for k in range(a, b)):
                draft.append({"entity_type": entity["entity_type"], "text": text[a:b], "start": a, "end": b,
                              "proposal_source": "pipeline_prediction", "recognizer": entity.get("recognizer", ""),
                              "review_required": True})
                issues.append(f"Unannotated {entity['entity_type']} at {a}:{b}: accept or reject after context review")
            if entity["entity_type"].casefold() in {"city", "sensitive_date", "date_time"}:
                issues.append("Apply the same patient-linked location/calendar-date policy to every transcript")
        if removed:
            issues.append("Inspect removed_predictions: ensure no true PII was lost")
        prepared.append({**row, "row_index": row_id, "source_fingerprint": fingerprint(text),
                         "legacy_benchmark_row_id": reference.get("row_id", "") if reference else "",
                         "proposed_tags": json.dumps(draft, ensure_ascii=False),
                         "new_predictions": json.dumps(added, ensure_ascii=False),
                         "removed_predictions": json.dumps(removed, ensure_ascii=False),
                         "uncovered_legacy_tags": json.dumps(uncovered, ensure_ascii=False),
                         "verified_semantic_tags": "", "review_status": "pending", "annotation_scope": "",
                         "reviewer": "", "review_notes": "; ".join(dict.fromkeys(issues))})
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["benchmark-review", "run-review", "heldout", "freeze"], required=True)
    parser.add_argument("--benchmark", help="Legacy reference CSV for run-review")
    parser.add_argument("--previous-run", help="Previous saved output for run-review differences")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--group-column")
    parser.add_argument("--rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity", type=float, default=0.85)
    args = parser.parse_args()
    rows = read_rows(args.input)
    if args.mode == "heldout":
        if not args.exclude:
            raise ValueError("Held-out preparation requires exclusion datasets used for tuning")
        excluded = [r for path in args.exclude for r in read_rows(path)]
        rows = select_heldout(rows, excluded, args.group_column, args.rows, args.seed, args.similarity)
    if args.mode == "run-review":
        if not args.benchmark:
            raise ValueError("run-review requires --benchmark")
        rows = run_review_draft(rows, read_rows(args.benchmark), read_rows(args.previous_run) if args.previous_run else [])
    elif args.mode == "freeze":
        validate_reviewed(rows)
    else:
        prepared = []
        for i, row in enumerate(rows, 1):
            draft, issues = benchmark_draft(row) if args.mode == "benchmark-review" else ([], [])
            row_id = str(row.get("row_index") or row.get("row_id") or i)
            if args.mode == "benchmark-review" and row_id in {"16", "19", "25", "28", "40", "42", "43", "47"}:
                issues.append("Precision review flagged potentially missing AGE/date/person annotations; adjudicate exhaustively")
            prepared.append({**row, "row_index": row_id, "source_fingerprint": fingerprint(row["context"]),
                             "proposed_tags": json.dumps(draft, ensure_ascii=False), "verified_semantic_tags": "",
                             "review_status": "pending", "annotation_scope": "", "reviewer": "",
                             "review_notes": "; ".join(issues)})
        rows = prepared
    if not rows:
        raise ValueError("No eligible rows remain after exclusions")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    # Exclusive creation prevents accidental replacement of a locked dataset.
    with out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"mode": args.mode, "rows": len(rows), "seed": args.seed, "group_column": args.group_column,
                "near_duplicate_similarity": args.similarity, "dataset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                "exclusions": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.exclude},
                "human_review_complete": args.mode == "freeze"}
    if args.mode == "run-review":
        manifest["benchmark_sha256"] = hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest()
        manifest["previous_run_sha256"] = hashlib.sha256(Path(args.previous_run).read_bytes()).hexdigest() if args.previous_run else None
    out.with_suffix(out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out}; human review complete: {args.mode == 'freeze'}")


if __name__ == "__main__":
    main()
