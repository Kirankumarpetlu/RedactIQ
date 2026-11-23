"""Replay saved detector output with and without the new semantic checks.

This uses fixed saved model predictions and reruns only the cheap name/place
rules. It is an offline policy ablation, not a fresh neural-model benchmark.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pii_pipeline import json_codec as json
from pii_pipeline.policy import entity_policy
from pii_pipeline.recognizers import indic_identifiers


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/output/new_benchmarks_run_01"
OUTPUT = ROOT / "output/semantic_sanity_replay"
NAMES = {
    "IndicNER": "IndicNER", "GLiNER2 primary": "GLiNER2PIIPrimary",
    "spaCy (en_core_web_sm)": "SpacyRecognizer", "Name gazetteer": "IndicPersonNameGazetteer",
    "Name-context rule": "IndicNameContextRecognizer", "Age recognizer": "AgeRecognizer",
    "Short-date recognizer": "ShortDateRecognizer", "Pattern recognizers": "PatternRecognizer",
}


def _run(rows, saved, before):
    name = indic_identifiers.TeluguNameCueRecognizer()
    place = indic_identifiers.SpokenPlaceRecognizer()
    out, counts = [], Counter()
    original_priority = entity_policy._evidence_priority

    def old_priority(entity):
        if entity.get("recognizer") == "TeluguNameCueRecognizer" and (
            entity.get("recognition_metadata") or {}).get("boundary_evidence") == "patient_introduction":
            return 3
        if entity.get("recognizer") == "SpokenPlaceRecognizer" and (
            entity.get("recognition_metadata") or {}).get("boundary_evidence") in {"personal_place_frame", "landmark_relation"}:
            return 1
        return original_priority(entity)

    def identity_span(text, start, end):
        return start, end

    patches = (
        patch.object(entity_policy, "implausible_person", lambda value: False),
        patch.object(entity_policy, "implausible_place", lambda value: False),
        patch.object(entity_policy, "is_known_place", lambda value: False),
        patch.object(entity_policy, "unsupported_username", lambda text, start, end: False),
        patch.object(entity_policy, "normalize_place_span", identity_span),
        patch.object(entity_policy, "_evidence_priority", old_priority),
        patch.object(indic_identifiers, "implausible_person", lambda value: False),
        patch.object(indic_identifiers, "implausible_place", lambda value: False),
        patch.object(indic_identifiers, "normalize_place_span", identity_span),
    ) if before else ()
    from contextlib import ExitStack
    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        for row in rows:
            text = row["context"]
            row_id = row["row_index"]
            candidates = []
            for display, recognizer in NAMES.items():
                for item in saved[display].get(row_id, []):
                    candidates.append({**item, "source": "saved_detector", "recognizer": recognizer})
            # Only the two affected rules are rerun. Saved place/name-context
            # outputs were produced by a different source snapshot.
            if before:
                with patch.object(indic_identifiers, "_load_place_gazetteer", return_value=(frozenset(), None)):
                    name_results = name.analyze(text, ["PERSON"])
            else:
                name_results = name.analyze(text, ["PERSON"])
            for result in name_results + place.analyze(text, ["address"]):
                candidates.append(entity_policy.entity_from_result(result, "replayed_rule"))
            selected, _ = entity_policy.resolve_entities(text, candidates)
            counts.update(item["canonical_entity_type"] for item in selected)
            out.append({"row_index": row_id, "context": text,
                        "pipeline_entities": json.dumps(selected, ensure_ascii=False)})
    return out, counts


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (DATA / "control_full_095.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    raw = json.loads((DATA / "detector_raw_predictions.json").read_text(encoding="utf-8"))
    saved = {name: {row["row_index"]: row["entities"] for row in values}
             for name, values in raw.items()}
    for label, before in (("before", True), ("after", False)):
        predictions, counts = _run(rows, saved, before)
        with (OUTPUT / f"{label}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["row_index", "context", "pipeline_entities"])
            writer.writeheader()
            writer.writerows(predictions)
        print(label, "rows", len(rows), "entities", sum(counts.values()), "types", dict(counts))
    print("Replay predictions:", OUTPUT)


if __name__ == "__main__":
    main()
