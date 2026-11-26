"""Score final spans and each detector's emitted candidates by class and script.

The real 500-row reference is pending/incomplete. Precision and F1 here are
*apparent* values assuming its annotations are exhaustive. Detector rows are
raw emitted candidates before the common policy and overlap resolver; the
conditional GLiNER fallback is not a standalone full-corpus model run.
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
from collections import Counter, defaultdict
from pathlib import Path

from scripts.score_against_benchmark import detect_language
from scripts.score_pending_groundtruth import _label, _match, _spans


CORE_SOURCES = {
    "final_pipeline": "final_pipeline",
    "GLiNER2PIIPrimary": "neural_candidate",
    "SpacyRecognizer": "neural_candidate",
    "IndicNER_raw": "neural_candidate_before_enrichment_gate",
    "IndicNER_emitted": "neural_candidate_after_enrichment_gate",
    "GLiNER2VanillaFallback_confirmed": "conditional_neural_candidate",
    "RuleRecognizers_combined": "rule_candidates",
}
NEURAL_ANALYZER_NAMES = {"GLiNER2PIIPrimary", "SpacyRecognizer"}
FIELDS = ("match", "source", "source_kind", "language", "entity_class", "rows",
          "reference_spans", "predicted_spans", "tp", "fp", "fn",
          "precision_pct", "recall_pct", "f1_pct")


def _language(text: str) -> str:
    # Clinical rows are often code-mixed. Use the dominant Indic script if
    # present; a Latin-only row is labelled English/Latin, not language-ID'ed.
    value = detect_language(text)
    return "English/Latin" if value == "Hinglish" else value


def _validated_candidates(items: list[dict], text: str) -> tuple[list[dict], int]:
    valid, seen, invalid = [], set(), 0
    for item in items:
        if not isinstance(item, dict):
            invalid += 1
            continue
        start, end, label = item.get("start"), item.get("end"), item.get("entity_type")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                or not isinstance(label, str) or not label.strip()):
            invalid += 1
            continue
        key = (_label(label), start, end)
        if key not in seen:
            seen.add(key)
            valid.append({"entity_type": key[0], "start": start, "end": end})
    return valid, invalid


def _sources(trace: dict) -> dict[str, list[dict]]:
    sources: dict[str, list[dict]] = defaultdict(list)
    indic_trace = trace.get("indic_ner") or {}
    if isinstance(indic_trace, dict):
        sources["IndicNER_raw"].extend(indic_trace.get("raw_candidates") or [])
    analyzer = trace.get("analyzer_candidates") or []
    for item in analyzer:
        name = str(item.get("recognizer") or "unknown")
        if name in NEURAL_ANALYZER_NAMES:
            sources[name].append(item)
        else:
            sources[f"recognizer:{name}"].append(item)
            sources["RuleRecognizers_combined"].append(item)
    for item in trace.get("policy_candidates") or []:
        if item.get("source") == "indic_ner" or item.get("recognizer") == "IndicNER":
            sources["IndicNER_emitted"].append(item)
        elif item.get("source") == "fallback_routed" and item.get("fallback_confirmed") is True:
            sources["GLiNER2VanillaFallback_confirmed"].append(item)
    return sources


def _rate(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def _metric_row(match: str, source: str, kind: str, language: str, entity_class: str,
                rows: int, counts: Counter) -> dict:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    # Compute F1 from integer support, not already-rounded P/R percentages.
    f1 = _rate(2 * tp, 2 * tp + fp + fn)
    return dict(zip(FIELDS, (match, source, kind, language, entity_class, rows,
                             tp + fn, tp + fp, tp, fp, fn, precision, recall, f1)))


def evaluate(reference_rows: list[dict], prediction_rows: list[dict],
             diagnostic_rows: list[dict]) -> tuple[list[dict], dict]:
    if not (len(reference_rows) == len(prediction_rows) == len(diagnostic_rows)):
        raise ValueError("Reference, prediction, and diagnostic row counts differ")
    diagnostics = {str(row["row_index"]): row for row in diagnostic_rows}
    if len(diagnostics) != len(diagnostic_rows):
        raise ValueError("Duplicate diagnostic row_index")
    rows_by_source: dict[str, list[tuple[str, list[dict], list[dict]]]] = defaultdict(list)
    source_kinds = dict(CORE_SOURCES)
    invalid_by_source = Counter()
    languages = Counter()
    reference_counts = Counter()
    for reference, prediction in zip(reference_rows, prediction_rows):
        row_id, text = reference["row_index"], reference["context"]
        if prediction["row_index"] != row_id or prediction["context"] != text:
            raise ValueError(f"Prediction alignment differs at row {row_id}")
        diagnostic = diagnostics.get(row_id)
        if diagnostic is None or diagnostic.get("source_sha256") != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError(f"Diagnostic alignment differs at row {row_id}")
        language = _language(text)
        languages[language] += 1
        gold = _spans(reference, "verified_semantic_tags")
        reference_counts.update(_label(item["entity_type"]) for item in gold)
        final = _spans(prediction, "pipeline_entities")
        proposals = _sources(diagnostic.get("decision_trace") or {})
        proposals["final_pipeline"] = final
        for source, raw in proposals.items():
            if source.startswith("recognizer:"):
                source_kinds[source] = "individual_recognizer_candidate"
            valid, invalid = _validated_candidates(raw, text)
            invalid_by_source[source] += invalid
            rows_by_source[source].append((language, gold, valid))
        # Append empty predictions to every known source on rows without an
        # emission. Recall must use the entire same 500-row reference.
        for source in source_kinds:
            if source not in proposals:
                rows_by_source[source].append((language, gold, []))

    # Newly seen recognizers may have emitted only after earlier rows. Fill
    # their missing earlier rows in the original order to preserve denominators.
    for source in source_kinds:
        entries = rows_by_source[source]
        if len(entries) < len(reference_rows):
            # Reconstruct by row ID from diagnostics, avoiding any reliance on
            # when a recognizer first appeared.
            entries = []
            for reference in reference_rows:
                text, row_id = reference["context"], reference["row_index"]
                raw = _sources((diagnostics[row_id].get("decision_trace") or {})).get(source, [])
                valid, _ = _validated_candidates(raw, text)
                entries.append((_language(text), _spans(reference, "verified_semantic_tags"), valid))
            rows_by_source[source] = entries

    output = []
    reference_classes = set(reference_counts)
    matrix_classes = reference_classes | {
        _label(item["entity_type"])
        for entries in rows_by_source.values()
        for _, _, predicted in entries
        for item in predicted
    }
    for match, exact in (("exact", True), ("iou_50", False)):
        for source in sorted(source_kinds):
            buckets: dict[tuple[str, str], Counter] = defaultdict(Counter)
            for language, gold, predicted in rows_by_source[source]:
                matched_gold, matched_predicted = _match(gold, predicted, exact=exact)
                for index, item in enumerate(predicted):
                    label = _label(item["entity_type"])
                    key = "tp" if index in matched_predicted else "fp"
                    for bucket in (("ALL", "ALL"), (language, "ALL"),
                                   ("ALL", label), (language, label)):
                        buckets[bucket][key] += 1
                for index, item in enumerate(gold):
                    if index not in matched_gold:
                        label = _label(item["entity_type"])
                        for bucket in (("ALL", "ALL"), (language, "ALL"),
                                       ("ALL", label), (language, label)):
                            buckets[bucket]["fn"] += 1
            # Emit a complete source x language x class matrix. Empty cells
            # remain N/A rather than disappearing or masquerading as 0%.
            for language in ("ALL", *sorted(languages)):
                buckets[(language, "ALL")]
                for entity_class in matrix_classes:
                    buckets[(language, entity_class)]
            for (language, entity_class), counts in sorted(buckets.items()):
                count_rows = len(reference_rows) if language == "ALL" else languages[language]
                output.append(_metric_row(match, source, source_kinds[source], language,
                                          entity_class, count_rows, counts))

    output.sort(key=lambda row: (row["match"], row["source"], row["language"], row["entity_class"]))
    metadata = {
        "reference_status": "provisional_pending_incomplete_ai_review",
        "rows": len(reference_rows), "reference_spans": sum(reference_counts.values()),
        "language_rows": dict(sorted(languages.items())),
        "reference_classes": dict(sorted(reference_counts.items())),
        "invalid_candidate_counts": dict(sorted(invalid_by_source.items())),
        "language_rule": "Dominant Indic script among non-Latin letters; English/Latin if no Indic script. Rows may be code-mixed.",
        "model_rule": "Analyzer candidates are emitted spans before shared policy/resolver. IndicNER_raw is before its enrichment gate; IndicNER_emitted is after that gate. Vanilla fallback includes only confirmed conditional proposals.",
        "undefined_rates": "Empty CSV values mean no denominator; they are not zero accuracy.",
        "precision_caveat": "Apparent precision/F1 assume the pending reference is exhaustive; human-verified precision is unavailable.",
    }
    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.reference.open(encoding="utf-8-sig", newline="") as stream:
        reference = list(csv.DictReader(stream))
    with args.predictions.open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    with args.diagnostics.open(encoding="utf-8") as stream:
        diagnostics = [json.loads(line) for line in stream if line.strip()]
    rows, metadata = evaluate(reference, predictions, diagnostics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for match in ("exact", "iou_50"):
        path = args.output_dir / f"{match}_prf_by_source_language_class.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in rows if row["match"] == match)
        print(path)
    (args.output_dir / "report_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        if (row["match"] == "exact" and row["language"] == "ALL" and row["entity_class"] == "ALL"
                and row["source"] in CORE_SOURCES):
            print(f"{row['source']}: TP={row['tp']} FP={row['fp']} FN={row['fn']} "
                  f"P={row['precision_pct']}% R={row['recall_pct']}% F1={row['f1_pct']}%")
    print("Reference is pending/incomplete; precision and F1 are apparent, not verified.")


if __name__ == "__main__":
    main()
