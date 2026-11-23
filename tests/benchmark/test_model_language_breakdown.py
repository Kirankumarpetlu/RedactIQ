import hashlib
from pii_pipeline import json_codec as json

from scripts.evaluate_model_language_breakdown import evaluate


def _span(text, value, entity_type, **extra):
    start = text.index(value)
    return {"entity_type": entity_type, "start": start, "end": start + len(value),
            "text": value, **extra}


def test_each_model_uses_its_own_candidates_and_language_denominators():
    texts = ["శైలజ chest pain", "Call 9876543210"]
    gold = [[_span(texts[0], "శైలజ", "PERSON")],
            [_span(texts[1], "9876543210", "PHONE_NUMBER")]]
    final = gold
    references = [{"row_index": str(i), "context": text,
                   "verified_semantic_tags": json.dumps(gold[i-1], ensure_ascii=False)}
                  for i, text in enumerate(texts, 1)]
    predictions = [{"row_index": str(i), "context": text,
                    "pipeline_entities": json.dumps(final[i-1], ensure_ascii=False)}
                   for i, text in enumerate(texts, 1)]
    diagnostics = [
        {"row_index": "1", "source_sha256": hashlib.sha256(texts[0].encode()).hexdigest(),
         "decision_trace": {
             "indic_ner": {"raw_candidates": [
                 _span(texts[0], "శైలజ", "PERSON"),
                 _span(texts[0], "chest pain", "PERSON")]},
             "analyzer_candidates": [
                 _span(texts[0], "శైలజ", "first_name", recognizer="GLiNER2PIIPrimary"),
                 _span(texts[0], "శైలజ", "person", recognizer="GLiNER2PIIPrimary"),
                 _span(texts[0], "chest pain", "PERSON", recognizer="SpacyRecognizer"),
                 _span(texts[0], "శైలజ", "PERSON", recognizer="TeluguNameCueRecognizer")],
             "policy_candidates": [
                 _span(texts[0], "శైలజ", "PERSON", source="indic_ner")]}},
        {"row_index": "2", "source_sha256": hashlib.sha256(texts[1].encode()).hexdigest(),
         "decision_trace": {
             "analyzer_candidates": [
                 _span(texts[1], "Call", "username", recognizer="GLiNER2PIIPrimary"),
                 _span(texts[1], "9876543210", "PHONE_NUMBER", recognizer="PhoneRecognizer")],
             "policy_candidates": [
                 _span(texts[1], "Call", "username", source="fallback_routed",
                       fallback_confirmed=False)]}},
    ]
    rows, metadata = evaluate(references, predictions, diagnostics)

    def metric(source, language="ALL", entity_class="ALL"):
        return next(row for row in rows if row["match"] == "exact" and
                    row["source"] == source and row["language"] == language and
                    row["entity_class"] == entity_class)

    assert metadata["reference_spans"] == 2
    assert metadata["language_rows"] == {"English/Latin": 1, "Telugu": 1}
    assert (metric("final_pipeline")["tp"], metric("final_pipeline")["fp"],
            metric("final_pipeline")["fn"]) == (2, 0, 0)
    assert (metric("GLiNER2PIIPrimary")["tp"], metric("GLiNER2PIIPrimary")["fp"],
            metric("GLiNER2PIIPrimary")["fn"]) == (1, 1, 1)
    assert metric("GLiNER2PIIPrimary")["f1_pct"] == 50.0
    assert metric("GLiNER2PIIPrimary", "Telugu", "PERSON")["recall_pct"] == 100.0
    assert metric("GLiNER2PIIPrimary", "English/Latin")["recall_pct"] == 0.0
    assert (metric("IndicNER_raw")["tp"], metric("IndicNER_raw")["fp"]) == (1, 1)
    assert (metric("IndicNER_emitted")["tp"], metric("IndicNER_emitted")["fp"]) == (1, 0)
    assert (metric("RuleRecognizers_combined")["tp"],
            metric("RuleRecognizers_combined")["fn"]) == (2, 0)
    assert (metric("recognizer:PhoneRecognizer")["tp"],
            metric("recognizer:PhoneRecognizer")["fn"]) == (1, 1)
    # Unconfirmed fallback proposals must not be counted as model positives.
    assert metric("GLiNER2VanillaFallback_confirmed")["predicted_spans"] == 0
    assert metric("GLiNER2VanillaFallback_confirmed")["precision_pct"] is None
    matrix = {(row["match"], row["source"], row["language"], row["entity_class"])
              for row in rows}
    for source in {row["source"] for row in rows}:
        for language in {row["language"] for row in rows}:
            for entity_class in {row["entity_class"] for row in rows}:
                assert ("exact", source, language, entity_class) in matrix
                assert ("iou_50", source, language, entity_class) in matrix
