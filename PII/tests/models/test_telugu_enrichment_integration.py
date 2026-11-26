"""Model-free checks for IndicNER's Telugu-only postproposal wiring."""

from pii_pipeline.models.indic_ner import IndicNERPipeline


def _pipeline(enabled, surface, entity_type="PERSON"):
    pipeline = IndicNERPipeline.__new__(IndicNERPipeline)
    pipeline.max_chunk_chars = 120
    pipeline.chunk_overlap_chars = 40
    pipeline.min_indic_fraction = 0.0
    pipeline.min_confidence = .95
    pipeline.confidence_thresholds = {}
    pipeline.overlap_strategy = "best_context"
    pipeline.telugu_enrichment = {"enabled": enabled}
    pipeline._enrich_chunks = lambda chunks, diagnostics=None: [[{
        "start": chunk.index(surface), "end": chunk.index(surface) + len(surface),
        "entity_type": entity_type, "score": .99, "text": surface,
        "source": "indic_ner",
    }] for chunk in chunks]
    return pipeline


def test_non_telugu_indicner_output_is_identical_with_layer_on_or_off():
    for text in ("Ravi Kumar", "நிலா", "ಮೇಘನಾ"):
        on = _pipeline(True, text)
        off = _pipeline(False, text)
        assert on.enrich_text_with_diagnostics(text) == off.enrich_text_with_diagnostics(text)


def test_trace_retains_raw_proposal_before_telugu_suffix_repair():
    text = "నా పేరు రవిని"
    result, trace = _pipeline(True, "రవిని").enrich_text_with_diagnostics(text)
    assert trace["raw_candidates"][0]["text"] == "రవిని"
    assert trace["enriched_candidates"][0]["text"] == "రవి"
    assert result[0]["text"] == "రవి"
    assert "suffix_stripped" in result[0]["recognition_metadata"]["enrichment_actions"]


def test_telugu_enrichment_is_idempotent_with_optional_cue_boost():
    text = "నా పేరు రవిని"
    pipeline = _pipeline(True, "రవిని")
    pipeline.telugu_enrichment = {"enabled": True, "cue_boost": .02}
    raw = [{"start": text.index("రవి"), "end": len(text), "entity_type": "PERSON",
            "score": .90, "text": "రవిని", "source": "indic_ner"}]
    first, _ = pipeline._enrich_telugu_candidates(text, raw)
    second, _ = pipeline._enrich_telugu_candidates(text, first)
    assert second == first
