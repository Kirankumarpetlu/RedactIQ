"""Synthetic unit checks for Hindi IndicNER postprocessing."""
from pii_pipeline.enrichment.hindi_enrichment import enrich_candidates


def _proposal(text, surface, label="PERSON", score=.60):
    start = text.index(surface)
    return {"start": start, "end": start + len(surface), "text": surface,
            "entity_type": label, "score": score, "source": "indic_ner",
            "recognition_metadata": {"acceptance_threshold": .95}}


def test_non_devanagari_is_byte_identical():
    text = "My name is Elena Torres."
    original = [_proposal(text, "Elena Torres")]
    output, rejected = enrich_candidates(text, original)
    assert output is original and rejected == []
    assert output[0] == original[0]


def test_other_indic_script_is_byte_identical():
    text = "నా పేరు అరుణ్ అని చెప్పారు."
    original = [_proposal(text, "అరుణ్")]
    output, rejected = enrich_candidates(text, original)
    assert output is original and rejected == []


def test_latin_candidate_unchanged_inside_mixed_chunk():
    text = "मेरा नाम नीलम है; contact Elena Torres."
    latin = _proposal(text, "Elena Torres")
    devanagari = _proposal(text, "नीलम")
    output, _ = enrich_candidates(text, [latin, devanagari])
    assert output[0] is latin
    assert output[0] == latin


def test_cue_lowers_gate_only_for_matching_model_proposal():
    text = "मेरा नाम नीलिमा भट्ट है।"
    original = [_proposal(text, "नीलिमा भट्ट", score=.46)]
    output, rejected = enrich_candidates(text, original)
    assert not rejected
    assert output[0]["recognition_metadata"]["enrichment_threshold"] == .45
    assert output[0]["recognition_metadata"]["raw_indic_score"] == .46
    assert "cue_threshold" in output[0]["enrichment_actions"]


def test_organization_type_reconciliation_preserves_model_provenance():
    text = "मैं नवदीप अस्पताल में इलाज कराती हूँ।"
    original = [_proposal(text, "नवदीप अस्पताल", label="LOCATION")]
    output, _ = enrich_candidates(text, original)
    assert output[0]["entity_type"] == "ORGANIZATION"
    assert output[0]["source"] == "indic_ner"
    assert "type_reconciled" in output[0]["enrichment_actions"]


def test_no_mid_combining_mark_boundary():
    text = "मेरा नाम कौमुदी सेन है।"
    original = [_proposal(text, "ौमुदी सेन")]
    output, _ = enrich_candidates(text, original)
    assert output[0]["text"] == "कौमुदी सेन"


def test_disabled_is_identity_even_for_devanagari():
    text = "मेरा नाम नीलिमा भट्ट है।"
    original = [_proposal(text, "नीलिमा भट्ट")]
    output, rejected = enrich_candidates(text, original, options={"enabled": False})
    assert output is original and rejected == []


def test_uncued_common_word_keeps_strict_gate():
    text = "राजा की कहानी पढ़ी।"
    original = [_proposal(text, "राजा", score=.72)]
    output, rejected = enrich_candidates(text, original)
    assert not rejected
    assert "enrichment_threshold" not in output[0]["recognition_metadata"]
