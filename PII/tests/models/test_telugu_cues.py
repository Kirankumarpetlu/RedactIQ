"""Small string-only checks for the optional IndicNER cue stage."""

from pii_pipeline.enrichment.telugu_cues import cue_evidence, enrich_cues


def _proposal(text, value, label="PERSON", score=.94):
    start = text.index(value)
    return {"start": start, "end": start + len(value), "text": value,
            "entity_type": label, "score": score, "source": "indic_ner"}


def test_name_cue_preserves_raw_provenance_and_score():
    text = "నా పేరు రవి"
    result = enrich_cues(text, [_proposal(text, "రవి")])[0]
    assert result["score"] == .94
    assert result["source"] == "indic_ner"
    assert result["recognition_metadata"]["raw_indic_score"] == .94
    assert result["recognition_metadata"]["enrichment_threshold"] == .95
    assert result["recognition_metadata"]["telugu_cue_evidence"]["name"] == "explicit_name"


def test_first_person_pronoun_is_not_sufficient_for_name():
    text = "నేను హైదరాబాద్‌లో ఉంటాను"
    candidate = _proposal(text, "హైదరాబాద్", "PERSON")
    result = enrich_cues(text, [candidate])[0]
    assert result["entity_type"] == "LOCATION"
    assert result["recognition_metadata"]["enrichment_threshold"] == .95
    assert cue_evidence(text, candidate["start"], candidate["end"])["name"] is None


def test_explicit_address_reconciles_person_to_location():
    text = "గ్రామం కొత్తపల్లి"
    result = enrich_cues(text, [_proposal(text, "కొత్తపల్లి")])[0]
    assert result["entity_type"] == "LOCATION"
    assert "type_reconciled" in result["enrichment_actions"]
    assert result["recognition_metadata"]["raw_indic_type"] == "PERSON"


def test_non_values_are_advisory_rejections():
    text = "నేను డాక్టర్"
    result = enrich_cues(text, [_proposal(text, "డాక్టర్")])[0]
    assert result["enrichment_rejection_reason"] == "telugu_non_value_in_context"


def test_non_telugu_is_equal_to_raw_candidate():
    candidate = _proposal("My name is Ravi", "Ravi")
    assert enrich_cues("My name is Ravi", [candidate]) == [candidate]


def test_score_lift_requires_explicit_option():
    text = "పేరు: రవి"
    result = enrich_cues(text, [_proposal(text, "రవి")], options={"cue_score_lift": .01})[0]
    assert result["score"] == .95
    assert result["recognition_metadata"]["raw_indic_score"] == .94
    assert "cue_boost" in result["enrichment_actions"]


def test_lower_cue_threshold_requires_explicit_option():
    text = "పేరు: రవి"
    result = enrich_cues(text, [_proposal(text, "రవి")],
                         options={"cue_thresholds": {"person": .93}})[0]
    assert result["recognition_metadata"]["enrichment_threshold"] == .93


def test_normalized_view_keeps_original_offsets():
    class View:
        def __init__(self, text):
            self.text = text
            self.origins = tuple((i, i + 1) for i in range(len(text)))

    text = "నా పేరు రవి"
    candidate = _proposal(text, "రవి")
    result = enrich_cues(text, [candidate], normalized=View(text))[0]
    assert (result["start"], result["end"], result["text"]) == (
        candidate["start"], candidate["end"], candidate["text"])
    assert result["recognition_metadata"]["telugu_cue_evidence"]["name"] == "explicit_name"
