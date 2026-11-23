"""Real mixed-language failures and independent positives; no model downloads."""
from __future__ import annotations

import itertools
from pii_pipeline import json_codec as json
import math

from models.indicner.entity_reconstruction import EntityReconstructor
import pytest

from pii_pipeline.config import load_pipeline_config
from pii_pipeline.policy.entity_policy import resolve_entities
from pii_pipeline.policy.indic_filters import indic_script_fraction
from pii_pipeline.models.indic_ner import IndicNERPipeline
from pii_pipeline.models.indic_scoring import decode_probabilities, validate_inference_controls


def proposal(text, value, recognizer="SpacyRecognizer", score=.85, label="PERSON"):
    start = text.index(value)
    return dict(entity_type=label, start=start, end=start+len(value), text=value,
                score=score, recognizer=recognizer)


@pytest.mark.parametrize("text,value", [
    ("ஏதாவது ஒரு particular இடத்துல தொட்டா வலி இருக்கா?", "particular இடத்துல"),
    ("எதுவும் periods related pain இல்ல.", "எதுவும் periods"),
    ("இது HMS இது டாக்டரோடது தானே?", "இது HMS"),
    ("அட்ரோவா 40 எடுக்கணும். Then Bison எடுக்கணும்.", "Bison"),
    ("Clopid plain. Dozer? சரி, doser. Basal 40. RDS 10. Cardies 10.", "Dozer"),
    ("Neuro kind தான். Risk இல்ல. Neuro kind D3.", "Risk இல்ல"),
    ("Renal D க்கு பதிலா neuro kindன்னு கொடுத்தார்.", "neuro kindன்னு"),
])
def test_english_spacy_cannot_establish_identity_in_mixed_clinical_speech(text, value):
    accepted, rejected = resolve_entities(text, [proposal(text, value)])
    assert not accepted
    assert rejected[0]["reason"] in {"unsupported_spacy_mixed_context", "unsupported_spacy_script"}


@pytest.mark.parametrize("text,value", [
    ("Name: Robert. தமிழில் சொல்லுங்கள்.", "Robert"),
    ("My name is Martha Lake. మీకు ఎలా ఉంది?", "Martha Lake"),
    ("Mr. Gawe. எப்படி இருக்கீங்க?", "Gawe"),
    ("Dr. Senol. మీకు ఎలా ఉంది?", "Senol"),
    ("My name is Şenol.", "Şenol"),
    ("Name: அனன்யா. எப்படி இருக்கீங்க?", "அனன்யா"),
    ("I met Ananya yesterday.", "Ananya"),
])
def test_identity_fields_unfamiliar_names_and_english_text_remain_supported(text, value):
    accepted, _ = resolve_entities(text, [proposal(text, value)])
    assert [e["text"] for e in accepted] == [value]


def test_only_validated_independent_candidates_can_support_spacy():
    text = "Robert வந்தார். Tablet எடுக்கணும்."
    spacy, model = proposal(text, "Robert"), proposal(text, "Robert", "GLiNER2PIIPrimary", .92)
    for order in [spacy, model], [model, spacy]:
        accepted, _ = resolve_entities(text, order)
        assert accepted[0]["text"] == "Robert"
    malformed = {**model, "text": "wrong"}
    assert resolve_entities(text, [spacy, malformed])[0] == []


def test_an_imported_guess_alone_is_not_independent_spacy_confirmation():
    text = "Particular முடிந்ததும் வலி."
    _, rejected = resolve_entities(text, [proposal(text, "Particular"),
        proposal(text, "Particular", "ImportedSemanticTag", .98)])
    assert any(r["reason"] == "unsupported_spacy_mixed_context" for r in rejected)


def test_a_corroborated_complete_latin_name_does_not_lose_its_first_name():
    text = "Leo Messi గురించి చెప్పారు. Patient note."
    proposals = [proposal(text, "Leo Messi"), proposal(text, "Messi", "GLiNER2PIIPrimary", .7)]
    accepted, _ = resolve_entities(text, proposals)
    assert accepted[0]["text"] == "Leo Messi"


def test_a_name_repeated_after_a_validated_introduction_stays_masked():
    text = "My name is Dr. Martha Lake. Martha Lake. మీకు నొప్పి ఉందా?"
    first = proposal(text, "Martha Lake")
    start = text.index("Martha Lake", first["end"])
    repeated = {**first, "start": start, "end": start+len("Martha Lake")}
    accepted, _ = resolve_entities(text, [repeated, first])
    assert [(e["start"], e["end"]) for e in accepted] == [(first["start"], first["end"]), (repeated["start"], repeated["end"])]
    assert accepted[1]["recognition_metadata"]["supported_name_repetition"]["start"] == first["start"]


@pytest.mark.parametrize("name", ["Messi", "Mbappé"])
def test_public_names_are_not_dropped_to_match_empty_benchmark_annotations(name):
    text = f"अच्छा चांस है यार। {name} scores a goal."
    assert resolve_entities(text, [proposal(text, name)])[0][0]["text"] == name


def test_accented_latin_is_not_indic_but_all_main_indic_blocks_are():
    assert indic_script_fraction("Şenol François") == 0
    for text in ["राम", "అనన్య", "அனன்யா", "નામ", "ਨਾਮ", "ନାମ"]:
        assert indic_script_fraction(text) == 1


def test_constrained_bio_uses_best_legal_path_without_changing_probabilities():
    labels = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
    probabilities = [[.10, .25, .60, .04, .01], [.01, .02, .93, .02, .02]]
    assert decode_probabilities(probabilities, labels, "argmax") == [2, 2]
    assert decode_probabilities(probabilities, labels, "constrained_bio") == [1, 2]
    # Independently enumerate all legal paths and score their joint probability.
    paths = [p for p in itertools.product(range(5), repeat=2)
             if not labels[p[0]].startswith("I-") and
             (not labels[p[1]].startswith("I-") or labels[p[0]][:2] in {"B-", "I-"}
              and labels[p[0]][2:] == labels[p[1]][2:])]
    best = max(paths, key=lambda p: math.prod(probabilities[i][v] for i, v in enumerate(p)))
    assert tuple(decode_probabilities(probabilities, labels, "constrained_bio")) == best


def test_constrained_bio_cannot_continue_a_different_entity_type():
    labels = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC", 4: "I-LOC"}
    probabilities = [[.01, .96, .01, .01, .01], [.01, .01, .20, .38, .40]]
    assert decode_probabilities(probabilities, labels, "constrained_bio") == [1, 3]
    assert decode_probabilities([], labels, "constrained_bio") == []


@pytest.mark.parametrize("changes", [
    {"temperature": 0}, {"temperature": True}, {"temperature": float("nan")},
    {"strategy": "unknown"}, {"min_word_margin": -.1}, {"min_word_margin": 2},
])
def test_invalid_inference_controls_fail_before_model_loading(changes):
    with pytest.raises(ValueError):
        validate_inference_controls(**changes)


def test_word_margin_distinguishes_ambiguous_evidence_and_protects_a_clear_name():
    raw = {"start": 0, "end": 6, "label": "B-PER", "confidence": .45,
           "probabilities": {"B-PER": .45, "I-PER": .1, "O": .45}}
    baseline = EntityReconstructor().reconstruct("Ananya", [raw])
    assert baseline["entities"]
    assert baseline["word_predictions"][0]["probability_margin"] == pytest.approx(.1)
    assert not EntityReconstructor(.15).reconstruct("Ananya", [raw])["entities"]
    clear = {**raw, "probabilities": {"B-PER": .92, "I-PER": .03, "O": .05}}
    assert EntityReconstructor(.15).reconstruct("Ananya", [clear])["entities"]
    assert baseline["subword_predictions"][0]["probabilities"] == raw["probabilities"]


class OverlapService:
    def analyze_batch(self, chunks):
        output = []
        for chunk in chunks:
            index = chunk.index("NAME")
            # The first view is near its edge and more confident. Context ranking
            # must not simply choose the largest score.
            score = .99 if index == 12 else .96
            output.append({"entities": [{"start": index, "end": index+4,
                "type": "PERSON", "confidence": score, "text": "NAME"}],
                "subword_predictions": [], "word_predictions": []})
        return output


def test_overlapping_chunks_keep_best_context_and_all_original_scores(monkeypatch):
    import pii_pipeline.models.indic_ner as module
    text = "అనన్య అనన్య NAME అనన్య అనన్య"
    index = text.index("NAME")
    chunks = [(index-12, text[index-12:index+4]),
              (index-5, text[index-5:index+9])]
    monkeypatch.setattr(module, "iter_chunks", lambda *args: iter(chunks))
    p = IndicNERPipeline.__new__(IndicNERPipeline)
    p.use_onnx = False; p._local_service = OverlapService()
    p.max_chunk_chars = 120; p.chunk_overlap_chars = 40
    p.min_indic_fraction = 0; p.min_confidence = .95
    p.overlap_strategy = "best_context"; p.confidence_thresholds = {}
    diagnostics = {}
    values = p.enrich_text(text, diagnostics=diagnostics)
    assert len(values) == 1 and values[0]["start"] == index
    assert values[0]["score"] == .96
    assert len(values[0]["recognition_metadata"]["overlap_views"]) == 2
    assert len(diagnostics["token_scores"]) == 2
    p.overlap_strategy = "first"
    assert p.enrich_text(text)[0]["score"] == .99


def test_entity_and_script_gates_are_specific_not_global():
    p = IndicNERPipeline.__new__(IndicNERPipeline)
    p.min_confidence = .95
    p.confidence_thresholds = {"person:devanagari": .98, "person": .96}
    assert p.confidence_threshold({"entity_type": "PERSON", "text": "राम"}) == .98
    assert p.confidence_threshold({"entity_type": "PERSON", "text": "அனன்யா"}) == .96
    assert p.confidence_threshold({"entity_type": "LOCATION", "text": "दिल्ली"}) == .95


def test_tuning_configuration_loads_without_enabling_arbitrary_cutoffs(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {"indic_ner": {
        "decoding_strategy": "constrained_bio", "temperature": 1.2,
        "min_word_margin": .1, "confidence_thresholds": {"PERSON:TAMIL": .97},
        "overlap_strategy": "first"}}}), encoding="utf-8")
    cfg = load_pipeline_config(path).models.indic_ner
    assert cfg.decoding_strategy == "constrained_bio"
    assert cfg.temperature == 1.2 and cfg.min_word_margin == .1
    assert cfg.confidence_thresholds == {"person:tamil": .97}
    assert cfg.overlap_strategy == "first"
    current = load_pipeline_config("config/pipeline_config.yaml").models.indic_ner
    assert current.min_confidence == .95 and current.confidence_thresholds == {}
    assert current.temperature == 1 and current.min_word_margin == 0


@pytest.mark.parametrize("key,value", [("person:typo", .97), ("person:tamil", True),
                                      ("person", float("inf")), ("password", .9)])
def test_invalid_script_or_label_tuning_keys_cannot_silently_do_nothing(tmp_path, key, value):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {"indic_ner": {"confidence_thresholds": {key: value}}}}))
    with pytest.raises(ValueError):
        load_pipeline_config(path)


@pytest.mark.parametrize("text,value,label", [
    ("Take vitamin D3 after food.", "D3", "password"),
    ("HbA1c test was normal.", "HbA1c", "access_token"),
    ("Vitamin B12 supplement was prescribed.", "B12", "username"),
    ("Vitamin B12 supplement was prescribed.", "B12", "full_name"),
    ("எடுத்துக்கறீங்களா? Novel cure தான். Novel cure D3.", "Novel cure", "full_name"),
    ("విటమిన్ B12 వాడుతున్నావు.", "B12", "ICD10_CODE"),
    ("Kidney problem ఉంది.", "Kidney", "full_name"),
    ("Eat guavas and drink water.", "guavas", "last_name"),
    ("నీళ్లు, పాలు, మజ్జిగ, పళ్ళ రసం తాగండి.", "పళ్ళ రసం", "full_name"),
    ("శనగపిండి అట్లు, జొన్నలు, రాగులు తినండి.", "శనగపిండి", "last_name"),
    ("எடுத்துக்கறீங்களா? Neuro band 4. Neuro kind தான். Risk இல்ல. Neuro kind D3.", "Neuro kind", "username"),
])
def test_clinical_and_food_evidence_applies_to_neural_aliases(text, value, label):
    assert not resolve_entities(text, [proposal(text, value, "GLiNER2PIIPrimary", .99, label)])[0]


@pytest.mark.parametrize("value,label,field", [
    ("D3", "password", "Password:"), ("HbA1c", "access_token", "API key:"),
    ("Kidney", "PERSON", "Name:"), ("Guava", "PERSON", "Name:"),
])
def test_explicit_identity_or_credentials_preserve_unusual_values(value, label, field):
    text = f"{field} {value}."
    assert resolve_entities(text, [proposal(text, value, "GLiNER2PIIPrimary", .99, label)])[0]


def test_explicit_diagnosis_code_is_preserved():
    text = "Diagnosis ICD10: B12."
    assert resolve_entities(text, [proposal(text, "B12", "ICD10Recognizer", .99, "ICD10_CODE")])[0]


def test_medication_context_does_not_override_an_explicit_username():
    text = "Take vitamins. Username: Neuro kind. Vitamin D3."
    assert resolve_entities(text, [proposal(text, "Neuro kind", "GLiNER2PIIPrimary", .99, "username")])[0]


def test_vendor_decoder_changes_are_part_of_resume_identity(tmp_path):
    from pii_pipeline.run_csv_through_stage9 import _inference_code_hashes
    core = tmp_path / "core"; vendor = tmp_path / "vendor"
    core.mkdir(); (vendor / "indicner").mkdir(parents=True)
    (core / "pipeline.py").write_text("old")
    (vendor / "indicner/model.py").write_text("old")
    before = _inference_code_hashes(core, vendor)
    (vendor / "indicner/model.py").write_text("new")
    after = _inference_code_hashes(core, vendor)
    assert before["pipeline.py"] == after["pipeline.py"]
    assert before["indicner_package/indicner/model.py"] != after["indicner_package/indicner/model.py"]


@pytest.mark.parametrize("changed", ["models/rules.py", "policy/rules.py", "recognizers/rules.py"])
def test_nested_pipeline_changes_are_part_of_resume_identity(tmp_path, changed):
    from pii_pipeline.run_csv_through_stage9 import _inference_code_hashes
    core, vendor = tmp_path / "core", tmp_path / "vendor"
    paths = ["models/rules.py", "policy/rules.py", "recognizers/rules.py"]
    for relative in paths:
        target = core / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old")
    before = _inference_code_hashes(core, vendor)
    assert set(before) == set(paths)
    (core / changed).write_text("new")
    after = _inference_code_hashes(core, vendor)
    assert {relative for relative in paths if before[relative] != after[relative]} == {changed}
