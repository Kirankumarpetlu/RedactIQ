"""Context acceptance, clinical negatives and annotation review gates."""
from __future__ import annotations

from pii_pipeline import json_codec as json

import pytest

from pii_pipeline.policy.entity_policy import parse_semantic_tags, resolve_entities
from pii_pipeline.recognizers.indic_identifiers import AgeRecognizer, IndicPersonNameGazetteer
from scripts.evaluate_pipeline import PipelineEvaluator
from scripts.prepare_precision_dataset import run_review_draft, validate_reviewed


def candidate(text, value, label="PERSON", recognizer="SpacyRecognizer", **extra):
    start = text.index(value)
    return {"entity_type": label, "start": start, "end": start + len(value), "text": value,
            "score": .9, "recognizer": recognizer, **extra}


@pytest.mark.parametrize("text,value,reason", [
    ("Alcohol: I. Medications: please add paracetamol.", "I. Medications", "clinical_heading_not_name"),
    ("Ear drops ofloxacin place? Lidocaine drops.", "ofloxacin place", "medication_not_name"),
    ("Sleep irregular. Discomfort. Family history normal.", "Discomfort", "clinical_term_not_name"),
    ("Add tablet andosterone 4 mg if vomiting.", "tablet andosterone 4", "medication_not_name"),
    ("And tablet andosterone 4 mg if vomiting.", "andosterone 4", "medication_not_name"),
    ("And Olapatadin eye drops.", "Olapatadin", "medication_not_name"),
    ("Near my. Spinal. Cord, I guess.", "Cord", "anatomy_not_name"),
    # Independent negatives outside the saved sample.
    ("Give capsule ExampleDrug 250 mg twice daily.", "ExampleDrug", "medication_not_name"),
    ("Medical history: no previous admissions.", "Medical history", "clinical_heading_not_name"),
])
def test_clinical_guesses_are_rejected(text, value, reason):
    selected, rejected = resolve_entities(text, [candidate(text, value)])
    assert not selected
    assert rejected[0]["reason"] == reason


@pytest.mark.parametrize("text,value", [
    ("My name is Cord. I have back pain.", "Cord"),
    ("Name: Discomfort", "Discomfort"),
    ("Doctor Ananya Rao advised tablets.", "Ananya Rao"),
    ("Olapatadin attended the appointment.", "Olapatadin"),
])
def test_clinical_checks_do_not_blacklist_possible_names(text, value):
    selected, _ = resolve_entities(text, [candidate(text, value)])
    assert [e["text"] for e in selected] == [value]


@pytest.mark.parametrize("label", ["PERSON", "full_name", "first_name", "last_name", "middle_name"])
@pytest.mark.parametrize("value", ["హలో డాక్టర్", "లాస్ విజన్", "లాస్", "జస్ట్"])
def test_greetings_and_discourse_are_filtered_for_all_name_aliases(label, value):
    assert not resolve_entities(value, [candidate(value, value, label, "GLiNER2PIIPrimary")])[0]


def test_greeting_does_not_remove_nearby_real_name():
    text = "హలో డాక్టర్. My name is Ananya Rao."
    selected, _ = resolve_entities(text, [candidate(text, "హలో డాక్టర్", "full_name"), candidate(text, "Ananya Rao")])
    assert [e["text"] for e in selected] == ["Ananya Rao"]


@pytest.mark.parametrize("text,value,label,reason", [
    ("ఉమ్. Continue medicines.", "ఉమ్", "username", "credential_is_conversation_filler"),
    ("Add టాబ్లెట్ ఓ ఆర్ ఎస్, sip after each stool.", "ఓ ఆర్ ఎస్", "password", "credential_in_medication_context"),
    ("Give capsule ExampleDrug 20 mg.", "ExampleDrug", "api_key", "credential_in_medication_context"),
])
def test_credentials_need_non_clinical_evidence(text, value, label, reason):
    selected, rejected = resolve_entities(text, [candidate(text, value, label, "GLiNER2PIIPrimary")])
    assert not selected and rejected[0]["reason"] == reason


@pytest.mark.parametrize("text,value,label", [
    ("Password is ORS", "ORS", "password"),
    ("Password is ఓ ఆర్ ఎస్", "ఓ ఆర్ ఎస్", "password"),
    ("Username: ఉమ్", "ఉమ్", "username"),
    ("Secret = hello", "hello", "secret"),
    ("Password: 100 mg", "100 mg", "password"),
    ("user_48217", "user_48217", "username"),
    ("zX92!kPq", "zX92!kPq", "password"),
])
def test_real_or_explicit_credentials_survive(text, value, label):
    selected, _ = resolve_entities(text, [candidate(text, value, label, "GLiNER2PIIPrimary")])
    assert [e["text"] for e in selected] == [value]


def test_unlabelled_medical_measurement_is_not_a_password():
    text = "HbA1c=6.5"
    selected, rejected = resolve_entities(text, [candidate(text, text, "password", "GLiNER2PIIPrimary")])
    assert not selected
    assert rejected[0]["reason"] == "unsupported_ambiguous_model_type"


@pytest.mark.parametrize("value", ["1 month", "two weeks", "3-5 days", "24 hours"])
def test_durations_do_not_become_calendar_dates(value):
    text = "Pain since " + value + "."
    assert not resolve_entities(text, [candidate(text, value, "sensitive_date")])[0]


@pytest.mark.parametrize("value", ["April 5th", "12/5", "01/01/1990"])
def test_actual_calendar_dates_remain_protected(value):
    text = "Last visit was " + value + "."
    assert resolve_entities(text, [candidate(text, value, "sensitive_date")])[0]


def test_imported_tags_use_the_same_context_checks_and_metadata_is_safe():
    text = "Pain since 1 month; give ExampleDrug eye drops."
    raw = [dict(tag="sensitive_date", **{k:v for k,v in candidate(text, "1 month").items() if k in {"start","end","text"}}),
           dict(tag="PERSON", **{k:v for k,v in candidate(text, "ExampleDrug").items() if k in {"start","end","text"}})]
    parsed, _ = parse_semantic_tags(raw)
    assert not resolve_entities(text, parsed)[0]
    assert resolve_entities("Ananya", [candidate("Ananya", "Ananya", recognition_metadata=[])])[0]


@pytest.mark.parametrize("text,age", [
    ("36 ఇయర్స్ ఫీమేల్.", "36"),
    ("నాకు 45 ఇయర్స్. నాకు పెళ్లి అయింది.", "45"),
    ("ನಾನು 22 ಇಯರ್ಸ್ ಓಲ್ಡ್.", "22"),
    ("ಐ ಆಮ್ 20 ಇಯರ್ಸ್ ಓಲ್ಡ್.", "20"),
    ("I'm 47 years, and I live in Pune.", "47"),
    ("Patient: 52 yrs male.", "52"),
])
def test_supported_age_frames(text, age):
    spans = AgeRecognizer().analyze(text, ["AGE"])
    assert [text[e.start:e.end] for e in spans] == [age]


@pytest.mark.parametrize("text", [
    "Pain since 3 years.", "Treatment for 5 ఇయర్స్.", "Working for 8 ಇಯರ್ಸ್.",
    "I am 3 years on treatment.", "Age 0", "My age is 121.",
    "Dosage is 45 mg.", "Storage: 27 days.",
])
def test_new_age_rules_do_not_capture_durations_or_invalid_ages(text):
    assert not AgeRecognizer().analyze(text, ["AGE"])


def test_tamil_case_root_uses_original_offsets_and_keeps_the_suffix():
    text = "இல்ல, நம்ம ராதிகாக்கு உதவி தேவை."
    spans = IndicPersonNameGazetteer().analyze(text, ["PERSON"])
    assert [text[e.start:e.end] for e in spans] == ["ராதிகா"]
    assert spans[0].start == text.index("ராதிகா")
    assert text[spans[0].end:spans[0].end + len("க்கு")] == "க்கு"


@pytest.mark.parametrize("text", ["நம்ம அறியாதவருக்கு உதவி.", "ராதிகாக்கு உதவி.", "நம்ம ராதிகாமற்றது."])
def test_tamil_case_matching_requires_a_listed_root_supported_suffix_and_frame(text):
    assert not IndicPersonNameGazetteer().analyze(text, ["PERSON"])


def test_run_review_surfaces_missing_labels_and_removals_without_approving(tmp_path):
    import csv
    text = "My age is 45. Pain since 1 month."
    old = dict(row_index="sample-1", context=text, pipeline_entities=json.dumps([candidate(text, "1 month", "sensitive_date")]))
    new = dict(row_index="sample-1", context=text, pipeline_entities=json.dumps([candidate(text, "45", "AGE", "AgeRecognizer")]))
    benchmark = [dict(row_id="ref-7", context=text, pii_dictionary="{}")]
    rows = run_review_draft([new], benchmark, [old])
    row = rows[0]
    assert row["review_status"] == "pending" and row["verified_semantic_tags"] == ""
    assert json.loads(row["proposed_tags"])[0]["text"] == "45"
    assert json.loads(row["removed_predictions"])[0]["text"] == "1 month"
    assert json.loads(row["new_predictions"])[0]["text"] == "45"
    assert row["legacy_benchmark_row_id"] == "ref-7"
    with pytest.raises(ValueError, match="review"):
        validate_reviewed(rows)
    path = tmp_path / "pending.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="reviewed"):
        PipelineEvaluator().load_ground_truth(str(path))


def test_review_includes_unmatched_rows_and_repeated_name_occurrences():
    text = "Ananya met Ananya."
    row = dict(row_index="9", context=text, pipeline_entities="[]")
    review = run_review_draft([row], [dict(row_id="1", context=text, pii_dictionary='{"PERSON":["Ananya"]}')])[0]
    assert len(json.loads(review["proposed_tags"])) == 2
    assert len(json.loads(review["uncovered_legacy_tags"])) == 2
    assert "2 candidate occurrences" in review["review_notes"]
    unmatched = run_review_draft([row], [])[0]
    assert unmatched["review_status"] == "pending" and "No legacy reference" in unmatched["review_notes"]


def test_review_refuses_invalid_offsets_and_ambiguous_alignment():
    row = dict(row_index="1", context="Ananya", pipeline_entities=json.dumps([candidate("Ananya", "Ananya", start=-1)]))
    with pytest.raises(ValueError, match="source text"):
        run_review_draft([row], [])
    row["pipeline_entities"] = "[]"
    reference = dict(context="Ananya", pii_dictionary="{}")
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        run_review_draft([row], [reference, reference])
    row["pipeline_entities"] = "[null]"
    with pytest.raises(ValueError, match="object"):
        run_review_draft([row], [])


@pytest.mark.parametrize("status", ["pending", "reviewed"])
def test_legacy_scorer_cannot_ignore_verified_annotations(tmp_path, monkeypatch, status):
    import csv
    from scripts.score_against_benchmark import main
    path = tmp_path / "review.csv"
    row = dict(row_index="1", context="Ananya", pii_dictionary="{}", review_status=status,
               verified_semantic_tags='[{"entity_type":"PERSON","text":"Ananya","start":0,"end":6}]')
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    monkeypatch.setattr("sys.argv", ["score", "--benchmark", str(path), "--run", "unused.csv"])
    with pytest.raises(ValueError, match="verified_semantic_tags"):
        main()
