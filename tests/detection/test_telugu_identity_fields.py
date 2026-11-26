"""Synthetic explicit identity fields, independent of name/place gazetteers."""
from types import SimpleNamespace
import random
import re
import string

import pytest
from presidio_analyzer import RecognizerResult

from pii_pipeline.policy.entity_policy import entity_from_result, resolve_entities
from pii_pipeline.recognizers.indic_identifiers import (
    AgeRecognizer, IndicNameContextRecognizer, SpokenPlaceRecognizer, build_indic_identifier_recognizers,
)
from tests.helpers import fake_pipeline


@pytest.mark.parametrize("name", ["నిఖిల్ బాబు", "వేణు బాబు", "అమల రావు"])
def test_explicit_full_name_does_not_trim_a_plausible_surname(name):
    text = f"నా పేరు {name}."
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == [name]


@pytest.mark.parametrize("text,expected", [
    ("నా పేరు నిఖిల్ సరే.", "నిఖిల్"),
    ("My name is Nova okay.", "Nova"),
    ("నా పేరు నిఖిల్ గారు.", "నిఖిల్"),
])
def test_explicit_names_still_stop_before_filler_and_honorifics(text, expected):
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == [expected]


def test_telugu_named_patient_cue_excludes_medication_names():
    rec = IndicNameContextRecognizer()
    text = "అనయ అనే రోగి వచ్చారు. ఈ మాత్రకు అరోమా అనే పేరు పెట్టారు."
    results = rec.analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == ["అనయ"]


@pytest.mark.parametrize("text, expected", [
    ("హలో, అనయ. నమస్తే.", "అనయ"),
    ("అనయ అని ఒక అబ్బాయి వచ్చారు.", "అనయ"),
    ("Rina ma’am is the patient.", "Rina"),
    ("This is done by Rina on 19 June 2025.", "Rina"),
])
def test_bounded_name_contexts_select_the_name_only(text, expected):
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == [expected]


def test_credited_name_needs_an_actual_calendar_month():
    assert IndicNameContextRecognizer().analyze("This is done by Tools on 19 widgets 2025.", ["PERSON"]) == []


def test_telugu_name_confirmation_preserves_question_suffix():
    text = "పేరు ఏంది అనయనా?"
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in results])
    assert [item["text"] for item in selected] == ["అనయ"]


def test_telephone_introduction_excludes_patient_role():
    recognizer = IndicNameContextRecognizer()
    text = "హలో. ఆ, నేను అనయ. హలో. నేను రోగిని."
    assert [text[r.start:r.end] for r in recognizer.analyze(text, ["PERSON"])] == ["అనయ"]


def test_listed_hindi_name_in_first_person_introduction(monkeypatch):
    monkeypatch.setattr("pii_pipeline.recognizers.indic_identifiers.load_person_name_gazetteer",
                        lambda *args: frozenset({"अवनि"}))
    text = "मैं अवनि हूँ। मैं परेशान हूँ।"
    assert [text[r.start:r.end] for r in IndicNameContextRecognizer().analyze(text, ["PERSON"])] == ["अवनि"]


def test_unlisted_trip_destination_needs_corroborating_travel_context():
    recognizer = SpokenPlaceRecognizer()
    recognizer.places, recognizer.place_pattern = frozenset(), None
    text = "కలవరికి. కలవరికి ఎప్పుడు? పాస్పోర్ట్లు సిద్ధమా? అక్కడ కలవరిలో. ఈ అమ్మాయికి జాబ్ ఉంది."
    results = recognizer.analyze(text, ["address"])
    assert [text[r.start:r.end] for r in results] == ["కలవరి"] * 3
    plain = "కలవరికి. కలవరికి ఎప్పుడు? అక్కడ కలవరిలో."
    assert recognizer.analyze(plain, ["address"]) == []


def test_residence_answer_preserves_repeated_locality_before_hospital():
    recognizer = SpokenPlaceRecognizer()
    recognizer.places, recognizer.place_pattern = frozenset(), None
    text = "ఏ ఊర్లో ఉంటారు? చింతలపేట, కలవరి. కలవరి ప్రభుత్వ ఆసుపత్రి."
    results = recognizer.analyze(text, ["address"])
    assert [text[r.start:r.end] for r in results] == ["చింతలపేట", "కలవరి", "కలవరి"]


@pytest.mark.parametrize("text, expected", [
    ("అనయ వచ్చాడు డ్యూటీకి ఈరోజు?", ["అనయ"]),
    ("అతను వచ్చాడు డ్యూటీకి ఈరోజు?", []),
    ("అనయ కాల్ చేశాడు. నాకు చెప్పాడు.", ["అనయ"]),
    ("ఫోన్ చేసింది ఎవరు? అనయ మేడం.", ["అనయ"]),
])
def test_telugu_names_attributed_to_duty_or_calls(text, expected):
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == expected


def test_destination_question_and_nonpersonal_place_description():
    recognizer = SpokenPlaceRecognizer()
    recognizer.places, recognizer.place_pattern = frozenset(), None
    question = "మళ్ళా కలవరి పోతారా? ఇక్కడే ఉంటారా?"
    assert [question[r.start:r.end] for r in recognizer.analyze(question, ["address"])] == ["కలవరి"]
    assert recognizer.analyze("ఒక ఊరు కలవరి?", ["address"]) == []


def test_origin_correction_keeps_the_confirmed_place():
    recognizer = SpokenPlaceRecognizer()
    text = "Where are you from? York. Huh? Ely, Ely. Bath. Okay."
    assert "Bath" in [text[r.start:r.end] for r in recognizer.analyze(text, ["address"])]


def test_person_named_rich_is_distinct_from_a_diet_adjective():
    for text, expected in [("Rich diet was discussed.", []), ("Patient name: Rich.", ["Rich"])]:
        start = text.index("Rich")
        candidate = RecognizerResult("PERSON", start, start + 4, .9,
                                     recognition_metadata={"recognizer_name": "SpacyRecognizer"})
        picked, _ = resolve_entities(text, [entity_from_result(candidate, "resolved")])
        assert [entity["text"] for entity in picked] == expected


def test_telugu_listed_place_repetition_requires_a_personal_frame():
    rec = SpokenPlaceRecognizer()
    rec.places = frozenset({"అమరావతి"})
    rec.place_pattern = re.compile(r"(అమరావతి)(?=$|\s|లో)")
    text = "వెళ్తున్నా, అమరావతిలో. అమరావతిలో."
    results = rec.analyze(text, ["address"])
    assert [text[r.start:r.end] for r in results] == ["అమరావతి", "అమరావతి"]
    assert rec.analyze("వాతావరణం అమరావతిలో బాగుంది.", ["address"]) == []


def test_telugu_landmark_uses_exact_listed_place_after_an_adverb():
    rec = SpokenPlaceRecognizer()
    rec.places = frozenset({"అమరావతి"})
    rec.place_pattern = re.compile(r"(అమరావతి)(?=$|\s)")
    text = "సరిగ్గా అమరావతి దగ్గర ఆపాము."
    results = rec.analyze(text, ["address"])
    assert [text[r.start:r.end] for r in results] == ["అమరావతి"]


def test_age_weight_pair_keeps_weight_and_duration_visible():
    rec = AgeRecognizer()
    text = "41 ఇయర్స్, 68 కేజీలు. I was ill for 3 years, 10 kg was lost."
    results = rec.analyze(text, ["AGE"])
    assert [text[r.start:r.end] for r in results] == ["41"]


@pytest.mark.parametrize("subject,predicate", [
    ("నేను", "నివసిస్తున్నాను"), ("మేము", "నివసిస్తున్నాము"),
    ("మేము", "నివసిస్తున్నాం"),
])
@pytest.mark.parametrize("place", ["నవచంద్రపురం", "కొత్త వసంత నగర్"])
def test_personal_residence_finds_unlisted_places_and_preserves_locative_suffix(subject, predicate, place):
    recognizer = SpokenPlaceRecognizer()
    recognizer.places, recognizer.place_pattern = frozenset(), None
    text = f"{subject} {place}లో {predicate}."
    results = recognizer.analyze(text, ["address"])
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in results])
    assert [e["text"] for e in selected] == [place]
    pipe = fake_pipeline()
    pipe.orchestrator.analyze = lambda text, **kw: SimpleNamespace(resolved=results, needs_fallback=[])
    assert pipe.run(text)["redacted_text"] == f"{subject} <ADDRESS_1>లో {predicate}."


@pytest.mark.parametrize("text", [
    "నేను ఇక్కడ నివసిస్తున్నాను.", "నేను ఇంట్లో నివసిస్తున్నాను.",
    "నేను ఆసుపత్రిలో నివసిస్తున్నాను.", "మందు నవచంద్రపురంలో దొరుకుతుంది.",
    "నేను నవచంద్రపురంలో మందు కొన్నాను.",
])
def test_generic_locations_and_nonresidence_statements_do_not_create_addresses(text):
    recognizer = SpokenPlaceRecognizer()
    recognizer.places, recognizer.place_pattern = frozenset(), None
    assert recognizer.analyze(text, ["address"]) == []


@pytest.mark.parametrize("field", ["హెల్త్ ఐడి", "హెల్త్ ఐడీ", "health ID", "ABHA number"])
@pytest.mark.parametrize("value", ["7621-3084-9056", "HC-472910", "7621 3084 9056"])
def test_labelled_health_id_beats_bank_or_government_guesses(field, value):
    text = f"{field}: {value}."
    recognizers = build_indic_identifier_recognizers()
    results = [r for rec in recognizers if rec.name == "LabelledHealthIdRecognizer"
               for r in rec.analyze(text, ["MEDICAL_ID"])]
    assert [text[r.start:r.end] for r in results] == [value]
    start = text.index(value)
    for label in ("account_number", "bank_account", "GOVERNMENT_ID", "phone_number"):
        guess = RecognizerResult(label, start, start+len(value), .999,
                                 recognition_metadata={"recognizer_name": "GLiNER2PIIPrimary"})
        selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in [guess, *results]])
        assert [(e["entity_type"], e["text"]) for e in selected] == [("MEDICAL_ID", value)]


@pytest.mark.parametrize("text", [
    "health ID is unavailable.", "health ID: none", "health ID: 45 mg.",
    "bank account 7621-3084-9056.", "phone number 7621-3084-9056.",
    "7621-3084-9056", "మందు 500 mg.",
])
def test_health_id_requires_an_explicit_field_and_an_identifier_value(text):
    for rec in build_indic_identifier_recognizers():
        if rec.name == "LabelledHealthIdRecognizer":
            assert rec.analyze(text, ["MEDICAL_ID"]) == []


@pytest.mark.parametrize("name", ["నిఖిల్ బాబు", "అవని", "మను", "వేణు కిరణ్"])
@pytest.mark.parametrize("ending", ["", " ని", " ను"])
def test_telugu_self_introduction_needs_demographic_context_and_preserves_name_words(name, ending, monkeypatch):
    monkeypatch.setattr("pii_pipeline.recognizers.indic_identifiers.load_person_name_gazetteer", lambda *args, **kwargs: frozenset())
    text = f"నేను {name}{ending}. నా వయస్సు 36."
    results = IndicNameContextRecognizer().analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == [name]


@pytest.mark.parametrize("text", [
    "నేను రోగిని. నా వయసు 36.", "నేను నర్సును. నా వయసు 36.",
    "నేను డాక్టర్ ని. నా వయస్సు 36.", "నేను బాగున్నాను. నా వయసు 36.",
    "నేను మందులు వాడుతున్నాను. నా వయస్సు 36.",
    "నేను చికిత్సలో. నా వయసు 36.",
    "నేను నవచంద్రపురం నుండి వచ్చాను. నా వయసు 36.",
    "నేను నవచంద్రపురంలో ఉంటాను. నా వయసు 36.",
    "నేను ఏదో చెప్పాను.",
])
def test_telugu_clinical_predicates_and_origin_clauses_are_not_person_names(text):
    assert IndicNameContextRecognizer().analyze(text, ["PERSON"]) == []


@pytest.mark.parametrize("relation", [" నుండి వచ్చాను", "నుంచి వస్తున్నాను", " నుంచి", "నుండి", "లో ఉంటున్నాను", "లో ఉంటాను", "\u200cలో నివసిస్తున్నాను"])
def test_telugu_origin_and_residence_do_not_need_listed_places(relation):
    rec = SpokenPlaceRecognizer()
    rec.places, rec.place_pattern = frozenset(), None
    place = "నవచంద్రపురం"
    text = f"నేను {place}{relation}."
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in rec.analyze(text, ["address"])])
    assert [e["text"] for e in selected] == [place]
    assert selected[0]["start"] == text.index(place)
    assert selected[0]["end"] == text.index(place)+len(place)


@pytest.mark.parametrize("text", [
    "నేను ఇంటి నుండి వచ్చాను.", "నేను చికిత్స నుండి వచ్చాను.",
    "నేను ఆసుపత్రిలో ఉంటున్నాను.", "మందు నవచంద్రపురం నుండి వచ్చింది.",
    "నేను నొప్పి నుండి కోలుకుంటున్నాను.",
])
def test_telugu_generic_origin_and_medication_sources_are_not_addresses(text):
    rec = SpokenPlaceRecognizer()
    rec.places, rec.place_pattern = frozenset(), None
    assert rec.analyze(text, ["address"]) == []


@pytest.mark.parametrize("field", ["మెడికల్ నంబర్", "మెడికల్ నెంబర్", "వైద్య రికార్డు సంఖ్య", "వైద్య రికార్డ్ నంబర్"])
def test_telugu_medical_number_overrides_phone_guess(field):
    value = "843-207-56192"
    text = f"{field}: {value}."
    results = [r for rec in build_indic_identifier_recognizers() if rec.name == "LabelledHealthIdRecognizer"
               for r in rec.analyze(text, ["MEDICAL_ID"])]
    start = text.index(value)
    phone = RecognizerResult("phone_number", start, start+len(value), .999,
                             recognition_metadata={"recognizer_name": "GLiNER2PIIPrimary"})
    selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in [phone, *results]])
    assert [(e["entity_type"], e["text"]) for e in selected] == [("MEDICAL_ID", value)]


@pytest.mark.parametrize("field", ["పాన్ నంబర్", "పాన్ నెంబర్"])
@pytest.mark.parametrize("value,valid", [("AZTPQ4821L", True), ("qwert12345", False)])
def test_telugu_pan_field_masks_valid_and_mistyped_values(field, value, valid):
    text = f"{field} {value}."
    results = [r for rec in build_indic_identifier_recognizers() if rec.name == "LabelledPanRecognizer"
               for r in rec.analyze(text, ["tax_id"])]
    assert [text[r.start:r.end] for r in results] == [value]
    assert results[0].recognition_metadata["format_valid"] is valid


def test_300_generated_telugu_identity_rows_without_name_or_place_lookups(monkeypatch):
    # A fixed seed makes this regression reproducible. These generated values
    # are test inputs only, never entries in runtime rules or gazetteers.
    rng = random.Random(2026101801)
    monkeypatch.setattr("pii_pipeline.recognizers.indic_identifiers.load_person_name_gazetteer", lambda *args, **kwargs: frozenset())
    recs = build_indic_identifier_recognizers()
    names = next(r for r in recs if r.name == "IndicNameContextRecognizer")
    places = next(r for r in recs if r.name == "SpokenPlaceRecognizer")
    places.places, places.place_pattern = frozenset(), None
    pan = next(r for r in recs if r.name == "LabelledPanRecognizer")
    medical = next(r for r in recs if r.name == "LabelledHealthIdRecognizer")
    age_rec = AgeRecognizer()

    def word():
        return "".join(rng.choices(("క", "మ", "న", "ర", "వ", "స", "ల", "త"), k=rng.randint(3, 5)))

    for i in range(300):
        name = f"{word()} {word()}"
        place = word()+"పురం"
        age = str(rng.randint(18, 90))
        pan_value = "".join(rng.choices(string.ascii_uppercase, k=5))+"".join(rng.choices(string.digits, k=4))+rng.choice(string.ascii_uppercase)
        medical_value = f"{rng.randint(200,999)}-{rng.randint(200,999)}-{rng.randint(10000,99999)}"
        intro = (f"నా పేరు {name}.", f"నేను {name}.", f"నేను {name} ని.")[i % 3]
        location = (f"నేను {place} నుండి వచ్చాను.", f"నేను {place} నుంచి.",
                    f"నేను {place}లో ఉంటున్నాను.", f"నేను {place}లో నివసిస్తున్నాను.")[i % 4]
        text = f"{intro} నా వయస్సు {age}. {location} పాన్ నంబర్ {pan_value}. మెడికల్ నంబర్ {medical_value}."
        results = [r for rec in (names, age_rec, places, pan, medical)
                   for r in rec.analyze(text, list(rec.supported_entities))]
        selected, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in results])
        assert [(e["text"], e["canonical_entity_type"]) for e in selected] == [
            (name, "PERSON"), (age, "AGE"), (place, "LOCATION"),
            (pan_value, "TAX_ID"), (medical_value, "MEDICAL_ID"),
        ], (i, text, selected)
