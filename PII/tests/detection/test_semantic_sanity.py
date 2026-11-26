"""False positive and boundary regressions for spoken clinical text."""
import pytest

from pii_pipeline.policy.entity_policy import entity_from_result, resolve_entities
from pii_pipeline.recognizers.indic_identifiers import TeluguNameCueRecognizer, SpokenPlaceRecognizer, AgeRecognizer


def test_validated_tamil_case_stem_is_not_expanded_back_into_suffix():
    text = "எங்கள் லதாவுக்கு பரிசோதனை செய்யுங்கள்."
    start = text.index("லதா")
    candidate = {"entity_type": "PERSON", "start": start, "end": start + len("லதா"),
                 "score": .9, "source": "resolved", "recognizer": "IndicPersonNameGazetteer",
                 "recognition_metadata": {"boundary_evidence": "listed_tamil_case_root"}}
    selected, _ = resolve_entities(text, [candidate])
    assert [item["text"] for item in selected] == ["லதா"]


def test_joiner_before_locative_preserves_known_personal_city():
    text = "నేను హైదరాబాద్‌లో ఉంటున్నాను."
    assert [item["text"] for item in resolve(text, SpokenPlaceRecognizer(), "address")] == ["హైదరాబాద్"]


@pytest.mark.parametrize("text", [
    "హైదరాబాద్‌లో ఆసుపత్రులు ఎక్కువ.",
    "చెయ్యిలో నొప్పి ఉంది, మీరు ఎప్పుడు వచ్చారు?",
    "chest దగ్గర pain ఉంది, నేను వచ్చాను.",
])
def test_locative_and_personal_speech_do_not_create_unknown_or_body_places(text):
    assert not resolve(text, SpokenPlaceRecognizer(), "address")


def test_hometown_answer_drops_possessive_pronoun_from_value():
    text = "ఏ ఊరు మీది? వాళ్లది మధుపురం."
    assert [item["text"] for item in resolve(text, SpokenPlaceRecognizer(), "address")] == ["మధుపురం"]


def test_quotative_particle_is_not_part_of_explicit_name():
    from pii_pipeline.recognizers.indic_identifiers import IndicNameContextRecognizer
    text = "నా పేరు మేఘన అని చెప్పారు."
    assert [item["text"] for item in resolve(text, IndicNameContextRecognizer(), "PERSON")] == ["మేఘన"]


@pytest.mark.parametrize("text,value,label", [
    ("పొద్దున, మధ్యాహ్నం, రాత్రి మాత్రలు వాడండి.", "మధ్యాహ్నం", "last_name"),
    ("పొద్దున, మధ్యాహ్నం, రాత్రి మాత్రలు వాడండి.", "రాత్రి", "last_name"),
    ("ఆయన వివరంగా చెప్తాడు.", "ఆయన", "first_name"),
    ("మళ్లా పరీక్ష చేయించండి.", "మళ్లా", "last_name"),
    ("Okay, ma'am. Please take the tablet.", "ma'am", "first_name"),
    ("అలర్జీ మెడిసిన్ ఏమి లేదు.", "అలర్జీ", "full_name"),
    ("రక్తం, థైరాయిడ్, కొలెస్ట్రాల్ పరీక్షలు చేయాలి.", "థైరాయిడ్", "last_name"),
    ("రథర్ దెన్ యూరినరీ ఇన్ఫెక్షన్.", "రథర్ దెన్", "city"),
    ("గుడ్డు, చేపలు, పండ్లు తెచ్చుకొని తినండి.", "పండ్లు", "last_name"),
])
def test_clinical_or_grammatical_words_do_not_become_entities_in_short_tail(text, value, label):
    start = text.index(value)
    selected, _ = resolve_entities(text, [{"entity_type": label, "start": start,
        "end": start + len(value), "score": .99, "source": "resolved",
        "recognizer": "GLiNER2PIIPrimary"}])
    assert selected == []


@pytest.mark.parametrize("title", ["ma'am", "sir", "గారు"])
def test_name_keeps_both_words_when_trimming_trailing_honorific(title):
    text = f"Anita Rao {title} is the patient."
    value = f"Anita Rao {title}"
    selected, _ = resolve_entities(text, [{"entity_type": "full_name", "start": 0,
        "end": len(value), "score": .92, "source": "resolved", "recognizer": "GLiNER2PIIPrimary"}])
    assert [item["text"] for item in selected] == ["Anita Rao"]


def resolve(text, recognizer, label):
    return resolve_entities(text, [entity_from_result(r, "test") for r in recognizer.analyze(text, [label])])[0]


@pytest.mark.parametrize("value", [
    "ఇస్తా", "స్టార్ట్ చేసి", "ఒక టూ డేస్", "ఒక టూ డేస్ బ్యాక్",
    "chest pain", "సెట్ చేసి ఇస్తా", "ఒక", "రాసి ఇస్తా",
    "చూసి రాసి ఇస్తాను అన్నారు.", "రాసి ఇస్తాను. గ్యాస్ట్రిక్ ప్రాబ్లం.",
])
def test_telugu_name_cue_rejects_non_names(value):
    assert not resolve(f"నేను {value}", TeluguNameCueRecognizer(), "PERSON")


@pytest.mark.parametrize("name", ["విజయలక్ష్మి", "శైలజ", "శోభన", "నిఖిల్ బాబు"])
def test_telugu_name_cue_keeps_names(name):
    text = f"నా పేరు {name}."
    assert [e["text"] for e in resolve(text, TeluguNameCueRecognizer(), "PERSON")] == [name]


@pytest.mark.parametrize("text", [
    "చెయ్యి దగ్గర పెయిన్ ఉంది", "లిప్ దగ్గర స్వెల్లింగ్", "chest దగ్గర pain",
    "கை దగ్గర வலி", "ಕೈ దగ్గర ನೋವು",
])
def test_body_location_is_not_address(text):
    assert not resolve(text, SpokenPlaceRecognizer(), "address")


def test_place_relation_and_suffix_boundary():
    rec = SpokenPlaceRecognizer()
    for text in ("హైదరాబాద్ దగ్గర ఉంటాను", "నేను హైదరాబాద్‌లో ఎవరో నీకు చెప్పాను"):
        selected = resolve(text, rec, "address")
        if "దగ్గర" in text:
            assert [(e["text"], e["canonical_entity_type"]) for e in selected] == [("హైదరాబాద్", "LOCATION")]
    text = "నేను హైదరాబాద్‌లో ఎవరో నీకు చెప్పాను"
    start = text.index("హైదరాబాద్")
    candidate = {"entity_type": "address", "start": start, "end": len(text), "score": .99,
                 "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert [e["text"] for e in resolve_entities(text, [candidate])[0]] == ["హైదరాబాద్"]


def test_place_beats_cue_person_on_same_span():
    text = "నేను హైదరాబాద్ దగ్గర ఉంటాను"
    start = text.index("హైదరాబాద్")
    guesses = [
        {"entity_type": "PERSON", "start": start, "end": start + len("హైదరాబాద్"), "score": .96,
         "source": "test", "recognizer": "TeluguNameCueRecognizer",
         "recognition_metadata": {"boundary_evidence": "patient_introduction"}},
        {"entity_type": "address", "start": start, "end": start + len("హైదరాబాద్"), "score": .8,
         "source": "test", "recognizer": "SpokenPlaceRecognizer",
         "recognition_metadata": {"boundary_evidence": "personal_place_frame"}},
    ]
    assert [(e["canonical_entity_type"], e["text"]) for e in resolve_entities(text, guesses)[0]] == [("LOCATION", "హైదరాబాద్")]


def test_cue_does_not_turn_known_city_into_person():
    assert not resolve("నేను హైదరాబాద్", TeluguNameCueRecognizer(), "PERSON")


def test_multiword_place_stops_at_locative():
    text = "నేను కొత్త వసంత నగర్‌లో ఎవరో చూశాను"
    start = text.index("కొత్త")
    candidate = {"entity_type": "address", "start": start, "end": len(text), "score": .99,
                 "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert [e["text"] for e in resolve_entities(text, [candidate])[0]] == ["కొత్త వసంత నగర్"]


def test_plain_medical_word_is_not_username():
    text = "Nitro was prescribed"
    guess = {"entity_type": "username", "start": 0, "end": 5, "score": .99,
             "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert not resolve_entities(text, [guess])[0]
    text = "Username: Nitro"
    guess["start"], guess["end"] = text.index("Nitro"), len(text)
    assert resolve_entities(text, [guess])[0]


@pytest.mark.parametrize("label,text,value", [
    ("PERSON", "chest pain ఉంది", "chest pain"),
    ("PERSON", "ఇస్తా", "ఇస్తా"),
    ("address", "చెయ్యి దగ్గర పెయిన్ ఉంది", "చెయ్యి"),
    ("address", "లిప్ దగ్గర స్వెల్లింగ్", "లిప్"),
])
def test_high_model_score_still_needs_type_compatibility(label, text, value):
    start = text.index(value)
    candidate = {"entity_type": label, "start": start, "end": start + len(value),
                 "score": .99, "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert not resolve_entities(text, [candidate])[0]


def test_geographic_neighbourhood_and_structured_contact_values_survive():
    text = "నేను కుకట్‌పల్లిలో ఉంటాను"
    assert [e["text"] for e in resolve(text, SpokenPlaceRecognizer(), "address")] == ["కుకట్‌పల్లి"]
    for label, value in (("PHONE_NUMBER", "+91 9876543210"), ("EMAIL_ADDRESS", "patient@example.org")):
        text = f"Contact: {value}"
        start = text.index(value)
        candidate = {"entity_type": label, "start": start, "end": start + len(value),
                     "score": .95, "source": "test", "recognizer": "PatternRecognizer"}
        assert [e["text"] for e in resolve_entities(text, [candidate])[0]] == [value]


def test_weak_telugu_introduction_needs_identity_evidence():
    recognizer = TeluguNameCueRecognizer()
    for text in ("నేను చెప్పాను మేడం", "నేను కొన్ని టాబ్లెట్స్ రాస్తాను",
                 "నేను మామిడికాయలు తిన్నావా?", "నేను మళ్లా. 10 రోజులు"):
        assert not resolve(text, recognizer, "PERSON")
    assert [e["text"] for e in resolve("నేను రాఘవ జలారావు. 73.", recognizer, "PERSON")] == [
        "రాఘవ జలారావు"]
    assert [e["text"] for e in resolve("డాక్టర్ సత్యదేవ్ గారా.", recognizer, "PERSON")] == [
        "సత్యదేవ్"]


def test_near_requires_geographic_evidence():
    recognizer = SpokenPlaceRecognizer()
    for text in ("ఆ షోల్డర్ దగ్గర నొప్పి", "చెస్ట్ దగ్గర పెయిన్",
                 "టాబ్లెట్స్ ఉన్నాయా అన్ని దగ్గర"):
        assert not resolve(text, recognizer, "address")
    assert [e["text"] for e in resolve("ఉలవపాడు దగ్గర ఆ ఉలవపాడు", recognizer, "address")] == [
        "ఉలవపాడు"]
    assert not resolve("నెల్లూరు 12 గంటలకు ఉలవపాడు దగ్గర", recognizer, "address") or all(
        e["text"] != "గంటల" for e in resolve("నెల్లూరు 12 గంటలకు ఉలవపాడు దగ్గర", recognizer, "address"))


@pytest.mark.parametrize("label,text,value", [
    ("sensitive_date", "Since yesterday night", "yesterday"),
    ("sensitive_date", "कल फिर आना", "कल"),
    ("password", "P4P, P4D check చేయండి", "P4P"),
    ("account_number", "15000 బిల్ చేశారు", "15000"),
])
def test_ambiguous_gliner_types_need_independent_evidence(label, text, value):
    start = text.index(value)
    candidate = {"entity_type": label, "start": start, "end": start + len(value),
                 "score": .99, "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert not resolve_entities(text, [candidate])[0]


@pytest.mark.parametrize("label,text,value", [
    ("sensitive_date", "Visit on 15 August 2026", "15 August 2026"),
    ("password", "Password: P4P", "P4P"),
    ("account_number", "Account number: 15000", "15000"),
])
def test_supported_model_types_survive(label, text, value):
    start = text.index(value)
    candidate = {"entity_type": label, "start": start, "end": start + len(value),
                 "score": .99, "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert [e["text"] for e in resolve_entities(text, [candidate])[0]] == [value]


def test_telugu_age_requires_age_frame_and_keeps_numeric_boundary():
    recognizer = AgeRecognizer()
    for text in ("నాకు రెండు నెలలు తర్వాత కనిపించింది", "నాకు 3 నెలలు కనిపించింది"):
        assert not resolve(text, recognizer, "AGE")
    assert [e["text"] for e in resolve("నా వయసు 50 సంవత్సరాలు", recognizer, "AGE")] == ["50"]


@pytest.mark.parametrize("recognizer,label,text,value", [
    ("PhoneRecognizer", "PHONE_NUMBER", "BP 80. 180/70. 190", "80. 180/70. 190"),
    ("UsLicenseRecognizer", "US_DRIVER_LICENSE", "Dose 817931", "817931"),
    ("HealthPlanBeneficiaryRecognizer", "HEALTH_PLAN_BENEFICIARY_NUM", "Value 817938694", "817938694"),
])
def test_unlabelled_structured_model_values_are_not_pii(recognizer, label, text, value):
    start = text.index(value)
    guess = {"entity_type": label, "start": start, "end": start + len(value),
             "score": .99, "source": "test", "recognizer": recognizer}
    assert not resolve_entities(text, [guess])[0]


def test_public_event_names_are_not_private_pii_but_contacts_remain():
    text = "The referee saw a goal. The team scored; Messi took the penalty in Belgium. Call 9876543210."
    guesses = []
    for label, value, recognizer in (("PERSON", "Messi", "SpacyRecognizer"),
                                     ("city", "Belgium", "GLiNER2PIIPrimary"),
                                     ("PHONE_NUMBER", "9876543210", "PhoneRecognizer")):
        start = text.index(value)
        guesses.append({"entity_type": label, "start": start, "end": start + len(value),
                        "score": .9, "source": "test", "recognizer": recognizer})
    selected, rejected = resolve_entities(text, guesses)
    assert [e["text"] for e in selected] == ["9876543210"]
    assert sum(e["reason"] == "public_commentary_not_private_identifier" for e in rejected) == 2


def test_clinical_case_is_not_suppressed_by_incidental_sports_words():
    text = "Doctor: patient Ravi had pain after a goal, team match, and penalty."
    start = text.index("Ravi")
    guess = {"entity_type": "PERSON", "start": start, "end": start + 4,
             "score": .9, "source": "test", "recognizer": "GLiNER2PIIPrimary"}
    assert [e["text"] for e in resolve_entities(text, [guess])[0]] == ["Ravi"]
