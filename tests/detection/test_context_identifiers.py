"""Multilingual names, ages, dates and personal places with clinical negatives."""
from __future__ import annotations

import pytest

from pii_pipeline.recognizers.clinical_calendar import ClinicalCalendarDateRecognizer
from pii_pipeline.policy.entity_policy import entity_from_result, parse_semantic_tags, resolve_entities
from pii_pipeline.recognizers.indic_identifiers import (
    AgeRecognizer,
    IndicNameContextRecognizer,
    SpokenPlaceRecognizer,
)


def candidate(text, value, label="PERSON", **extra):
    start = text.index(value)
    return dict(entity_type=label, start=start, end=start+len(value), score=.98,
                recognizer="GLiNER2PIIPrimary", **extra)


def selected(text, recognizer):
    values, _ = resolve_entities(text, [entity_from_result(result, "resolved") for result in
                                      recognizer.analyze(text, recognizer.supported_entities)])
    return [value["text"] for value in values]


@pytest.mark.parametrize("text,value", [
    ("आपका नाम क्या है? आपकी उम्र कितनी है?", "क्या"),
    ("Ready. Give it to Anagha.", "Ready"),
    ("Make it Smiley, interactive.", "Smiley"),
    ("ప్రిన్సిపల్ గారు వచ్చారు.", "ప్రిన్సిపల్"),
    ("మా ఫాదర్ గారు మంత్రి గారు.", "మంత్రి"),
    ("You can take Tolo or Paracetamol.", "Tolo"),
    ("Last night I took Lolo, then felt fine.", "Lolo"),
    ("Take ExampleDrug or paracetamol.", "ExampleDrug"),
    ("Take Tolo or Paracetamol.", "Tolo or Paracetamol"),
    ("తుమ్ములు వస్తా ఉంటాయి ఏమైనా? ఆహా, రావు.", "రావు"),
    ("Mr. అన్నా, నాకు సంబంధం లేదు.", "Mr. అన్నా"),
    ("So here I am. అ ఏం ఇబ్బంది అని వచ్చారు?", "So here I am"),
])
@pytest.mark.parametrize("imported", [False, True])
def test_ordinary_roles_questions_and_medication_across_sources(text, value, imported):
    value = candidate(text, value)
    if imported:
        proposals, _ = parse_semantic_tags([value])
    else:
        proposals = [value]
    assert not resolve_entities(text, proposals)[0]


@pytest.mark.parametrize("value", ["Ready", "Principal", "Tolo", "రావు", "क्या"])
def test_explicit_person_name_values_survive(value):
    text = "Name: " + value + "."
    assert resolve_entities(text, [candidate(text, value)])[0][0]["text"] == value


@pytest.mark.parametrize("pronoun", ["నేను", "ನಾನು", "मैं", "நான்", "আমি", "ഞാൻ"])
def test_first_person_pronouns_are_not_person_names_but_explicit_fields_survive(pronoun):
    speech = f"హలో డాక్టర్, {pronoun} ఒక patient."
    assert not resolve_entities(speech, [candidate(speech, pronoun)])[0]
    identity = f"Name: {pronoun}."
    assert resolve_entities(identity, [candidate(identity, pronoun)])[0][0]["text"] == pronoun


def test_clinical_allen_eponym_is_not_a_person_but_identity_context_survives():
    clinical = "The patient has a positive Allen's test."
    assert not resolve_entities(clinical, [candidate(clinical, "Allen")])[0]
    identity = "Dr. Allen's test results arrived."
    assert resolve_entities(identity, [candidate(identity, "Allen")])[0][0]["text"] == "Allen"


@pytest.mark.parametrize(
    'text,expected',
    [
        pytest.param('నేను రియా, మీ వైద్య సహాయకురాలు.', ['రియా'], id='supported_unfamiliar_names_and_role_negatives-1'),
        pytest.param('నేను జయనిక, మీ వైద్య సహాయకురాలు.', ['జయనిక'], id='supported_unfamiliar_names_and_role_negatives-2'),
        pytest.param('ನಾನು ನಿರವಿಕಾ, ನಿಮ್ಮ ವೈದ್ಯ ಸಹಾಯಕ.', ['ನಿರವಿಕಾ'], id='supported_unfamiliar_names_and_role_negatives-3'),
        pytest.param('ಮಧುಸೂದನ್. అ ఏం ఇబ్బంది అని వచ్చారు అప్పుడు?', ['ಮಧುಸೂದನ್'], id='supported_unfamiliar_names_and_role_negatives-4'),
        pytest.param('సుబ్రహ్మణ్యం ఆ. చెప్పండి. ఏమైంది మీకు?', ['సుబ్రహ్మణ్యం'], id='supported_unfamiliar_names_and_role_negatives-5'),
        pytest.param('டாக்டர் நிலவனன் அதே தான் prescribe பண்ணிருக்கார்.', ['நிலவனன்'], id='supported_unfamiliar_names_and_role_negatives-6'),
        pytest.param('आपका नाम क्या है? Okay. आपकी age कितनी है?', [], id='supported_unfamiliar_names_and_role_negatives-7'),
        pytest.param('నేను stable, మీ వైద్య సహాయకురాలు.', [], id='supported_unfamiliar_names_and_role_negatives-8'),
        pytest.param('Headache. ఏం ఇబ్బంది అని వచ్చారు?', [], id='supported_unfamiliar_names_and_role_negatives-9'),
        pytest.param('ప్రిన్సిపల్ గారు వచ్చారు.', [], id='supported_unfamiliar_names_and_role_negatives-10'),
        pytest.param('So here I am. అ ఏం ఇబ్బంది అని వచ్చారు?', [], id='supported_unfamiliar_names_and_role_negatives-11'),
        pytest.param('నీ పేరేంటి? సంజయ్. మొబైల్.', ['సంజయ్'], id='name_questions_and_unlisted_honorific_names-1'),
        pytest.param('పేరు ఏంది? సులోచనమ్మ.', ['సులోచనమ్మ'], id='name_questions_and_unlisted_honorific_names-2'),
        pytest.param('అబ్బాయి పేరు ఏంటమ్మా? దినేష్.', ['దినేష్'], id='name_questions_and_unlisted_honorific_names-3'),
        pytest.param('உங்களோட பேரு? రీతికా.', ['రీతికా'], id='name_questions_and_unlisted_honorific_names-4'),
        pytest.param('நான் இப்ப பேரு வெங்கடேஷ் కదండీ.', ['வெங்கடேஷ்'], id='name_questions_and_unlisted_honorific_names-5'),
        pytest.param('హలో, నజరు గారు. మందులు వాడండి.', ['నజరు'], id='name_questions_and_unlisted_honorific_names-6'),
        pytest.param('తెలుసు, రామన్న గారు సార్.', ['రామన్న'], id='name_questions_and_unlisted_honorific_names-7'),
        pytest.param('డాక్టర్ పద్మ ఉంటారు.', ['పద్మ'], id='name_questions_and_unlisted_honorific_names-8'),
        pytest.param('Name: Speaker.', ['Speaker'], id='name_questions_and_unlisted_honorific_names-9'),
    ],
)
def test_supported_unfamiliar_names_and_role_negatives(text, expected):
    assert selected(text, IndicNameContextRecognizer()) == expected


@pytest.mark.parametrize(
    'text,expected',
    [
        pytest.param('మీ ఊరు ఎక్కడ? నెల్లూరే సార్.', ['నెల్లూరే'], id='personal_multilingual_places_and_whole_word_guards-1'),
        pytest.param('మీ ఊరు ఎక్కడ? మల్లివరం సార్.', ['మల్లివరం'], id='personal_multilingual_places_and_whole_word_guards-2'),
        pytest.param('તમારું ગામ કયું છે? વડોદરા.', ['વડોદરા'], id='personal_multilingual_places_and_whole_word_guards-3'),
        pytest.param('હું, અહ, વાઈઝકથી આવ્યો હતો. હું બેંગલોર જઈને આવ્યો.', ['વાઈઝક', 'બેંગલોર'], id='personal_multilingual_places_and_whole_word_guards-4'),
        pytest.param('मैं 2 दिन पहले दिल्ली ट्रैवल किया था।', ['दिल्ली'], id='personal_multilingual_places_and_whole_word_guards-5'),
        pytest.param('మేము బెంగళూరులో సార్ ఉన్నాము.', ['బెంగళూరు'], id='personal_multilingual_places_and_whole_word_guards-6'),
        pytest.param('తిరుపతి నుంచి వస్తున్నాను నేను.', ['తిరుపతి'], id='personal_multilingual_places_and_whole_word_guards-7'),
        pytest.param('హోటల్ లో పని చేస్తావా? వెంకట్ రెడ్డిపల్లి. ఆధూర్పల్లి దగ్గర.', ['వెంకట్ రెడ్డిపల్లి', 'ఆధూర్పల్లి'], id='personal_multilingual_places_and_whole_word_guards-8'),
        pytest.param('Delhi is a city. బెంగళూరులో వర్షం పడుతోంది.', [], id='personal_multilingual_places_and_whole_word_guards-9'),
        pytest.param('This pill comes from Delhi.', [], id='personal_multilingual_places_and_whole_word_guards-10'),
        pytest.param('వెంకట్ రెడ్డిపల్లి ఒక గ్రామం.', [], id='personal_multilingual_places_and_whole_word_guards-11'),
        pytest.param('I visited Delhiwood.', [], id='personal_multilingual_places_and_whole_word_guards-12'),
        pytest.param('હું બીમાર છું. બેંગલોર એક શહેર છે.', [], id='personal_multilingual_places_and_whole_word_guards-13'),
        pytest.param('నేను ఆంధ్రాలో చూయించుకున్నాను.', ['ఆంధ్రా'], id='personal_places_are_distinct_from_generic_mentions-1'),
        pytest.param('मैं पुणे गया, पुणे में इलाज कराया।', ['पुणे', 'पुणे'], id='personal_places_are_distinct_from_generic_mentions-2'),
        pytest.param('I travelled to Pune.', ['Pune'], id='personal_places_are_distinct_from_generic_mentions-3'),
        pytest.param('Pune is a city. Andhra is a state.', [], id='personal_places_are_distinct_from_generic_mentions-4'),
        pytest.param('मैं ठीक हो गया।', [], id='personal_places_are_distinct_from_generic_mentions-5'),
        pytest.param('This pill comes from Pune.', [], id='personal_places_are_distinct_from_generic_mentions-6'),
    ],
)
def test_personal_multilingual_places_and_whole_word_guards(text, expected):
    assert selected(text, SpokenPlaceRecognizer()) == expected


@pytest.mark.parametrize(
    'text,expected',
    [
        pytest.param('Blood reports investigations done on 15 ఆగస్టు 2026. Hb 7.9.', ['15 ఆగస్టు 2026'], id='clinical_dates_and_history_years-1'),
        pytest.param('బ్లడ్ రిపోర్ట్స్ ఇన్వెస్టిగేషన్స్ డన్ ఆన్ 15 ఆగస్టు 2026. హీమోగ్లోబిన్ 7.9.', ['15 ఆగస్టు 2026'], id='clinical_dates_and_history_years-2'),
        pytest.param('బ్లడ్ రిపోర్ట్స్ డన్ ఆన్ 15 ఆగస్టు 2026. HbA1c 6.1%, ఎకో డన్ ఆన్ 15 ఆగస్టు 2026, EF 45%.', ['15 ఆగస్టు 2026', '15 ఆగస్టు 2026'], id='clinical_dates_and_history_years-3'),
        pytest.param('எக்கோட நான் 9 ஆகஸ்ட் 2026. EF 35%.', ['9 ஆகஸ்ட் 2026'], id='clinical_dates_and_history_years-4'),
        pytest.param('మార్చ్ లో ఆపరేషన్ చేశాను.', ['మార్చ్'], id='clinical_dates_and_history_years-5'),
        pytest.param('జనవరిలోనే మందులు వాడేది.', ['జనవరి'], id='clinical_dates_and_history_years-6'),
        pytest.param('జనవరిలో ఆపరేషన్ చేశాను.', ['జనవరి'], id='clinical_dates_and_history_years-7'),
        pytest.param('బీపీ ఉంది. ఎన్ని సంవత్సరాల నుంచి? 2018 నుంచి వాడుతున్నాను.', ['2018'], id='clinical_dates_and_history_years-8'),
        pytest.param('బీపీ ఉంది. ఎన్ని సంవత్సరాల నుంచి? 2007 నుంచి వాడుతున్నాను.', ['2007'], id='clinical_dates_and_history_years-9'),
        pytest.param('ఎన్ని సంవత్సరాల నుంచి? 2018 నుంచి పని చేస్తున్నాను.', [], id='clinical_dates_and_history_years-10'),
        pytest.param('January weather is cold. Surgery advice is general.', [], id='clinical_dates_and_history_years-11'),
        pytest.param('Visit on 15 ఆగస్టు 2026 mg.', [], id='clinical_dates_and_history_years-12'),
        pytest.param('ట్రీట్మెంట్ మార్చి కొత్త టాబ్లెట్ ఇచ్చారు.', [], id='clinical_dates_and_history_years-13'),
        pytest.param('తొమ్మిదో తారీకు నొప్పుల సంగతి రాశాను.', ['తొమ్మిదో'], id='spoken_calendar_days_and_explicit_clinical_years-1'),
        pytest.param('నాలుగో తేదీ. ఈ నెల నాలుగో.', ['నాలుగో', 'నాలుగో'], id='spoken_calendar_days_and_explicit_clinical_years-2'),
        pytest.param('తొమ్మిది. తొమ్మిది కాబట్టి ఆ డేట్ కి కుదరదు.', ['తొమ్మిది', 'తొమ్మిది'], id='spoken_calendar_days_and_explicit_clinical_years-3'),
        pytest.param('11వ తేదీకి పోవాలి.', ['11'], id='spoken_calendar_days_and_explicit_clinical_years-4'),
        pytest.param('Echo done on 15 August 2026, EF 62%.', ['15 August 2026'], id='spoken_calendar_days_and_explicit_clinical_years-5'),
        pytest.param('Angio is done on yesterday, August 20.', ['August 20'], id='spoken_calendar_days_and_explicit_clinical_years-6'),
        pytest.param('2014 నుంచి షుగర్ ఉందా? 2014 నుంచి బీపీ కూడా ఉంది.', ['2014', '2014'], id='spoken_calendar_days_and_explicit_clinical_years-7'),
        pytest.param('Diabetes since 2014.', ['2014'], id='spoken_calendar_days_and_explicit_clinical_years-8'),
        pytest.param('Last menstrual period is జనవరి 15.', ['జనవరి 15'], id='clinical_calendar_values_and_spoken_visit_days-1'),
        pytest.param('లాస్ట్ మెన్స్ట్రల్ పీరియడ్ ఇస్. మార్చ్ 20.', ['మార్చ్ 20'], id='clinical_calendar_values_and_spoken_visit_days-2'),
        pytest.param('Last visit was June 17.', ['June 17'], id='clinical_calendar_values_and_spoken_visit_days-3'),
        pytest.param('Last visit was 17 June 2025.', ['17 June 2025'], id='clinical_calendar_values_and_spoken_visit_days-4'),
        pytest.param('Treatment started in ఏప్రిల్.', ['ఏప్రిల్'], id='clinical_calendar_values_and_spoken_visit_days-5'),
        pytest.param('జనవరి నుంచి మందు తీసుకుంటున్నాను.', ['జనవరి'], id='telugu-spoken-medication-start-month'),
        pytest.param('ఫిబ్రవరిలో మందు తాగడం మొదలుపెట్టాను.', ['ఫిబ్రవరి'], id='telugu-medication-month-with-locative'),
        pytest.param('పదో. పదో కాబోలు. పదో తారీకు వచ్చాను.', ['పదో', 'పదో', 'పదో'], id='telugu-echoed-calendar-day'),
        pytest.param('మేము 19 డిసెంబర్ చేశాం.', ['19 డిసెంబర్'], id='telugu-calendar-date-before-care-action'),
        pytest.param('జనవరి మొన్నే చేయించుకున్నావ్.', ['జనవరి'], id='clinical_calendar_values_and_spoken_visit_days-6'),
        pytest.param('20వ తారీకు తీసుకున్నావు. 21 తీసుకున్నావు. 28 వచ్చావు.', ['20', '21', '28'], id='clinical_calendar_values_and_spoken_visit_days-7'),
        pytest.param('ఏంటి, 25. వన్ ఇయర్? 25 డేట్ వి స్టార్టెడ్.', ['25', '25'], id='clinical_calendar_values_and_spoken_visit_days-8'),
        pytest.param('20వ తారీకు. 20, 26, 5 రాశాను నేను. 12/5 వచ్చావు.', ['20', '20', '26', '5'], id='clinical_calendar_values_and_spoken_visit_days-9'),
        pytest.param('Date of birth: August 19, 1996.', ['August 19, 1996'], id='clinical_calendar_values_and_spoken_visit_days-10'),
    ],
)
def test_clinical_dates_and_history_years(text, expected):
    assert selected(text, ClinicalCalendarDateRecognizer()) == expected


def test_clinical_history_year_beats_unsupported_birth_label():
    text = "బీపీ ఉంది. ఎన్ని సంవత్సరాల నుంచి? 2018 నుంచి వాడుతున్నాను."
    proposals = [candidate(text, "2018", "date_of_birth")]
    proposals += [entity_from_result(r, "resolved") for r in
                  ClinicalCalendarDateRecognizer().analyze(text, ["sensitive_date"])]
    selected_values, _ = resolve_entities(text, proposals)
    assert len(selected_values) == 1 and selected_values[0]["entity_type"] == "sensitive_date"


def test_actual_birth_date_is_preserved():
    text = "Date of birth: August 19, 1996."
    values, _ = resolve_entities(text, [entity_from_result(r, "resolved") for r in
        ClinicalCalendarDateRecognizer().analyze(text, ["date_of_birth", "sensitive_date"])])
    assert len(values) == 1 and values[0]["entity_type"] == "date_of_birth"


@pytest.mark.parametrize(
    'text,expected',
    [
        pytest.param('Age of 13 ல இருந்து இருக்கு. 13. Good.', ['13', '13'], id='local_age_corrections_do_not_cross_doses_or_new_topics-1'),
        pytest.param('ఎంత వయసు మీకు? 30. 30, 54 ఉన్నాను సార్.', ['30', '30', '54'], id='local_age_corrections_do_not_cross_doses_or_new_topics-2'),
        pytest.param('ఎంత వయసు మీకు? 30. 30, 5 mg.', ['30', '30'], id='local_age_corrections_do_not_cross_doses_or_new_topics-3'),
        pytest.param('Age of 13. Take 13 mg.', ['13'], id='local_age_corrections_do_not_cross_doses_or_new_topics-4'),
        pytest.param('Age of 13. Pain for 13 years.', ['13'], id='local_age_corrections_do_not_cross_doses_or_new_topics-5'),
        pytest.param('ఆ, 72 సంవత్సరాలు వచ్చాయి.', ['72'], id='explicit_age_answers_suffixes_and_corrections-1'),
        pytest.param('మా నాన్నకు 70 సంవత్సరాల వయసులో వచ్చింది.', ['70'], id='explicit_age_answers_suffixes_and_corrections-2'),
        pytest.param('మీ వయసు ఎంత? 62 దాకా. 62.', ['62', '62'], id='explicit_age_answers_suffixes_and_corrections-3'),
        pytest.param('62 ఏజ్. మందులు వాడండి.', ['62'], id='explicit_age_answers_suffixes_and_corrections-4'),
        pytest.param('ఎంత వయసు? ఎంత? 41. 40? సరే, 5.', ['41', '40'], id='explicit_age_answers_suffixes_and_corrections-5'),
        pytest.param('ఏజ్ ఎంత మీకు? 58 వచ్చింది, సార్. 58.', ['58', '58'], id='explicit_age_answers_suffixes_and_corrections-6'),
        pytest.param('మీ పేషెంట్ ఏజ్ ఎంత, అమ్మా? Sonia 35 years, sir. 35 years, సార్.', ['35', '35'], id='explicit_age_answers_suffixes_and_corrections-7'),
        pytest.param('आपकी age कितनी है अभी? 26।', ['26'], id='explicit_age_answers_suffixes_and_corrections-8'),
        pytest.param('వயசு? 24.', [], id='explicit_age_answers_suffixes_and_corrections-9'),
        pytest.param('வயசு? 24.', ['24'], id='explicit_age_answers_suffixes_and_corrections-10'),
        pytest.param('47. 47 ఇయర్స్. ఏజ్ ఏజ్ కాదు. ఆ 47.', ['47', '47', '47'], id='age_answers_repetitions_and_historical_ages-1'),
        pytest.param('ఎంత వయసు మీకు? 36.', ['36'], id='age_answers_repetitions_and_historical_ages-2'),
        pytest.param('ఎంత వయసు నీకు? 42.', ['42'], id='age_answers_repetitions_and_historical_ages-3'),
        pytest.param('My age is 40. Age of menarche at 14 years.', ['40', '14'], id='age_answers_repetitions_and_historical_ages-4'),
        pytest.param('My menstrual history: menarche is 12. Years.', ['12'], id='age_answers_repetitions_and_historical_ages-5'),
        pytest.param('మెన్స్ట్రల్ హిస్టరీ: ఏజ్ అట్ 16 ఇయర్స్.', ['16'], id='age_answers_repetitions_and_historical_ages-6'),
        pytest.param('I matured at the age of 13.', ['13'], id='age_answers_repetitions_and_historical_ages-7'),
        pytest.param('My menstrual history is 8 of the menarche is 15. Years.', ['15'], id='age_answers_repetitions_and_historical_ages-8'),
        pytest.param("Baby's age is 4 months.", ['4'], id='age_answers_repetitions_and_historical_ages-9'),
        pytest.param('Age is 15 days.', ['15'], id='age_answers_repetitions_and_historical_ages-10'),
        pytest.param('How old are you? 6 months.', ['6'], id='age_answers_repetitions_and_historical_ages-11'),
        pytest.param('I am 47 years old. Dose 47 mg.', ['47'], id='age_answers_repetitions_and_historical_ages-12'),
        pytest.param('I am 25 years old. Pain for 25 days.', ['25'], id='age_answers_repetitions_and_historical_ages-13'),
    ],
)
def test_local_age_corrections_do_not_cross_doses_or_new_topics(text, expected):
    assert selected(text, AgeRecognizer()) == expected


def test_repeated_script_name_keeps_two_exact_original_occurrences():
    text = "My name is శ్రీవల్లి ಶ್ರೀವಲ್ಲಿ."
    broad = candidate(text, "శ్రీవల్లి ಶ್ರೀವಲ್ಲಿ")
    fragments = [dict(candidate(text, value), recognizer="IndicPersonNameGazetteer")
                 for value in ["శ్రీవల్లి", "ಶ್ರೀವಲ್ಲಿ"]]
    values, _ = resolve_entities(text, [broad]+fragments)
    assert [r["text"] for r in values] == ["శ్రీవల్లి", "ಶ್ರೀವಲ್ಲಿ"]
    assert all(text[r["start"]:r["end"]] == r["text"] for r in values)
    assert resolve_entities(text, list(reversed([broad]+fragments)))[0] == values
    assert broad["end"] == len("My name is శ్రీవల్లి ಶ್ರೀವಲ್ಲಿ")


@pytest.mark.parametrize("value", ["శ్రీవల్లి ರೆಡ್ಡಿ", "Ananya Rao", "R. Kumar", "రామ రామ"])
def test_different_given_name_surname_and_same_script_are_not_split(value):
    text = "Name: " + value + "."
    values, _ = resolve_entities(text, [candidate(text, value)])
    assert [r["text"] for r in values] == [value]


def test_split_name_respects_script_specific_acceptance_threshold():
    text = "My name is శ్రీవల్లి ಶ್ರೀವಲ್ಲಿ."
    values, rejected = resolve_entities(text, [candidate(text, "శ్రీవల్లి ಶ್ರೀವಲ್ಲಿ")],
                                       score_thresholds={"person:kannada":.99})
    assert [r["text"] for r in values] == ["శ్రీవల్లి"]
    assert any(r["reason"] == "below_min_score" and r["text"] == "ಶ್ರೀವಲ್ಲಿ" for r in rejected)


def test_initial_i_remains_part_of_a_supported_complete_name():
    text = "He met I. Kumar yesterday."
    whole = candidate(text, "I. Kumar")
    surname = dict(candidate(text, "Kumar"), recognizer="IndicPersonNameGazetteer")
    values, _ = resolve_entities(text, [whole, surname])
    assert [value["text"] for value in values] == ["I. Kumar"]


@pytest.mark.parametrize("text", [
    "ఈ పేరు ఏంటి? కదా?", "పేరు వస్తుంది.", "పేరు మర్చిపోయాను.",
    "పేరు మారుద్ది. మందులు అదే ఉంటాయి.", "మరో గారు ఉన్నారా?",
    "డాక్టర్ గారు చెప్పండి.", "This is Speaker 1, my interview.",
])
def test_name_rules_keep_questions_predicates_and_generic_roles(text):
    assert selected(text, IndicNameContextRecognizer()) == []


@pytest.mark.parametrize(
    'text',
    [
        pytest.param('షుగర్ 40 సంవత్సరాల నుంచి ఉంది.', id='new_age_rules_preserve_doses_and_durations-1'),
        pytest.param('Symptoms for 72 years.', id='new_age_rules_preserve_doses_and_durations-2'),
        pytest.param('ఎంత వయసు? 5 mg. 10 mg.', id='new_age_rules_preserve_doses_and_durations-3'),
        pytest.param('ఏజ్ ఎంత మీకు? 58 kg.', id='new_age_rules_preserve_doses_and_durations-4'),
        pytest.param('I visited 72 years ago.', id='new_age_rules_preserve_doses_and_durations-5'),
        pytest.param('Pain since 3 years.', id='age_rules_keep_doses_durations_and_weights-1'),
        pytest.param('Working for 8 ಇಯರ್ಸ್.', id='age_rules_keep_doses_durations_and_weights-2'),
        pytest.param('What is your age? 5 mg.', id='age_rules_keep_doses_durations_and_weights-3'),
        pytest.param('Symptoms for 15 days.', id='age_rules_keep_doses_durations_and_weights-4'),
        pytest.param('Menarche treatment for 12 days.', id='age_rules_keep_doses_durations_and_weights-5'),
        pytest.param('47 mg. 47 mg.', id='age_rules_keep_doses_durations_and_weights-6'),
        pytest.param('వయసు ఎంత? 50 kg.', id='age_rules_keep_doses_durations_and_weights-7'),
        pytest.param('Treatment started 15 years ago.', id='age_rules_keep_doses_durations_and_weights-8'),
        pytest.param('My menstrual is ఒకే. 15 years. My cycle is regular.', id='age_rules_keep_doses_durations_and_weights-9'),
        pytest.param('My menstrual history is 12 years of irregular periods.', id='age_rules_keep_doses_durations_and_weights-10'),
    ],
)
def test_new_age_rules_preserve_doses_and_durations(text):
    assert selected(text, AgeRecognizer()) == []


@pytest.mark.parametrize(
    'text',
    [
        pytest.param('నాలుగో మాత్ర రోజూ వేసుకోండి.', id='spoken_day_and_year_rules_require_calendar_evidence-1'),
        pytest.param('తొమ్మిది మాత్రలు ఉన్నాయి.', id='spoken_day_and_year_rules_require_calendar_evidence-2'),
        pytest.param('ఈ మందు 2014 mg. 11 mg.', id='spoken_day_and_year_rules_require_calendar_evidence-3'),
        pytest.param('The city was founded in 2014.', id='spoken_day_and_year_rules_require_calendar_evidence-4'),
        pytest.param('Pain for 15 years. Echo advised.', id='spoken_day_and_year_rules_require_calendar_evidence-5'),
        pytest.param('Take tablets 20 mg for 5 days.', id='dates_require_calendar_evidence_and_keep_measurements-1'),
        pytest.param('Pain since 2 months.', id='dates_require_calendar_evidence_and_keep_measurements-2'),
        pytest.param('January weather is cold.', id='dates_require_calendar_evidence_and_keep_measurements-3'),
        pytest.param('May I return for a visit?', id='dates_require_calendar_evidence_and_keep_measurements-4'),
        pytest.param('Last visit was April 31.', id='dates_require_calendar_evidence_and_keep_measurements-5'),
        pytest.param('LMP was February 30.', id='dates_require_calendar_evidence_and_keep_measurements-6'),
        pytest.param('LMP was February 29, 2023.', id='dates_require_calendar_evidence_and_keep_measurements-7'),
        pytest.param('Date is 25 mg.', id='dates_require_calendar_evidence_and_keep_measurements-8'),
        pytest.param('Visit on June 15 mg.', id='dates_require_calendar_evidence_and_keep_measurements-9'),
        pytest.param('BP 17/12. Tablets 21 తీసుకున్నావు.', id='dates_require_calendar_evidence_and_keep_measurements-10'),
        pytest.param('Date 12/5. Dose 5, 10 tablets.', id='dates_require_calendar_evidence_and_keep_measurements-11'),
        pytest.param('Patient returned after 20 days.', id='dates_require_calendar_evidence_and_keep_measurements-12'),
        pytest.param('ట్రీట్మెంట్ మార్చి కొత్త టాబ్లెట్ ఇచ్చారు.', id='dates_require_calendar_evidence_and_keep_measurements-13'),
        pytest.param('మే బీ ట్రీట్మెంట్ అవసరం.', id='dates_require_calendar_evidence_and_keep_measurements-14'),
        pytest.param('Doctor April scheduled a visit.', id='dates_require_calendar_evidence_and_keep_measurements-15'),
        pytest.param('பிறந்த தேதி 03-03-1993.', id='spoken_day_rules_do_not_split_complete_numeric_dates-1'),
        pytest.param('పుట్టిన తేదీ 04-04-1994.', id='spoken_day_rules_do_not_split_complete_numeric_dates-2'),
        pytest.param('ಜನ್ಮ ದಿನಾಂಕ 05-05-1995.', id='spoken_day_rules_do_not_split_complete_numeric_dates-3'),
        pytest.param('Birth date 04/04/1994.', id='spoken_day_rules_do_not_split_complete_numeric_dates-4'),
        pytest.param('Birth date 04.04.1994.', id='spoken_day_rules_do_not_split_complete_numeric_dates-5'),
        pytest.param('Date 04-04-1994. 04 visited.', id='spoken_day_rules_do_not_split_complete_numeric_dates-6'),
    ],
)
def test_spoken_day_and_year_rules_require_calendar_evidence(text):
    assert selected(text, ClinicalCalendarDateRecognizer()) == []


def test_place_answer_does_not_spread_into_later_clinical_questions():
    text = "ఏ ఊరు మీది? ఆంధ్ర. ఏం. బాలేదు మీకు? ఇంత. వచ్చినారు కదా మీరు? వచ్చినాం."
    assert selected(text, SpokenPlaceRecognizer()) == ["ఆంధ్ర"]


def test_month_date_birth_label_is_preserved():
    text = "Date of birth: August 19, 1996."
    result = ClinicalCalendarDateRecognizer().analyze(text, ["date_of_birth"])
    assert len(result) == 1 and result[0].entity_type == "date_of_birth"
    assert text[result[0].start:result[0].end] == "August 19, 1996"
