"""Span selection, identifier boundaries and output provenance."""
from __future__ import annotations

from types import SimpleNamespace
import csv
from pii_pipeline import json_codec as json

from presidio_analyzer import RecognizerResult
import pytest

from pii_pipeline.anonymizer import PlaceholderAnonymizer
from pii_pipeline.policy.entity_policy import parse_semantic_tags, resolve_entities
from pii_pipeline.policy.indic_filters import is_clinical_measurement
from pii_pipeline.recognizers.indic_identifiers import (
    AgeRecognizer,
    ConversationalPhoneRecognizer,
    CredentialFieldRecognizer,
    IndicNameContextRecognizer,
    ShortDateRecognizer,
    SpokenPlaceRecognizer,
)
from scripts.evaluate_pipeline import PipelineEvaluator


def candidate(text,value,label="PERSON",score=.95,recognizer="GLiNER2PIIPrimary",**extra):
    start=text.index(value)
    return {"entity_type":label,"start":start,"end":start+len(value),"text":value,
            "score":score,"recognizer":recognizer,**extra}


@pytest.mark.parametrize("text,fragment,whole", [
    ("నా పేరు అనిరుధ్.", "రుధ్", "అనిరుధ్"),
    ("Name: Ananya Rao.", "nanya Ra", "Ananya Rao"),
])
def test_partial_person_span_masks_complete_words(text, fragment, whole):
    selected, _ = resolve_entities(text, [candidate(
        text, fragment, recognizer="IndicNameContextRecognizer", source="resolved")])
    assert [(item["start"], item["end"], item["text"]) for item in selected] == [
        (text.index(whole), text.index(whole) + len(whole), whole)]
    assert selected[0]["recognition_metadata"]["expanded_name_word_from"] == [
        text.index(fragment), text.index(fragment) + len(fragment)]
    assert PlaceholderAnonymizer().redact_selected(text, selected) == text.replace(whole, "<PERSON_1>")


def test_name_word_expansion_does_not_change_other_labels_or_imported_tags():
    text = "నా పేరు అనిరుధ్."
    partial = candidate(text, "రుధ్", recognizer="IndicNameContextRecognizer", source="resolved")
    assert resolve_entities(text, [{**partial, "entity_type": "username"}])[0][0]["text"] == "రుధ్"
    assert resolve_entities(text, [{**partial, "source": "gemma_tag"}])[0][0]["text"] == "రుధ్"


def test_personal_telugu_city_is_redacted_without_masking_generic_mentions():
    recognizer = SpokenPlaceRecognizer()
    personal = "నాది హైదరాబాద్."
    found = recognizer.analyze(personal, ["address"])
    assert [personal[item.start:item.end] for item in found] == ["హైదరాబాద్"]
    selected, _ = resolve_entities(personal, [
        {"entity_type": item.entity_type, "start": item.start, "end": item.end,
         "score": item.score, "recognizer": recognizer.name, "source": "resolved"}
        for item in found])
    assert PlaceholderAnonymizer().redact_selected(personal, selected) == "నాది <ADDRESS_1>."
    assert recognizer.analyze("హైదరాబాద్ ఒక నగరం.", ["address"]) == []


def test_tamil_identity_name_stops_before_possessive_pronoun():
    text="நோயாளர் ஆரவ் சோதனை அவர்களின் தொலைபேசி எண் +91 90000 00003 ஆகும்."
    found=IndicNameContextRecognizer().analyze(text,["PERSON"])
    assert [text[e.start:e.end] for e in found]==["ஆரவ் சோதனை"]


@pytest.mark.parametrize("value",["5 mg","125 gm","35 mg/dL","1.09 mg/dL","3MG","250 mcg","0.5ml"])
def test_measurement_units_precede_digit_density(value):
    assert is_clinical_measurement("card_cvv",value,0,len(value))


@pytest.mark.parametrize("text,value,label",[
    ("Take టాబ్లెట్ డోలో 650 after food.","డోలో 650","card_cvv"),
    ("HbA1c 6.3. Continue review.","HbA1c 6.3","license_number"),
    ("Hemoglobin 13.4 g/dL today.","Hemoglobin 13.4","PERSON"),
    ("షుగర్ 157 ఉంది.","షుగర్ 157","street_address"),
    ("The patient takes Amlodipine 5 mg.","Amlodipine","PERSON"),
    ("Doctor gave me azithromycin to ram babu.","azithromycin","PERSON"),
    ("Take Ashwagandha after discussion.","Ashwagandha","PERSON"),
    ("Exercise regularly. Do Surya Namaskar.","Surya Namaskar","full_name"),
    ("Do Surya— Namaskar regularly.","Surya","first_name"),
    ("చెమగొట్టకుండా చూడండి.","చూడండి","last_name"),
    ("హలో, ఆల్ గుడ్.","ఆల్ గుడ్","full_name"),
    ("హలో, బీపీ చూడండి.","బీపీ","first_name"),
    ("Windows screenshot uses F11.","F11","ICD10_CODE"),
    ("Take vitamin B12 daily.","B12","ICD10_CODE"),
    ("What is that reading? T34. 20% sugar.", "T34", "ICD10_CODE"),
    ("Oxygen O2 was discussed.","O2","US_DRIVER_LICENSE"),
    ("Give 150000 for the sponsor loan.","150000","MEDICAL_RECORD_NUMBER"),
    ("The cost is 3000-4000.","3000-4000","PHONE_NUMBER"),
    ("Cost was 15000, then 3000. 5000.","3000. 5000","PHONE_NUMBER"),
    ("நன்றி, நன்றி.","நன்றி","national_id_number"),
    ("Medicines have a drug name: ExampleDrug.","ExampleDrug","PERSON"),
    ("Taking Medic card 2.5 mg.","Medic card","payment_card"),
    ("Completed 60 for the health check.","60","national_id_number"),
    ("Give paracetamol 650 mg, for 5 days.","for 5 days","password"),
    ("Doctor gave me azithromycin.","azithromycin","password"),
    ("No RWMA in the echo.","RWMA","username"),
    ("Normal valve function. RWMA ప్రెసెంట్, నో పెరికార్డియల్ ఎఫ్యూషన్.","RWMA","username"),
    ("పాదాలు నరాలు నొప్పి ఉంది.","పాదాలు","password"),
    ("Patient: मेरा demo address is Flat 7.","मेरा demo","PERSON"),
])
def test_observed_and_independent_clinical_negatives(text,value,label):
    selected,rejected=resolve_entities(text,[candidate(text,value,label)])
    assert not selected and rejected


@pytest.mark.parametrize("text,value,label",[
    ("CVV: 123", "123", "card_cvv"),
    ("Credit card security code CVC 0042", "0042", "card_cvv"),
    ("IP address: 192.168.1.10", "192.168.1.10", "ip_address"),
    ("IP address: 2001:db8::1", "2001:db8::1", "ip_address"),
    ("MRN: 150000", "150000", "MEDICAL_RECORD_NUMBER"),
    ("MRN150000; patient paid the bill.", "MRN150000", "MEDICAL_RECORD_NUMBER"),
    ("Diagnosis ICD-10 B12", "B12", "ICD10_CODE"),
    ("My name is Rest.", "Rest", "PERSON"),
    ("Name: Discomfort", "Discomfort", "PERSON"),
    ("Doctor Zeyara Rao attended.", "Zeyara Rao", "PERSON"),
    ("Password: 100 mg", "100 mg", "password"),
    ("Account number: 250000", "250000", "account_number"),
    ("Your contact number? 90.", "90", "PHONE_NUMBER"),
    ("Username: RWMA", "RWMA", "username"),
    ("Username: RWMA present in my login list", "RWMA", "username"),
    ("Password: పాదాలు", "పాదాలు", "password"),
    ("విజయలక్ష్మి. ನಂಬರ್ 150661576.","150661576","MEDICAL_ID"),
    ("MRN MRN-TEST-ML-007", "MRN-TEST-ML-007", "MEDICAL_RECORD_NUMBER"),
])
def test_real_identifiers_and_explicit_values_remain_protected(text,value,label):
    assert [e['text'] for e in resolve_entities(text,[candidate(text,value,label)])[0]]==[value]


@pytest.mark.parametrize("value",["9. 52","17. 3. 30","999.1.1.1"])
def test_invalid_literal_ips_are_not_kept_as_ip(value):
    assert not resolve_entities(value,[candidate(value,value,"ip_address")])[0]


@pytest.mark.parametrize("full,part",[("రాజేష్ కుమార్","రాజేష్"),("విజయచంద్ర రెడ్డి","రెడ్డి"),("సత్య రెడ్డి","రెడ్డి"),("Zeyara Rao","Rao")])
def test_valid_complete_name_beats_contained_gazetteer_fragment(full,part):
    text="He met "+full+" yesterday."
    whole=candidate(text,full,score=.98,recognizer="ImportedSemanticTag",source="gemma_tag")
    short=candidate(text,part,score=.9,recognizer="IndicPersonNameGazetteer",source="resolved")
    accepted,rejected=resolve_entities(text,[short,whole])
    assert [e['text'] for e in accepted]==[full]
    assert accepted[0]['supporting_evidence'][0]['evidence_kind']=='contained_name_fragment'
    assert resolve_entities(text,[whole,short])[0]==accepted
    assert short.get('supporting_evidence') is None


def test_complete_name_does_not_expand_into_a_predicate():
    text="Zeyara feels well"
    short=candidate(text,"Zeyara",recognizer="IndicPersonNameGazetteer")
    broad=candidate(text,text,score=.99)
    assert [e['text'] for e in resolve_entities(text,[short,broad])[0]]==['Zeyara']


@pytest.mark.parametrize("text,expected",[
    ("నాకు గత 10 సంవత్సరాలుగా మధుమేహం ఉంది.",[]),
    ("చేయించి 2 సంవత్సరాలు అయింది.",[]),
    ("Pain since 7 years.",[]),
    ("Treatment for 5 years.",[]),
    ("How old are you? 37.",["37"]),
    ("సర్, మీ ఏజ్ ఇప్పుడు? 65.",["65"]),
    ("మీ వయసు ఎంత? 12.",["12"]),
    ("What is your age? 33.",["33"]),
    ("My age is 10. Diabetes for 3 years.",["10"]),
    ("I am 12 years old; symptoms since 2 years.",["12"]),
    ("How old are you? 131.",[]),
])
def test_age_duration_and_answers(text,expected):
    results=AgeRecognizer().analyze(text,["AGE"])
    assert [text[r.start:r.end] for r in results]==expected


@pytest.mark.parametrize("value,valid",[("30/70",False),("20/80",False),("40/85",False),("31/04",False),
    ("29/02/2023",False),("29/02/2024",True),("12/5",True),("05/20",True),("01/01/1990",True)])
def test_real_calendar_validation(value,valid):
    rec=ShortDateRecognizer()
    assert rec.validate_result(value) is valid
    assert bool(rec.analyze(value,["sensitive_date"])) is valid


def test_fraction_is_preserved_but_a_real_date_is_protected():
    text="Quantity 1/4 కేజీ; last visit 12/5."
    selected,_=resolve_entities(text,[candidate(text,"1/4","sensitive_date"),candidate(text,"12/5","sensitive_date")])
    assert [e['text'] for e in selected]==['12/5']


@pytest.mark.parametrize("text,expected",[
    ("Name: Zeyara Nkruma.","Zeyara Nkruma"),
    ("What's your name? Zeyara.","Zeyara"),
    ("పేరు డాక్టర్ కృష్ణ ప్రసాద్.","కృష్ణ ప్రసాద్"),
    ("పేరు సుధాకర్ కదా?", "సుధాకర్"),
    ("Name: Conscious.","Conscious"),
    ("Doctor gave me azithromycin to ram babu.","ram babu"),
])
def test_unlisted_names_titles_particles_and_recipients(text,expected):
    assert expected in [text[r.start:r.end] for r in IndicNameContextRecognizer().analyze(text,["PERSON"])]


def test_treatment_purpose_is_not_a_recipient_name():
    assert not IndicNameContextRecognizer().analyze("Doctor prescribed medicine to reduce fever.",["PERSON"])


def test_contact_fragments_are_masked_at_original_offsets():
    text="మొబైల్ నెంబర్ చెప్తారా? 96. మీరు కూర్చోండి. 96? 00. 00? 135. 135? 484. 484. షుగర్ 120."
    results=ConversationalPhoneRecognizer().analyze(text,["PHONE_NUMBER"])
    assert [text[r.start:r.end] for r in results]==['96','96','00','00','135','135','484','484']
    assert not ConversationalPhoneRecognizer().analyze("Costs 3000 and 5000; mobile service available.",["PHONE_NUMBER"])
    assert not ConversationalPhoneRecognizer().analyze("ఫోన్ నెంబర్. ఇంపార్టెంట్. 47. 47 ఇయర్స్. ఏజ్ కాదు.",['PHONE_NUMBER'])


def test_location_question_and_idiom():
    rec=SpokenPlaceRecognizer()
    text="From where are you. From? చిన్నపూడు. Huh? పోరూరు, పోరూరు."
    assert {'చిన్నపూడు','పోరూరు'} <= {text[r.start:r.end] for r in rec.analyze(text,['address'])}
    assert not rec.analyze("पता चलेगा कितना चाहता है।",['address'])
    assert not rec.analyze("My interest is music.",['address'])
    text="My interest is Suraram."
    assert [text[r.start:r.end] for r in rec.analyze(text,['address'])]==['Suraram']


@pytest.mark.parametrize('text,expected',[
    ('फोन नंबर +९१ ९०००० ००००१ है। उम्र 25।','+९१ ९०००० ००००१'),
    ('ফোন নম্বর +91 90000 00001। বয়স 25।','+91 90000 00001'),
    ('ഫോൺ നമ്പർ +91 90000 00001. വയസ്സ് 25.','+91 90000 00001'),
])
def test_script_independent_contact_values_and_next_field_boundaries(text,expected):
    assert [text[r.start:r.end] for r in ConversationalPhoneRecognizer().analyze(text,['PHONE_NUMBER'])]==[expected]


@pytest.mark.parametrize('text,expected',[
    ('Password: 100 mg','100 mg'),
    ('Secret = "hello world!"','hello world!'),
    ('Username: user_48217; continue medications.','user_48217'),
    ('Access token: '+ 'x'*300, 'x'*300),
])
def test_opaque_credential_field_values(text,expected):
    rec=CredentialFieldRecognizer()
    assert [text[r.start:r.end] for r in rec.analyze(text,rec.supported_entities)]==[expected]


@pytest.mark.parametrize('text,expected',[
    ('Doctor: chart mein Meera Demo ka MRN MRN-TEST-HG-002 aur DOB 02/02/1992 note karo.','Meera Demo'),
    ('డాక్టర్: ఆరవ్ పరీక్ష డెమో గుర్తింపు 9999 0000 1113, మెడికల్ రికార్డు MRN-TEST-TE-004.','ఆరవ్ పరీక్ష'),
    ('ഡോക്ടർ: ആരവ് പരീക്ഷയുടെ ഡെമോ തിരിച്ചറിയൽ 9999 0000 1116, MRN MRN-TEST-ML-007.','ആരവ് പരീക്ഷ'),
    ('மருத்துவர்: ஆரவ் சோதனைக்கு டெமோ அடையாள எண் 9999 0000 1112 மற்றும் MRN-TEST-TA-003 உள்ளது.','ஆரவ் சோதனை'),
])
def test_identity_descriptions_do_not_expand_person_names(text,expected):
    rec=IndicNameContextRecognizer()
    results=rec.analyze(text,['PERSON'])
    assert expected in [text[r.start:r.end] for r in results]
    assert all('டெமோ' not in text[r.start:r.end] and 'డెమో' not in text[r.start:r.end] for r in results)


def test_explicit_birth_date_keeps_its_specific_type():
    text='DOB 02/02/1992'
    rec=ShortDateRecognizer()
    results=rec.analyze(text,rec.supported_entities)
    assert len(results)==1 and results[0].entity_type=='date_of_birth'


@pytest.mark.parametrize('text,value,label',[
    ('The dashboard still needs UI का review.','UI का','PERSON'),
    ('I have a. Headache and fever.','a. Headache','PERSON'),
    ('I. Worked until midnight.','I. Worked','PERSON'),
    ('I. Make sure that I eat.','I. Make','PERSON'),
    ('You are talking Hindi and he is talking Telugu.','Hindi','PERSON'),
    ('Recorded notes begin with COSO.','COSO','person'),
    ('I am a patient, I come to a doctor.','patient, I come to a doctor','person'),
    ('Medic card and Admin portal.','Medic','PERSON'),
    ('Medic card and Admin portal.','Admin','PERSON'),
    ('Subsequent luminal marrow needs review.','luminal marrow','full_name'),
    ('అది పేరు నొప్పి.','నొప్పి','PERSON'),
    ('The options are a. 2 and b. 3.','a. 2','PERSON'),
    ('I. Wish you well.','I. Wish','PERSON'),
    ('13 ఏళ్ళు, 14 ఏళ్ళు. సంవత్సరాల నుంచి.','13 ఏళ్ళు','date_of_birth'),
    ('Give 150000 for the sponsor loan.','150000','US_DRIVER_LICENSE'),
    ('The charge was 240000.','240000','US_DRIVER_LICENSE'),
])
def test_500_row_false_redactions_are_rejected(text,value,label):
    selected,rejected=resolve_entities(text,[candidate(text,value,label,score=.01 if label=='US_DRIVER_LICENSE' else .95)])
    assert selected==[] and rejected


@pytest.mark.parametrize('text,value',[
    ('My name is Hindi.','Hindi'),
    ('Name: Medic','Medic'),
    ('Dr. I. Kumar attended.','I. Kumar'),
])
def test_explicit_person_identity_survives_ordinary_word_filters(text,value):
    selected,_=resolve_entities(text,[candidate(text,value,'PERSON')])
    assert [item['text'] for item in selected]==[value]


@pytest.mark.parametrize('text,expected',[
    ('షుగర్ నాకు అయిపోయింది 15 సంవత్సరాలు. 15 సంవత్సరాలు.',[]),
    ('13 ఏళ్ళు, 14 ఏళ్ళు. సంవత్సరాల నుంచి.',[]),
    ('టాబ్లెట్ అంటే 3 సంవత్సరాలు.',[]),
    ('ఈ చికిత్సకు 4 సంవత్సరాలు.',[]),
    ('నా వయసు 4 ఏళ్ళు.',['4']),
    ('వయసు. 61 సంవత్సరాలు ఉంది.',['61']),
    ('I am 25 years old.',['25']),
])
def test_bare_years_need_age_evidence(text,expected):
    found=AgeRecognizer().analyze(text,['AGE'])
    assert [text[item.start:item.end] for item in found]==expected


def test_short_license_guess_needs_license_context():
    text='Driver license number: 150000'
    selected,_=resolve_entities(text,[candidate(text,'150000','US_DRIVER_LICENSE',score=.01)])
    assert [item['text'] for item in selected]==['150000']


def test_spoken_phone_fragment_is_not_discarded_as_weak_license():
    text='నెంబర్ చెప్పండి. 8179. 3867. 44. 817931.'
    selected,_=resolve_entities(text,[candidate(text,'817931','US_DRIVER_LICENSE',score=.01)])
    assert [item['text'] for item in selected]==['817931']


def test_named_doctor_is_protected_even_after_a_place():
    from pii_pipeline.recognizers.indic_identifiers import IndicPersonNameGazetteer
    text='గవర్నమెంట్ డాక్టర్? కూడూరులో భాస్కర్ రా సార్.'
    found=IndicPersonNameGazetteer().analyze(text,['PERSON'])
    proposals=[candidate(text,text[item.start:item.end],recognizer='IndicPersonNameGazetteer') for item in found]
    selected,_=resolve_entities(text,proposals)
    assert [item['text'] for item in selected]==['భాస్కర్']


def test_labelled_record_and_pin_have_priority_over_model_type_guesses():
    from pii_pipeline.recognizers.indic_identifiers import build_indic_identifier_recognizers
    text='MRN MRN-TEST-KN-005; UPI PIN 0003'
    candidates=[candidate(text,'MRN-TEST-KN-005','sensitive_account_id',.99),candidate(text,'0003','secret',.99)]
    for rec in build_indic_identifier_recognizers():
        for r in rec.analyze(text,rec.supported_entities,None):
            candidates.append(dict(entity_type=r.entity_type,start=r.start,end=r.end,score=r.score,
                                   recognizer=r.recognition_metadata.get('recognizer_name',rec.name),
                                   recognition_metadata=r.recognition_metadata))
    selected,_=resolve_entities(text,candidates)
    assert [(e['text'],e['entity_type']) for e in selected]==[('MRN-TEST-KN-005','MEDICAL_RECORD_NUMBER'),('0003','PASSWORD')]


def test_batch_diagnostics_provenance_and_resume_guard(tmp_path,monkeypatch):
    from tests.helpers import fake_pipeline
    from pii_pipeline import run_csv_through_stage9 as runner
    pipe=fake_pipeline()
    monkeypatch.setattr(runner,'Stage9Pipeline',lambda **kwargs:pipe)
    monkeypatch.setattr(runner,'configure_runtime',lambda *args:None)
    monkeypatch.setattr(runner,'get_hardware_info',lambda device:{})
    source=tmp_path/'input.csv';output=tmp_path/'output.csv'
    text='సరే Ananya'
    write_dataset(source,[dict(row_index='case-1',context=text,semantic_tags=json.dumps([
        {'tag':'person','text':'Ananya','start':text.index('Ananya'),'end':len(text)}]))])
    runner.run_pipeline_on_file(str(source),str(output),device='cpu',save_diagnostics=True)
    diagnostics=output.with_suffix('.diagnostics.jsonl')
    record=json.loads(diagnostics.read_text(encoding='utf-8').strip())
    assert record['row_index']=='case-1' and record['decision_trace']['policy_candidates']
    assert output.with_suffix('.csv.run_manifest.json').exists()
    runner.run_pipeline_on_file(str(source),str(output),device='cpu',save_diagnostics=True)
    assert len(diagnostics.read_text(encoding='utf-8').splitlines())==1
    with pytest.raises(ValueError,match='provenance'):
        runner.run_pipeline_on_file(str(source),str(output),device='cpu',gliner_threshold=.8)


def test_acceptance_threshold_is_source_type_script_specific():
    text="అజయ్"
    low=candidate(text,text,score=.7)
    assert not resolve_entities(text,[low],score_thresholds={'gliner2piiprimary:person:telugu':.8})[0]
    assert resolve_entities(text,[low],score_thresholds={'gliner2piiprimary:person:latin':.8})[0]
    assert resolve_entities(text,[{**low,'recognizer':'IndicNameContextRecognizer'}],score_thresholds={'gliner2piiprimary:person:telugu':.8})[0]
    with pytest.raises(ValueError):
        resolve_entities(text,[low],score_thresholds={'person':float('nan')})


def write_dataset(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_reviewed_evaluation_handles_bom_and_measures_partial_protection(tmp_path):
    text="Ananya Rao"
    reference=dict(row_index='case-1',context=text,source_fingerprint='',
        verified_semantic_tags=json.dumps([dict(entity_type='person',text=text,start=0,end=len(text))]),
        review_status='reviewed',annotation_scope='exhaustive',reviewer='test-fixture-author')
    prediction=dict(row_index='case-1',context=text,pipeline_entities=json.dumps([candidate(text,'Ananya')]))
    gt,pred=tmp_path/'gt.csv',tmp_path/'pred.csv'
    write_dataset(gt,[reference]);write_dataset(pred,[prediction])
    result=PipelineEvaluator(require_reviewed=True).evaluate(str(gt),str(pred))
    assert result['summary']['matched_source_rows']==1
    assert result['summary']['complete_span_coverage']['fully_hidden']==0
    assert result['summary']['verified_human_reference']
    reference['review_status']='pending';write_dataset(gt,[reference])
    with pytest.raises(ValueError,match='review'):
        PipelineEvaluator(require_reviewed=True).evaluate(str(gt),str(pred))


def test_reviewed_evaluation_requires_reference_for_every_output_row(tmp_path):
    gt,pred=tmp_path/'gt.csv',tmp_path/'pred.csv'
    reference=dict(row_index='case-1',context='Nothing private.',verified_semantic_tags='[]',
        review_status='reviewed',annotation_scope='exhaustive',reviewer='test-fixture-author')
    write_dataset(gt,[reference])
    write_dataset(pred,[dict(row_index='other-case',context='Zeyara',pipeline_entities='[]')])
    with pytest.raises(ValueError,match='every prediction row'):
        PipelineEvaluator(require_reviewed=True).evaluate(str(gt),str(pred))
