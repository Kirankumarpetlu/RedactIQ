"""Conservative context checks for clinical text misclassified as identifiers.

These checks run in the common acceptance path. They reject positive evidence
of ordinary clinical speech, rather than requiring every identifier to have a
keyword. Explicit identification and credential fields keep their values.
"""
from __future__ import annotations

import re
import ipaddress
import unicodedata
from datetime import date
from typing import Mapping, Any

from .indic_filters import is_suppressed, normalise_span, suppression_enabled
from ..taxonomy import canonical_entity_type


_NAME_FIELD = re.compile(
    r"(?:\b(?:my\s+)?(?:full\s+|first\s+|last\s+)?name\s*(?:is|:)\s*|"
    r"\bsurname\s*(?:is|:)\s*|పేరు\s*:?\s*|नाम\s*(?:है|:)\s*|"
    r"பெயர்\s*:?\s*|ಹೆಸರು\s*:?\s*)['\"‘“]?\s*$", re.I,
)
_CREDENTIAL_LABELS = frozenset({
    "username", "password", "secret", "api_key", "access_token",
    "recovery_code", "account_id", "sensitive_account_id",
})
_CREDENTIAL_FIELD = re.compile(
    r"(?:\b(?:user\s*name|login|password|passphrase|secret|api\s*key|access\s*token|"
    r"recovery\s*code|account\s*id|pin|otp)\b|పాస్\s*వర్డ్|యూజర్\s*నేమ్|"
    r"पासवर्ड|यूज़र\s*नेम|ಪಾಸ್\s*ವರ್ಡ್|கடவுச்சொல்)"
    r"\s*(?:is|:|=|ఇస్|है)?\s*['\"‘“]?\s*$", re.I,
)
_MEDICATION_BEFORE = re.compile(
    r"(?:\b(?:tablets?|capsules?|syrup|injection|(?:eye|ear)\s+drops?|"
    r"medications?)\b|టాబ్లెట్|క్యాప్సూల్|టాబ్లెట్లు|టాబ్లెట్స్|टैबलेट|ಕ್ಯಾಪ್ಸೂಲ್)"
    r"\s*[:,-]?\s*(?:(?:and|please\s+add)\s+)?$", re.I,
)
_MEDICATION_START = re.compile(r"^(?:tablets?|capsules?|syrup|injection|టాబ్లెట్)\s+", re.I)
_MEDICATION_AFTER = re.compile(
    r"^\s*(?:(?:eye|ear)\s+drops?\b|\d+(?:\.\d+)?\s*(?:mg|mcg|ml)\b)", re.I,
)
_ADMINISTRATION_BEFORE = re.compile(r"(?:\b(?:take|took|taken|give|add|start|continue|prescribed)\b|gave\s+(?:me|him|her)|టేక్|తీసుకోండి)\s*$",re.I)
_PRESCRIPTION_DURATION = re.compile(r"^(?:for|since|ఫర్|సిన్స్)\s+\d+\s*(?:days?|weeks?|months?|డేస్|వీక్స్|నెలలు)$",re.I)
_CLINICAL_HEADINGS = re.compile(
    r"^(?:(?:[ivx]+|\d+)\.\s*)?(?:medications?|prescriptions?|"
    r"(?:medical|personal|family)\s+history|chief\s+complaints?)$", re.I,
)
_CLINICAL_WORDS = frozenset({
    "discomfort", "nausea", "vomiting", "constipation", "diarrhoea", "diarrhea",
    "headache", "dizziness", "menstruation", "urination",
    "kidney", "liver", "cardiologist", "eating", "dropper",
})
_DURATION = re.compile(
    r"^(?:\d+(?:\.\d+)?(?:\s*[-–]\s*\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:seconds?|minutes?|hours?|hrs?|days?|weeks?|months?|years?|"
    r"రోజులు|రోజు|వీక్స్|మంత్స్|నెలలు|నెల|గంటలు|ఏళ్ళు|సంవత్సరాలు|సంవత్సరాల|दिन|हफ्ते|महीने|साल|ವರ್ಷ|ದಿನ|ತಿಂಗಳು)$", re.I,
)
_DATE_LABELS = frozenset({
    "sensitive_date", "date_time", "date_of_birth", "document_date",
    "expiration_date", "transaction_date",
})

# Comparison uses normalized surfaces; offsets always refer to original text.
_ORDINARY_PERSON_WORDS = frozenset({
    "no", "yes", "okay", "ok", "hello", "hi", "complaint", "pain", "normal", "sugar", "blood", "pressure", "मेरा",
    "rest", "stomach", "chicken", "turmeric", "glass", "eat", "hmm", "mm-hmm", "afternoon",
    "grandfather", "grandmother", "cholesterol", "insulin", "hemoglobin", "rwma", "gerd",
    "mg", "mcg", "gm", "dl", "neuro", "timestamps", "medicines", "symptom", "emr",
    "problem", "random", "intro", "literally", "submitter", "down", "perfect",
    "hindi", "medic", "admin", "shh", "em", "ui का", "luminal marrow", "coso", "jigwa",
    "ready", "smiley", "क्या", "है", "మీడియం", "so", "here",
    # First-person pronouns are frequent spaCy PERSON hallucinations in
    # code-switched clinical speech. Explicit Name: fields still override this.
    "నేను", "ನಾನು", "मैं", "நான்", "আমি", "ഞാൻ",
    "బీపీ", "నరాలు", "నరాల", "చూడండి", "మధ్యలో", "వద్దు", "చేప", "మటన్", "దద్దుర్లు",
    "పరీక్ష", "నొప్పి", "చెప్తారు", "వస్తుంది", "మీద", "ఏంటి", "ఏంది", "ఎట్లా",
    "మారుద్ది", "మర్చిపోయాను", "గుర్తుపెట్టుకోలే", "చెప్పారన్నమాట", "ముందుగా", "పప్పు", "పొక్కు", "గ్యాస్", "కరెంట్", "బీపీకి",
    "హలో", "డాక్టర్", "ఆల్", "గుడ్", "కోల్డ్", "హెడ్‌ఏక్", "హెడ్ ఏక్", "మ్",
    "ഹലോ", "ഡോക്ടർ", "ஒரு", "शुगर", "एडमिट", "सॉरी",
})
_MEDICINE_NAMES = frozenset({
    "amlodipine", "azithromycin", "paracetamol", "ashwagandha", "avipathikara churna",
    "praval panchamrat", "clozapin", "salbutamol", "sarbutamol", "nitro", "డ్యూర్",
})
# ASR spellings need medication evidence, rather than a global name blacklist.
_MEDICATION_ASR_NAMES = frozenset({"dolo", "tolo", "lolo"})
_MEDICATION_ALTERNATIVE = re.compile(
    r"^\s*(?:or|and)\s+(?:paracetamol|azithromycin|amlodipine|salbutamol)\b", re.I,
)
_MEDICATION_ALTERNATIVE_SPAN = re.compile(
    r"^[\w'-]+\s+(?:or|and)\s+(?:paracetamol|azithromycin|amlodipine|salbutamol)$", re.I,
)
_GENERIC_PERSON_ROLES = frozenset({
    "principal", "minister", "collector", "teacher", "ప్రిన్సిపల్", "మంత్రి",
    "కలెక్టర్", "టీచర్", "అన్నా",
})
_ORDINARY_PERSON_PHRASES = frozenset({"so here i am", "here i am", "i am"})
_ACTIVITIES = frozenset({"surya namaskar", "nadi pariksha"})
_PERSON_STOP_WORDS = _ORDINARY_PERSON_WORDS | _CLINICAL_WORDS | frozenset({
    "feels", "reports", "conscious", "oriented", "advised", "and", "is", "was", "has",
    "well", "better", "patient", "doctor", "nurse", "to", "take", "reduce", "improve",
    "medication", "medicine", "tablet", "capsule", "syrup", "history", "for", "since",
    "కదా", "అండి", "గారు", "సార్", "మీరు", "అంతా", "ఉంది", "లేదు", "మాత్ర",
}) | _GENERIC_PERSON_ROLES
_OBJECT_NAME = re.compile(r"(?:medicine|drug|tablet|test|procedure|దాని|దానికి|అది|మందు|మాత్ర)\s+(?:name|పేరు)\s*(?:is|:)?\s*$", re.I)
_PERSON_TITLE = re.compile(r"(?:\b(?:dr|mr|mrs|ms)\.?|\bdoctor|\bpatient|డాక్టర్|రోగి|திரு|திருமதி|ಡಾಕ್ಟರ್)\s*[:,-]?\s*$", re.I)
_LAB_EXPRESSION = re.compile(r"(?:hba1c|hemoglobin|haemoglobin|\bhb\b|platelets?|glucose|sugar|bp|blood\s*pressure|షుగర్|బీపీ|బిపి)\s*\d", re.I)
_MEASUREMENT_EXPRESSION = re.compile(r"\d(?:[\d.,/\s]*\d)?\s*(?:mg|mcg|gm|ml|g/dl|mg/dl|mmhg|mm|cm|kg|కేజీ|గ్రామ్స్)(?![A-Za-z])", re.I)
_CARD_CONTEXT = re.compile(r"\b(?:cvv|cvc|card\s*(?:verification|security)|(?:credit|debit|payment|bank)\s*card)\b", re.I)
_IDENTIFIER_FIELD = re.compile(r"(?:\b(?:mrn|uhid|patient\s*id|medical\s*(?:record|id)|record\s*(?:number|no)|registration\s*(?:id|number)|national\s*id|passport|(?:driver'?s?\s*)?licen[cs]e|account\s*(?:number|no)|aadhaar)\b|ఆధార్|పేషెంట్\s*ఐడి|రికార్డ్\s*నంబర్)\s*(?:number|no\.?|is|:|=|#)?\s*$", re.I)
_LICENSE_FIELD = re.compile(r"(?:\b(?:driver'?s?\s+)?licen[cs]e)\s*(?:(?:number|no\.?)\s*)?[:=#]?\s*$", re.I)
_MONEY_CONTEXT = re.compile(r"(?:\b(?:cost|rupees|paid|loan|sponsor|bill|amount)\b|"
                            r"(?<![A-Za-z\u0900-\u0D7F])(?:ఖర్చు|డబ్బు|లక్షలు?|స్పాన్సర్|అప్పు|ఇయ్యాలి)(?![A-Za-z\u0900-\u0D7F]))",re.I)
_CLINICAL_CODE_CONTEXT = re.compile(r"(?:vitamin|supplement|oxygen|tablet|capsule|సల్ఫా|మాత్ర|\bo2\b|\bd3\b)", re.I)
_CLINICAL_ANALYTE = frozenset({"hba1c", "b12", "d3"})
_ANALYTE_CONTEXT = re.compile(
    r"(?:\b(?:vitamin|supplement|test|blood|sugar|result)\b|విటమిన్|చేయించ|షుగర్|"
    r"विटामिन|जांच|டெஸ்ட்|வைட்டமின்|பரிசோதனை)", re.I)
_MEDICATION_ACTION = re.compile(
    r"(?:\b(?:take|took|prescribed|supplement)\b|ఎత్తుకో|తీసుకో|వాడుతు|"
    r"எடுத்துக்க|எடுக்க|மருந்து|மாத்திரை)", re.I)
_DIETARY_WORDS = frozenset({
    "guava", "guavas", "buttermilk", "juice", "water", "flour", "millet", "sorghum",
    "fruit", "fruits", "పండ్లు",
    "జామకాయలు", "మజ్జిగ", "నీళ్లు", "శనగపిండి", "జొన్నలు", "రాగులు", "మినపట్లు",
    "పళ్ళ", "పళ్ల", "రసం", "దోస", "రోట్టెలు", "రొట్టెలు", "పాలు", "కర్రీ",
})
_DIET_CONTEXT = re.compile(r"(?:\b(?:eat|drink|diet|fruit|fruits|food|grain|grains)\b|"
                           r"తిన|తాగ|పండ్లు|ఆహారం|కర్రీ\s*అంటే)", re.I)
_CLINICAL_SPEECH_CONTEXT = re.compile(
    r"(?:\b(?:patient|tablet|capsule|medicine|medication|prescription|pain|kidney|"
    r"cardiologist|periods|renal|hba1c|vitamin|blood|sugar|fever|outpatient)\b|"
    r"షుగర్|మందులు|నొప్పి|வலி|எடுக்க|எடுத்துக்க|மருந்து|நோயாளி|மாத்திரை)", re.I)
_PRESCRIPTION_LIST_ITEM = re.compile(r"(?<![\w])[A-Za-z][A-Za-z'-]{1,20}\s+\d{1,3}(?![\w])")
_KEYBOARD_CONTEXT = re.compile(r"\b(?:windows|screenshot|snipping|keyboard|function\s*key)\b", re.I)
_DIAGNOSIS_CONTEXT = re.compile(r"\b(?:icd(?:-?10)?|diagnosis|diagnostic\s*code)\b", re.I)
_YEAR_UNIT = re.compile(r"^\s*(?:years?|yrs?|సంవత్సరాలు|సంవత్సరాల|ఏళ్ళు|ఏళ్ల|साल|वर्ष|ವರ್ಷ|இயர்ஸ்|வருஷம்)", re.I)
_DURATION_BEFORE = re.compile(r"(?:\b(?:since|for|past|last|ago)\b|గత|ఎంత\s*కాలం|ఎంతకాలంగా|నుంచి|చేయించి|అయిపోయింది|మొత్తం|టాబ్లెట్\s*అంటే|ఎన్ని\s*సంవత్సరాలు|कब\s*से|से\s*$)", re.I)
_DURATION_AFTER = re.compile(r"^\s*(?:గా|అయింది|అయ్యింది|నుంచి|ago\b|on\s+treatment\b|of\s+(?:treatment|pain|diabetes)\b|से\b)", re.I)


def has_person_identity_context(text: str, entity: Mapping[str, Any]) -> bool:
    before = text[max(0, entity["start"] - 90):entity["start"]]
    if _OBJECT_NAME.search(before):
        return False
    surface=normalise_span(text[entity["start"]:entity["end"]])
    words=surface.split()
    strong_field=bool(re.search(r"(?:\b(?:patient\s+)?(?:full\s+|first\s+|last\s+)?name\s*:|my\s+name\s+is|నా\s*పేరు)\s*$",before,re.I))
    if re.sub(r"\s+", " ", surface) in _ORDINARY_PERSON_PHRASES and not strong_field:
        return False
    if words and all(w in _PERSON_STOP_WORDS for w in words) and not strong_field:
        return False
    metadata = entity.get("recognition_metadata") or {}
    boundary = metadata.get("boundary_evidence") if isinstance(metadata, Mapping) else None
    return bool(_NAME_FIELD.search(before) or _PERSON_TITLE.search(before) and
                plausible_person_surface(text[entity["start"]:entity["end"]])
                or entity.get("recognizer") == "IndicNameContextRecognizer" and boundary in
                {"explicit_name", "identity_details", "identity_relation", "patient_introduction"})


def has_clinical_speech_context(text: str, start: int, end: int) -> bool:
    local = text[max(0, start-100):end+100]
    return bool(_CLINICAL_SPEECH_CONTEXT.search(local) or len(_PRESCRIPTION_LIST_ITEM.findall(local)) >= 2)


def has_clinical_year_context(text: str, start: int, end: int) -> bool:
    """A year answer to a nearby disease/treatment-history question, not a DOB."""
    if not re.fullmatch(r"(?:19|20)\d{2}", text[start:end]):
        return False
    before, after = text[max(0, start-110):start], text[end:end+35]
    return bool(
        re.match(r"\s*నుంచి(?![A-Za-z\u0900-\u0D7F])", after)
        and re.search(r"(?:ఎన్ని\s*సంవత్సరాల\s*నుంచి)[\s?.:,]*$", before)
        and re.search(r"(?:షుగర్|బీపీ|బిపి|ఆస్తమా|మందు|మాత్ర|diabetes|hypertension|treatment)", before, re.I)
    )


def plausible_person_surface(surface: str, *, explicit: bool = False) -> bool:
    """Validate a proposed whole-name boundary, not whether a name is listed."""
    words = surface.split()
    if not 1 <= len(words) <= 4:
        return False
    for word in words:
        value = word.strip(".'’-")
        if not value or not all(unicodedata.category(c)[0] in {"L", "M"} or c in "-'’." for c in value):
            return False
        if not explicit and normalise_span(value) in _PERSON_STOP_WORDS:
            return False
    return True


_HISTORICAL_AGE_BEFORE = re.compile(
    r"(?:\bage\s*(?:of\s+menarche\s*)?(?:at|of|is|:)?|"
    r"\bmenarche\s*(?:is|at|age|:)?|"
    r"ఏజ్\s*(?:అట్|ఆఫ్|ఇస్))\s*[.:]?\s*$", re.I,
)
_NON_AGE_UNIT = re.compile(
    r"^\s*(?:mg|mcg|ml|kg|gm|mmhg|cm|mm|bpm|%)(?![A-Za-z])", re.I,
)
_SHORT_AGE_UNIT = re.compile(
    r"^\s*(?:days?|weeks?|months?|hours?|డేస్|రోజులు|నెలలు|మంత్స్|గంటలు|"
    r"दिन|महीने|ದಿನ|ತಿಂಗಳು)(?![A-Za-z])", re.I,
)
_EXPLICIT_AGE_BEFORE = re.compile(
    r"(?:\bage\s*(?:is|of|at|:)?|how\s+old\s+are\s+you|వయసు|వయస్సు|"
    r"వయసు\s*(?:మీకు|నీకు)|వయస్సు\s*(?:మీకు|నీకు)|ఏజ్\s*(?:ఇస్|is)?|"
    r"उम्र|आयु|ವಯಸ್ಸು|வயது)[\s?.:,;]*$", re.I,
)


def has_historical_age_context(text: str, start: int) -> bool:
    """A directly stated historical age, rather than time since an illness."""
    return bool(_HISTORICAL_AGE_BEFORE.search(text[max(0, start - 80):start]))


def age_is_duration(text: str, start: int, end: int) -> bool:
    before, after = text[max(0, start-55):start], text[end:end+65]
    if _NON_AGE_UNIT.match(after):
        return True
    short_unit = _SHORT_AGE_UNIT.match(after)
    if short_unit:
        # An explicit infant age can be stated in days or months. Symptom
        # durations and generic history fields do not supply this evidence.
        return not (_EXPLICIT_AGE_BEFORE.search(before) or
                    re.match(r"[ \t]+old\b", after[short_unit.end():], re.I))
    if has_historical_age_context(text, start):
        return False
    unit = _YEAR_UNIT.match(after)
    if not unit:
        return False
    remainder = after[unit.end():]
    if re.match(r"\s*(?:వయసు(?:లో)?|వచ్చాయి|ఏజ్|of\s+age\b)", remainder, re.I):
        return False
    if re.match(r"\s*(?:old\b|ఓల్డ్|ಓಲ್ಡ್)", remainder, re.I):
        return False
    if re.search(r"(?:\bage\s*(?:is|:)?|వయసు|వయస్సు|வயது|उम्र|ವಯಸ್ಸು)[\s?.:,;]*$", before, re.I):
        return False
    # A bare Indic "14 years" needs an age cue. Without one it commonly
    # states disease or treatment duration in these clinical conversations.
    if re.match(r"\s*(?:ఏళ్ళు|ఏళ్ల|సంవత్సరాలు|సంవత్సరాల|वर्ष|साल|ವರ್ಷ|வருஷம்)", after, re.I):
        if not re.search(r"(?:\bi\s+am|\bi['’]m|నేను|ನಾನು|நான்|मैं)\s*$", before, re.I):
            return True
    return bool(_DURATION_BEFORE.search(before) or _DURATION_AFTER.match(remainder))


def valid_calendar_literal(value: str) -> bool:
    """Accept real D/M, M/D, date/year, ISO dates and explicit month/year."""
    try:
        parts = value.split("/")
        numbers = [int(p) for p in parts]
        if len(parts) == 2 and len(parts[1]) == 4:
            return 1 <= numbers[0] <= 12 and 1000 <= numbers[1] <= 9999
        if len(parts) == 3 and len(parts[0]) == 4:
            date(*numbers)
            return True
        if len(parts) not in {2,3}:
            return False
        first,second = numbers[:2]
        year = numbers[2] if len(numbers)==3 else 2000
        if len(parts)==3 and len(parts[2])==2:
            year += 2000
        for month,day in ((second,first),(first,second)):
            try:
                date(year,month,day)
                return True
            except ValueError:
                pass
    except (ValueError,TypeError):
        pass
    return False


def identifier_rejection_reason(text: str, entity: Mapping[str, Any]) -> str | None:
    label = entity["entity_type"].casefold()
    a,b = entity["start"],entity["end"]
    surface, before, around = text[a:b].strip(),text[max(0,a-80):a],text[max(0,a-80):b+80]
    explicit = bool(_IDENTIFIER_FIELD.search(before) or re.match(r"^(?:MRN|UHID)[-_ :#]?[A-Z0-9]+",surface,re.I))
    if label == "ip_address":
        try:
            ipaddress.ip_address(surface)
        except ValueError:
            return "invalid_ip_address"
    if label == "card_cvv":
        if not re.fullmatch(r"\d{3,4}",surface):
            return "invalid_cvv_format"
        if not _CARD_CONTEXT.search(around):
            return "unsupported_cvv_context"
    if label in {"phone_number","phone"}:
        phone_field=bool(re.search(r"(?:phone|contact|mobile)\s*(?:number|no)|మొబైల్\s*నెంబర్|ఫోన్\s*నంబర్",around,re.I))
        if not phone_field and (_MONEY_CONTEXT.search(around) or
                len(re.sub(r"\D","",surface))<10 and re.fullmatch(r"\d{3,5}\s*[-.]\s*\d{3,5}",surface)):
            return "quantity_not_phone_number"
    if explicit:
        return None
    if label in {"payment_card","card_number"}:
        digits=re.sub(r"\D","",surface)
        card_field=bool(re.search(r"\b(?:card\s*number|(?:credit|debit|payment)\s*card)\s*(?:is|:|=)?\s*$",before,re.I))
        if not card_field and not (13<=len(digits)<=19 and re.fullmatch(r"[\d -]+",surface)):
            return "unsupported_payment_card_format"
    if label in {"national_id_number","passport_number"} and any(c.isdigit() for c in surface) and len(re.sub(r"\W","",surface))<5:
        return "unsupported_short_identity_value"
    if label == "medical_record_number" and _MONEY_CONTEXT.search(around):
        return "amount_not_medical_identifier"
    if label == "medical_record_number" and entity.get("score",0) <= .3:
        return "unsupported_medical_record_pattern"
    if label in {"us_driver_license","drivers_license_number","license_number"}:
        if _LICENSE_FIELD.search(before):
            return None
        # A weak license guess may actually be a spoken phone fragment. Keep
        # it protected when the speaker is explicitly dictating a number.
        if re.search(r"(?:\b(?:phone|mobile|contact)\s*(?:number|no)\b|నెంబర్\s*చెప్పండి|నంబర్\s*చెప్పండి)", before, re.I):
            return None
        if _LAB_EXPRESSION.search(surface) or _CLINICAL_CODE_CONTEXT.search(around):
            return "clinical_code_not_license"
        # A bare six-digit amount is an extremely weak license guess. An
        # explicit license field still wins via the early `explicit` return.
        if surface.isdecimal() and len(surface) <= 6 and entity.get("score", 0) <= .3:
            return "unsupported_short_license_pattern"
        if len(surface)<=4 and entity.get("score",0)<=.3:
            return "unsupported_license_pattern"
    if label == "icd10_code" and not _DIAGNOSIS_CONTEXT.search(around):
        if normalise_span(surface) in _CLINICAL_ANALYTE and _ANALYTE_CONTEXT.search(around):
            return "clinical_analyte_not_diagnosis_code"
        if _KEYBOARD_CONTEXT.search(around) or _CLINICAL_CODE_CONTEXT.search(around):
            return "ordinary_code_not_diagnosis"
        if re.search(r"\bwhat\s+is\b", before[-25:], re.I) and re.match(
            r"[.!? \t]+\d+(?:\.\d+)?\s*%\s*sugar\b", text[b:b+30], re.I,
        ):
            return "ordinary_code_not_diagnosis"
    if label in {"account_number","bank_account"} and (_MONEY_CONTEXT.search(around) or _LAB_EXPRESSION.search(around)):
        return "quantity_not_account_identifier"
    if label in {"national_id_number","passport_number","license_number"} and not any(c.isdigit() for c in surface):
        words = set(normalise_span(surface).split())
        if is_suppressed("PERSON",surface) or words & _ORDINARY_PERSON_WORDS or surface in {"ఐరన్ ఇస్ లెస్","ఇన్ దట్","అప్రెషన్","நன்றி"}:
            return "ordinary_expression_not_identifier"
    return None


def has_explicit_credential_context(text: str, entity: Mapping[str, Any]) -> bool:
    return (entity["entity_type"].strip().casefold() in _CREDENTIAL_LABELS
            and bool(_CREDENTIAL_FIELD.search(text[max(0, entity["start"] - 80):entity["start"]])))


def has_explicit_identifier_context(text: str, entity: Mapping[str, Any]) -> bool:
    before=text[max(0,entity['start']-80):entity['start']]
    surface=text[entity['start']:entity['end']]
    return bool(_IDENTIFIER_FIELD.search(before) or re.match(r'^(?:MRN|UHID)[-_ :#]?[A-Z0-9]+',surface,re.I))


def contextual_rejection_reason(text: str, entity: Mapping[str, Any]) -> str | None:
    """Return a rejection reason only when a candidate has non-PII evidence."""
    if not suppression_enabled():
        return None
    label = entity["entity_type"].strip().casefold()
    start, end = entity["start"], entity["end"]
    surface = text[start:end]
    normalized = normalise_span(surface)
    before, after = text[max(0, start - 80):start], text[end:end + 50]

    if label == "age" and age_is_duration(text,start,end):
        return "duration_not_patient_age"
    if label in _DATE_LABELS and re.fullmatch(r"\d{1,4}/\d{1,4}(?:/\d{2,4})?",normalized):
        if not valid_calendar_literal(normalized):
            return "invalid_calendar_date"
        if re.match(r"^\s*(?:kg|gm|mg|కేజీ|కిలో|గ్రామ్స్)(?![A-Za-z])",after,re.I):
            return "fraction_not_calendar_date"

    if label in _DATE_LABELS and _DURATION.fullmatch(normalized):
        return "duration_not_calendar_date"

    medication = bool(_MEDICATION_BEFORE.search(before) or _MEDICATION_START.search(surface)
                      or _MEDICATION_AFTER.search(after))
    medication = medication or bool(_ADMINISTRATION_BEFORE.search(before) and
        (normalized in _MEDICINE_NAMES or normalized in _MEDICATION_ASR_NAMES
         or _MEASUREMENT_EXPRESSION.search(surface) or _MEDICATION_AFTER.search(after)
         or _MEDICATION_ALTERNATIVE.match(after) or _MEDICATION_ALTERNATIVE_SPAN.fullmatch(surface)))
    # Repeated brand + supplement modifier gives medication evidence without
    # listing brand names. Explicit identity/credential fields still win below.
    medication = medication or bool(_MEDICATION_ACTION.search(before) and re.search(
        r"(?<!\w)" + re.escape(surface) + r"\s+(?:D3|B12|plus)(?!\w)", after, re.I))
    if label in _CREDENTIAL_LABELS:
        if has_explicit_credential_context(text, entity):
            return None
        if normalized in _CLINICAL_ANALYTE and _ANALYTE_CONTEXT.search(before+surface+after):
            return "credential_is_clinical_analyte"
        if _MEDICATION_ACTION.search(before) and _CLINICAL_CODE_CONTEXT.search(after):
            return "credential_in_medication_context"
        if is_suppressed("PERSON", surface):
            return "credential_is_conversation_filler"
        if medication:
            return "credential_in_medication_context"
        if _PRESCRIPTION_DURATION.fullmatch(normalized) and (_MEASUREMENT_EXPRESSION.search(before) or
                re.search(r"tablet|capsule|medicine|టాబ్లెట్|మాత్ర",before,re.I)):
            return "credential_duration_in_medication_context"
        if normalized=='rwma' and (
                re.search(r"(?:\bno\s*$|నో\s*$|లేదు\s*$|echo|echocardiogram|ఎకో)",before,re.I)
                or re.match(r"\s*(?:present\b|absent\b|ప్రెసెంట్|లేదు)",after,re.I)):
            return "credential_is_clinical_expression"
        if normalized=='పాదాలు' and re.search(r"నరాలు|కాళ్ళు|నొప్పి",before+after):
            return "credential_is_clinical_expression"
        # Unintroduced real handles/secrets remain candidates; morphology or
        # length alone cannot prove that a value is not a credential.
        return None

    if (identifier_reason := identifier_rejection_reason(text,entity)):
        return identifier_reason

    if canonical_entity_type(label)=="LOCATION":
        if normalized in {"ఇక్కడ","చిన్న","ఇంత","వచ్చినాం","కాలు","చెప్పండి","చూద్దాం","ఆడే ఆసుపత్రిలో","चलेगा"} or _LAB_EXPRESSION.search(surface):
            return "ordinary_expression_not_location"
        return None

    if canonical_entity_type(label) != "PERSON":
        return None
    # Telugu రావు is both a surname and the negative verb "do not come".
    # Reject only an unambiguous ordinary negation, preserving named Raos.
    if normalized == "రావు" and (re.search(r"(?:ఏమి|ఏం|ఎక్కువ\s*సార్లు)\s*$", before)
            or re.search(r"(?:తుమ్ములు|దగ్గు)[^.!?\n]{0,35}[?]\s*(?:ఆహా|లేదు)[\s,]*$", before)):
        return "negative_verb_not_name"
    if _OBJECT_NAME.search(before):
        return "object_name_not_person"
    metadata = entity.get("recognition_metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    explicit_rule = (entity.get("recognizer") == "IndicNameContextRecognizer"
                     and metadata.get("boundary_evidence") in {"explicit_name", "identity_details"})
    if has_person_identity_context(text,entity) and plausible_person_surface(surface,explicit=True):
        return None
    if (normalized == "allen"
            and re.search(r"\b(?:positive|negative)\s*$", before, re.I)
            and re.match(r"['’]s\s+test\b", after, re.I)):
        return "clinical_eponym_not_name"
    if (normalized in _CLINICAL_ANALYTE and _ANALYTE_CONTEXT.search(before+surface+after)
            and not has_person_identity_context(text, entity)):
        return "clinical_analyte_not_name"
    if normalized in {"rich", "balanced", "healthy"} and re.match(r"\s+diet\b", after, re.I):
        return "dietary_expression_not_name"
    dietary_words = normalized.split()
    dietary_context = text[max(0, start-100):end+100]
    dietary_hits = {normalise_span(word) for word in dietary_context.split()} & _DIETARY_WORDS
    if (dietary_words and all(word in _DIETARY_WORDS for word in dietary_words)
            and (_DIET_CONTEXT.search(dietary_context) or len(dietary_hits) >= 2)):
        return "dietary_expression_not_name"
    role_value = re.sub(r"^(?:mr|mrs|ms|dr)\.?\s+", "", surface, flags=re.I)
    if normalise_span(role_value) in _GENERIC_PERSON_ROLES:
        return "generic_role_not_name"
    if _CLINICAL_HEADINGS.fullmatch(normalized):
        return "clinical_heading_not_name"
    if medication:
        return "medication_not_name"
    if normalized in _MEDICINE_NAMES:
        return "medication_not_name"
    collapsed = re.sub(r"[\s.,;:!?—-]+"," ",normalized).strip()
    if collapsed in _ORDINARY_PERSON_PHRASES:
        return "ordinary_speech_not_name"
    if collapsed=='मेरा demo':
        return 'ordinary_speech_not_name'
    if collapsed in _ACTIVITIES or any(re.search(r"(?<!\w)"+re.escape(activity)+r"(?!\w)",collapsed) for activity in _ACTIVITIES):
        return "activity_not_name"
    if normalized in {"surya","namaskar"} and re.search(r"\bsurya[\s.,;:!?—-]+namaskar\b",text[max(0,start-25):end+25],re.I):
        return "activity_not_name"
    if _LAB_EXPRESSION.search(surface) or _MEASUREMENT_EXPRESSION.search(surface):
        return "measurement_not_name"
    if normalized in _CLINICAL_WORDS:
        return "clinical_term_not_name"
    if re.fullmatch(r"[a-z]\.\s*\d+", normalized):
        return "enumeration_not_name"
    if re.fullmatch(r"i\.\s*(?:worked|make|wish)", normalized):
        return "sentence_fragment_not_name"
    enumerated_word = re.sub(r"^[a-z]\.\s+", "", normalized)
    if enumerated_word in _CLINICAL_WORDS:
        return "enumerated_clinical_term_not_name"
    words = collapsed.split()
    if len(words) > 4 or re.search(r"[,!?;]", surface):
        return "non_name_phrase"
    if normalized in _ORDINARY_PERSON_WORDS or words and all(w in _PERSON_STOP_WORDS for w in words):
        return "ordinary_speech_not_name"
    if normalized == "cord" and re.search(r"\bspinal[\s.,;:-]*$", before, re.I):
        return "anatomy_not_name"
    return None
