"""
Suppression list for IndicNER's non-name PERSON/LOCATION/ORGANIZATION spans.

IndicNER assigns a near-constant confidence (~0.993) to both a real name and a
discourse particle, so score thresholding cannot separate them. On the 50-row
transcript sample it tagged ``సరే`` ("OK") 20 times and ``हाँ`` ("yes") 11
times as PERSON, inflating the PERSON count and redacting ordinary speech.

This filter uses a closed vocabulary of common non-identifying speech. Three
classes are suppressed:

1. Discourse particles and pronouns (yes/no/OK/so/I/you).
2. Clinical and honorific role words (doctor/madam/sir/patient/nurse).
3. English clinical and everyday vocabulary transliterated into Indic script
   (past/old/test/regular/alcohol/diet/night/history), which dominates these
   code-switched consultation transcripts.

A span is dropped only when EVERY token in it is in the vocabulary, so
``ఆరవ్ పరీక్ష`` survives on the strength of ``ఆరవ్`` even though ``పరీక్ష`` is
listed, while the compound ``సరే మేడం`` ("OK madam") is dropped because both of
its tokens are. One unrecognised token is enough to keep the whole span: the
filter avoids dropping a mixed span merely because it contains a listed word.
Vocabulary membership is not proof that a word can never be a personal name;
recall must also be checked on reviewed, previously unseen transcripts.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Set

from ..taxonomy import canonical_entity_type

# --- 1. Discourse particles, pronouns, quantifiers -------------------------
_DISCOURSE = {
    # Telugu
    "సరే", "అవును", "ఏం", "కి", "మీరు", "నేను", "నీ", "ఆ", "అలాగే", "కాదు",
    "లేదు", "ఏం లేదు", "అవసరం", "మామూలుగా", "ఇప్పుడు", "చేయించాం", "వై",
    "తెలుస్తుంది", "చేయాలో అర్థం", "ఆ సరే",
    "హలో", "జస్ట్", "లాస్",
    # Hindi / Devanagari
    "हाँ", "हां", "अच्छा", "ओके", "ना", "नहीं", "हैं", "है", "सो", "और",
    "मैं", "आप", "क्या", "ठीक", "सारे", "इट", "आल्सो",
    # Kannada
    "ಹೌದು", "ಸರಿ", "ಇಲ್ಲ", "ನಾನು", "ನೀವು",
    "ಹಲೋ",
    # Tamil
    "ஆம்", "இல்லை", "சரி", "நான்", "நீங்கள்",
    "ஹலோ",
    # Bengali
    "হ্যাঁ", "না", "আমি", "আপনি", "ঠিক",
    # Malayalam
    "അതെ", "ഇല്ല", "ഞാൻ", "നിങ്ങൾ", "ശരി",
    "hello", "hi", "um", "uh", "just", "loss", "vision",
}

# --- 2. Roles, honorifics, forms of address --------------------------------
_ROLES = {
    "మేడం", "డాక్టర్", "సార్", "డాక్టరు", "హాయ్ డాక్టర్", "రోగి",
    "डॉक्टर", "मैडम", "सर", "मरीज", "नर्स",
    "ಡಾಕ್ಟರ್", "ಮೇಡಂ", "ಸರ್", "ವೈದ್ಯರು", "ರೋಗಿ",
    "டாக்டர்", "மேடம்", "சார்", "மருத்துவர்", "நோயாளர்",
    "ডাক্তার", "ম্যাডাম", "স্যার", "রোগী",
    "ഡോക്ടർ", "മാഡം", "സർ", "രോഗി",
    "madam", "doctor", "sir", "nurse", "patient",
}

# --- 3. English clinical/common vocabulary written in Indic script ---------
_TRANSLITERATED = {
    # Telugu
    "పాస్ట్", "ఓల్డ్", "టెస్ట్", "రెగ్యులర్", "ఆల్కహాల్", "ఫస్ట్", "ప్రీవియస్",
    "డైట్", "లైక్", "కంటిన్యూస్", "రిక్వైర్డ్", "నాస్ట్రల్స్", "రన్నింగ్ నోస్",
    "టైట్ రైస్", "ఐ వి", "స్కాన్", "ఫుల్", "అండ్", "ఇచ్చింగ్", "నార్మల్",
    "ఇన్వెస్టిగేషన్స్", "టాబ్లెట్", "ఎడ్యుకేట్", "న్యూట్రోఫిల్",
    # Tokens seen only inside compound spans ("running nose", "tight rice").
    "రన్నింగ్", "నోస్", "టైట్", "రైస్", "ఐ", "వి", "హాయ్", "చేయాలో", "అర్థం",
    # Kannada
    "ನೈಟ್", "ಪರ್ಸನಲ್", "ನೋ ಸ್ಮೋಕಿಂಗ್", "ಹಿಸ್ಟರಿ ಆಫ್ ಆಸ್ತಮಾ",
    "ನೋ", "ಸ್ಮೋಕಿಂಗ್", "ಹಿಸ್ಟರಿ", "ಆಫ್", "ಆಸ್ತಮಾ",
    # Devanagari
    "स्लोली", "टेस्ट", "नॉर्मल", "टैबलेट",
    # Latin clinical terms IndicNER mislabels in code-switched text
    "bowel", "scan", "tablet", "normal", "test", "past", "old", "regular",
    "alcohol", "diet", "night", "history", "personal", "continuous",
    "bladder", "giddiness", "cbc", "bp", "cbp", "temperature", "asthma",
    "fever", "cough", "cold", "pain", "sugar", "thyroid", "mother", "father",
    # Transliterated English function words. These dominate the code-switched
    # transcripts and were the bulk of what chunked IndicNER newly surfaced:
    # "so", "rather", "she", "given", "only", "already", "voice", "I am".
    "సో", "రథర్", "షి", "గివెన్", "ఓన్లీ", "ఎస్", "ఆల్రెడీ", "వాయిస్",
    "ఆర్", "టి", "యామ్", "మోర్", "వన్", "హర్", "దోస్", "వీక్స్", "మంత్",
    "బట్", "ఓకే", "ఇట్", "దట్", "నో", "యస్", "ఓహ్", "ఉమ్", "ఆల్సో",
    "ఐ యామ్", "మై నేమ్", "మై", "నేమ్", "షి హాస్", "హాస్", "గాట్",
    # Transliterated clinical vocabulary and role words seen in these rows.
    "మదర్", "ఫాదర్", "బిపి", "సిబిపి", "టెంపరేచర్", "ఆస్తమా", "మంట",
    "ట్రాక్ట్", "ల్యూకోసైట్స్", "కంప్రెషన్", "మాయిశ్చరైజ్డ్", "ఆయింట్మెంట్",
    # Kannada / Devanagari equivalents of the same function words.
    "ಸೋ", "ಶಿ", "ಓನ್ಲಿ", "ಒಕೆ", "आई", "एनीवे", "कफिंग", "ओके",
    # Transliterated numerals. The spoken-place and name-context rules capture
    # the word after their keyword, and "ఫ్రమ్ సిక్స్" ("from six") was being
    # redacted as an address.
    "వన్", "టూ", "త్రీ", "ఫోర్", "ఫైవ్", "సిక్స్", "సెవెన్", "ఎయిట్",
    "నైన్", "టెన్", "ఇయర్స్", "ఇయర్", "ఏజ్", "నెల", "డేస్", "వీక్",
    "ಒಂದು", "ಎರಡು", "एक", "दो", "तीन", "चार", "पांच", "छह", "चले",
    # Sentence-final politeness particles that trail a captured span.
    "అండి", "గారు", "నాన్న", "అమ్మా", "అమ్మ", "బాబు", "యు", "మీ",
    "ಅಂಡಿ", "ಸಾರ್", "जी", "साहब",
    # English function words that the "I'm from X" rule captured as places
    # ("from sitting", "from when").
    "sitting", "when", "where", "there", "here", "home", "work", "morning",
    "evening", "yesterday", "today", "tomorrow", "that", "this", "then",
    "సివియల్", "ఆంధ్రాలో",
    # Clinical vocabulary and ordinary verbs the free-text detectors labelled as
    # names. None can be a personal name in these transcripts.
    "బ్లడ్", "విజన్", "సీజర్స్", "పసల్స్", "హిస్టరీ", "షుగర్", "కౌంట్",
    "రిపోర్ట్", "రిజల్ట్", "ప్రెజర్", "వెయిట్", "హైట్", "లెవెల్", "కేసు",
    "కెన్", "రీడ్", "రైట్", "లెఫ్ట్", "సైడ్", "పెయిన్", "ప్రాబ్లం",
    "దగ్గరికి", "ఎందుకు", "వెళ్ళినప్పుడు", "ఇయ్యాల", "చెక్", "తగ్గిద్ది",
    "जा", "रहा", "रही", "स्लोली", "कर", "के", "लिए", "बाद",
    "ಬ್ಲಡ್", "ಶುಗರ್", "ರಿಪೋರ್ಟ್",
}

_SUPPRESS: Set[str] = {
    s.casefold() for s in (_DISCOURSE | _ROLES | _TRANSLITERATED)
}

# Danda/purna viram and the usual transcript punctuation carry no identity.
_STRIP = re.compile(r"^[\s\.,;:!?\-–—'\"“”‘’।॥|]+|[\s\.,;:!?\-–—'\"“”‘’।॥|]+$")
_COLLAPSE_WS = re.compile(r"\s+")

# Canonical aliases from every detector share the same non-name vocabulary.
FILTERED_ENTITY_TYPES = frozenset({"PERSON", "LOCATION", "ORGANIZATION"})

# The free-text types spaCy guesses at. Its structured findings (dates, URLs,
# phone numbers) are pattern-driven and stay, since those are script-agnostic
# and are exactly the identifiers that must never be missed.
SPACY_DROPPED_TYPES = frozenset({"PERSON", "LOCATION", "ORGANIZATION", "NRP"})

# A personal name in these transcripts runs to at most a given name plus a
# surname. Anything longer that spaCy has labelled on Indic script is a clause.
SPACY_MAX_NAME_TOKENS = 2


# Numeric labels that a clinical reading can be mistaken for. A blood pressure
# is not a card verification value; a dosage is not a postal code.
NUMERIC_LABELS = frozenset({
    "card_cvv", "card_number", "postal_code", "national_id_number", "account_number",
    "GOVERNMENT_ID", "US_BANK_NUMBER", "US_SSN", "US_DRIVER_LICENSE", "US_PASSPORT",
    "ICD10_CODE", "sensitive_date", "date_of_birth", "MEDICAL_RECORD_NUMBER",
    "HEALTH_PLAN_BENEFICIARY_NUM", "password", "username", "account_id",
    "license_number", "drivers_license_number", "bank_account", "phone_number",
    "PHONE_NUMBER", "MEDICAL_ID", "payment_card", "postal_code",
})

# Units that follow a measurement.
_CLINICAL_UNIT = re.compile(
    r"^\s*(?:mg|mcg|ug|ml|mmol|mol|g|gm|kg|lbs?|mmhg|mm|cm|dl|iu|units?|%|°|celsius|"
    r"bpm|/min|/dl|/l|beats|weeks?|days?|hours?|hrs?|months?|"
    r"ఎంజి|ఎంఎల్|గ్రామ్స్|కేజీ|కిలో|మిగ్రా|मिग्रा|मिलीग्राम)(?![A-Za-z])",
    re.IGNORECASE,
)

# Words that introduce a clinical reading, in the scripts of this corpus.
_CLINICAL_CONTEXT = re.compile(
    r"(?:\bbp\b|blood\s*pressure|pulse|temperature|temp\b|sugar|glucose|fasting|"
    r"hba1c|\bhb\b|haemoglobin|hemoglobin|tsh|wbc|rbc|platelet|neutrophil|lymphocyte|"
    r"creatinine|urea|cholesterol|spo2|saturation|weight|height|count|level|dose|dosage|"
    r"tablet|capsule|syrup|injection|vitamin|"
    r"బిపి|బీపీ|షుగర్|టెంపరేచర్|బ్లడ్|ల్యూకోసైట్స్|న్యూట్రోఫిల్|టాబ్లెట్|"
    r"మాత్ర|ప్లేట్లెట్స్|యూరియా|క్రియాటినిన్|కొలెస్ట్రాల్|"
    r"बीपी|नब्ज़|नब्ज|नाड़ी|शुगर|टैबलेट|ಬಿಪಿ)\s*\S{0,12}$",
    re.IGNORECASE,
)

# A blood pressure or ratio reading: 118/76, 150-85. Not applied to date
# labels, because a date is legitimately written the same way ("12/5").
_READING = re.compile(r"^\d{1,3}\s*[/\-]\s*\d{1,3}$")
_DATE_LABELS = frozenset({"sensitive_date", "date_of_birth", "document_date",
                          "expiration_date", "transaction_date"})

_UNIT_ANYWHERE = re.compile(
    r"(?<![A-Za-z])(?:mg|mcg|ug|gm|ml|mmol|mmhg|kg|g/dl|mg/dl|dl|iu|bpm|units?|cm|mm|"
    r"ఎంజి|గ్రామ్స్|కేజీ|కిలో|मिग्रा)(?![A-Za-z])|%|°",
    re.IGNORECASE,
)
_VALUE_WITH_UNIT = re.compile(
    r"\d+(?:[.,/]\d+)*\s*(?:mg|mcg|ug|gm|ml|mmol|mmhg|kg|g/dl|mg/dl|dl|iu|bpm|units?|cm|mm|ఎంజి|గ్రామ్స్|కేజీ|किलो|मिग्रा)(?![A-Za-z])|\d\s*[%°]",re.I)


def is_clinical_measurement(entity_type: str, text: str, start: int, end: int) -> bool:
    """True when a numeric span is a clinical reading rather than an identifier.

    The user's contract is that a wrong placeholder is acceptable but redacting
    non-PII is not, and clinical readings were the most damaging false positive
    in the output: ``118/76 mmHg`` and ``98.5`` labelled ``card_cvv``, ``100 mg``
    labelled a postal code. Destroying a blood pressure loses real clinical
    content and protects nobody.

    A span qualifies when it is mostly digits and either reads as a ratio, is
    followed by a clinical unit, or follows a word that introduces a measurement.
    """
    entity_type = entity_type.casefold()
    if entity_type not in {label.casefold() for label in NUMERIC_LABELS}:
        return False
    span = text[start:end].strip()
    if not span:
        return False
    digits = sum(1 for ch in span if ch.isdigit())
    if not digits:
        return False
    # Unit-bearing values must be checked before density: "5 mg" is only
    # 25% digits, and "3MG" has no word boundary before the unit.
    if _VALUE_WITH_UNIT.search(span) or _CLINICAL_UNIT.match(text[end:end + 24]):
        return True
    if digits / len(span) < 0.4:
        return bool(_CLINICAL_CONTEXT.search(span))
    if entity_type not in _DATE_LABELS and _READING.match(span):
        return True
    # The unit is often inside the span rather than after it — GLiNER returns
    # "118/76 mmHg" and "100 mg" whole. Anything carrying its own unit is a
    # measurement, not an identifier.
    if _CLINICAL_CONTEXT.search(text[max(0, start - 40):start]):
        return True
    # An Indian postal code is exactly six digits; anything shorter labelled as
    # one is a number from ordinary speech.
    if entity_type == "postal_code" and len(re.sub(r"\D", "", span)) != 6:
        return True
    return False


def normalise_span(text: str) -> str:
    """Normalise a detected span for suppression-list comparison."""
    # NFKC folds the compatibility forms that differ across transcript sources.
    folded = unicodedata.normalize("NFKC", text)
    folded = _STRIP.sub("", folded)
    return _COLLAPSE_WS.sub(" ", folded).casefold()


def suppression_enabled() -> bool:
    """Whether filler suppression runs. On by default.

    Suppression trades recall for readability: every span it drops is a span
    that is no longer redacted, so this defaulted to off under the original
    contract that a wrongly placed placeholder is tolerable while an exposed
    identifier is not. Sentence-chunking IndicNER made that untenable — it
    surfaced enough transliterated English discourse words ("so", "rather",
    "she", "only") that ordinary speech was being replaced with placeholders
    and the redacted transcript stopped being usable. The vocabulary here is a
    closed list that reduces observed discourse errors, so suppression now runs by
    default. Set PII_SUPPRESS_FILLERS=0 to keep these candidates for diagnostics.
    This does not bypass span validation or overlap selection. The detectors
    still apply their own confidence gates.
    """
    return os.getenv("PII_SUPPRESS_FILLERS", "1").strip().lower() in {"1", "true", "yes", "on"}


def indic_script_fraction(text: str) -> float:
    """Share of letters in the Indic Unicode blocks, not every non-ASCII letter.

    Accented Latin names such as Senol must not be routed as Indic text.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if 0x0900 <= ord(ch) <= 0x0D7F) / len(letters)


def is_english_model_hallucination(recognizer: str, entity_type: str, text: str) -> bool:
    """True for an English-only recognizer's span over Indic-script text.

    The Presidio backbone is spaCy ``en_core_web_sm``. On Devanagari, Telugu,
    Kannada, Tamil, Bengali or Malayalam it has no training signal at all, yet
    it still returns PERSON/LOCATION/ORGANIZATION spans at its flat 0.85
    default — 1,104 of 1,228 spans over a 50-row production sample, including a
    37-character clause ("వస్తున్నాడని హాస్పిటల్ లో ఉంచుకున్నాం") labelled
    PERSON. None of it carries information; it is the single largest source of
    junk redaction in the output.

    Measured against the real-transcript benchmark, spaCy scored **1.4%**: 138
    spans over 49 rows, of which 2 were real PII. Every Indic-script free-text
    span it produces is therefore dropped. An earlier version kept spans of one
    or two tokens, on the theory that spaCy occasionally lands a short name —
    it does, but at that rate the two it finds cost 136 false positives, and
    both are recoverable from the gazetteer instead.

    Latin-script names in Hinglish rows still reach spaCy untouched, as do the
    Indic-aware detectors: IndicNER, IndicNameContextRecognizer and the
    gazetteer.
    """
    if not suppression_enabled():
        return False
    if "spacy" not in recognizer.casefold():
        return False
    if canonical_entity_type(entity_type) not in SPACY_DROPPED_TYPES:
        return False
    return indic_script_fraction(text) > 0.5


def is_discourse_particle(text: str) -> bool:
    """Distinguish unambiguous conversational fillers from ambiguous names."""
    return suppression_enabled() and normalise_span(text) in _DISCOURSE


def is_suppressed(entity_type: str, text: str) -> bool:
    """True when this span is a known non-name and should not be redacted."""
    if not suppression_enabled():
        return False
    if canonical_entity_type(entity_type) not in FILTERED_ENTITY_TYPES:
        return False
    normalised = normalise_span(text)
    if not normalised:
        return True  # A span of pure punctuation identifies nobody.
    if normalised in _SUPPRESS:
        return True
    # IndicNER frequently returns a run of adjacent filler words as one span
    # ("సరే మేడం"). Suppress it only when no token could be a name.
    tokens = [token for token in normalised.split(" ") if token]
    return len(tokens) > 1 and all(token in _SUPPRESS for token in tokens)


def filter_indic_entities(entities: Iterable[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Drop IndicNER spans whose whole surface form is a known non-name."""
    return [
        entity
        for entity in entities
        if not is_suppressed(entity.get("entity_type", ""), text[entity["start"]:entity["end"]])
    ]
