"""
Script-independent recognizers for Indian and clinical identifiers.

The Stage 9 detection stack is script-asymmetric: GLiNER2 only contributes on
Latin-script text, so on Devanagari/Telugu/Tamil/Kannada/Bengali/Malayalam rows
IndicNER is the sole model-based detector. Structured identifiers, however, are
written in ASCII regardless of the surrounding script (Aadhaar digits, a PAN, a
UPI handle, an MRN), so deterministic patterns recover them in every language.

Presidio's ``PatternRecognizer`` reports the whole regex match as the entity
span. Some identifiers here are only PII *because* of an adjacent keyword (a
bare ``0000`` is a UPI PIN only next to "UPI PIN"), and Presidio's context
enhancement cannot help: it lemmatises with the English spaCy engine, which
does not process Indic script. ``KeyedGroupRecognizer`` therefore matches the
keyword and the value together but reports only the capture group, so the
keyword itself is left readable in the redacted transcript.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from ..policy.indic_filters import is_discourse_particle, is_suppressed
from ..policy.semantic_sanity import implausible_person, implausible_place, normalize_place_span
from ..policy.context_filters import (
    age_is_duration,
    valid_calendar_literal,
    plausible_person_surface,
    has_historical_age_context,
)
from .clinical_calendar import ClinicalCalendarDateRecognizer
from .hindi_age import iter_hindi_age_spans
from ..data_assets import require_data_path
from ..telugu_utils import TELUGU_NUMBER_PATTERN, parse_telugu_number, split_telugu_suffix


class KeyedGroupRecognizer(EntityRecognizer):
    """Match a keyword plus its value, but report only the value's span.

    ``regex`` must define exactly one capture group holding the identifier.
    """

    def __init__(
        self,
        supported_entity: str,
        regex: str,
        score: float,
        name: str,
        flags: int = re.IGNORECASE | re.UNICODE,
    ):
        self.regex = re.compile(regex, flags)
        self.score = score
        super().__init__(
            supported_entities=[supported_entity],
            name=name,
            supported_language="en",
        )

    def load(self) -> None:
        """No model to load; the compiled pattern is built in __init__."""

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        entity_type = self.supported_entities[0]
        if entities and entity_type not in entities:
            return []
        results = []
        for match in self.regex.finditer(text):
            start, end = match.span(1)
            if start < 0:
                continue
            # These are deliberately recall-first rules: they capture whatever
            # word follows their keyword. Measured against the real-transcript
            # benchmark that made them a leading false-positive source — "from
            # సిక్స్" ("from six"), "డాక్టర్: సరే" ("doctor: OK"). The closed
            # suppression vocabulary already knows these are not names, so a
            # capture that is entirely non-name words is dropped here.
            if is_suppressed(entity_type, text[start:end]) or is_suppressed("PERSON", text[start:end]):
                continue
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    score=self.score,
                    analysis_explanation=None,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id},
                )
            )
        return results


def _verhoeff_valid(digits: str) -> bool:
    """Verhoeff checksum used by Aadhaar numbers."""
    d_table = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
        (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
        (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
        (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
        (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
    )
    p_table = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
        (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
        (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
        (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
    )
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        checksum = d_table[checksum][p_table[index % 8][int(digit)]]
    return checksum == 0


class AadhaarRecognizer(PatternRecognizer):
    """Aadhaar (Indian government ID) — 12 digits, usually grouped 4-4-4.

    Verhoeff validation is advisory only: it raises confidence for a real
    number but never drops a candidate, because synthetic and redacted-in-
    source records routinely carry structurally valid but checksum-invalid
    numbers that must still be treated as PII.
    """

    PATTERNS = [
        Pattern(name="aadhaar_grouped", regex=r"\b[2-9]\d{3}[ -]\d{4}[ -]\d{4}\b", score=0.6),
        Pattern(name="aadhaar_plain", regex=r"\b[2-9]\d{11}\b", score=0.4),
    ]
    CONTEXT = ["aadhaar", "aadhar", "uidai", "आधार", "ఆధార్", "ஆதார்", "ಆಧಾರ್", "আধার", "ആധാർ"]

    def __init__(self):
        super().__init__(
            supported_entity="GOVERNMENT_ID",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )

    def validate_result(self, pattern_text: str):
        digits = re.sub(r"\D", "", pattern_text)
        if len(digits) != 12:
            return None
        return True if _verhoeff_valid(digits) else None

    def analyze(self,text,entities,nlp_artifacts=None,**kwargs):
        results=super().analyze(text,entities,nlp_artifacts,**kwargs)
        field=re.compile(r"(?:aadhaar|aadhar|government\s*id|national\s*id|आधार|గుర్తింపు|அடையாள\s*எண்|ಗುರುತಿನ\s*ಸಂಖ್ಯೆ|পরিচয়\s*নম্বর|തിരിച്ചറിയൽ)\s*$",re.I)
        for r in results:
            if field.search(text[max(0,r.start-65):r.start]):
                r.recognition_metadata={**(r.recognition_metadata or {}),'recognizer_name':self.name,
                                        'boundary_evidence':'explicit_government_id'}
        return results


class PANRecognizer(PatternRecognizer):
    """Indian Permanent Account Number — AAAAA9999A."""

    PATTERNS = [
        Pattern(name="pan", regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", score=0.8),
    ]
    CONTEXT = ["pan", "permanent account number", "income tax", "पैन", "पॅन"]

    def __init__(self):
        super().__init__(
            supported_entity="TAX_ID",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


class LabelledPanRecognizer(KeyedGroupRecognizer):
    """Redact a value in a PAN field even when its entered format is invalid.

    Field context determines the privacy type. ``format_valid`` reports only
    the usual letter/digit shape, not issuance, ownership, or authenticity.
    """

    def __init__(self):
        super().__init__(
            supported_entity="tax_id",
            regex=(
                r"(?:\bpan[ \t]+(?:number\b|no\.?(?![A-Za-z]))|\bpan[ \t]*(?=[:=#])|"
                r"\bpermanent[ \t]+account[ \t]+number\b|"
                r"पैन[ \t]+(?:नंबर|नम्बर|संख्या)|పాన్[ \t]+(?:నంబర్|నెంబర్))"
                r"[ \t]*(?:is[ \t]+)?[:=#]?[ \t]*"
                r"([A-Z0-9]+(?:[-_][A-Z0-9]+)*)(?![A-Z0-9_-])"
            ),
            score=.95,
            name="LabelledPanRecognizer",
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for result in super().analyze(text, entities, nlp_artifacts):
            value = text[result.start:result.end]
            compact = re.sub(r"[-_]", "", value)
            if not 6 <= len(compact) <= 20 or not any(char.isdigit() for char in compact):
                continue
            result.recognition_metadata.update(
                boundary_evidence="explicit_tax_id",
                format_valid=bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value, re.I)),
                validation="format_only",
            )
            results.append(result)
        return results


class UPIHandleRecognizer(PatternRecognizer):
    """UPI virtual payment address, e.g. name.demo@okhdfcbank.

    Presidio's EMAIL_ADDRESS recognizer requires a registered-looking TLD, so a
    UPI handle (whose right-hand side is a PSP handle, not a domain) is missed
    entirely. Kept separate from EMAIL_ADDRESS so the taxonomy stays honest.
    """

    PATTERNS = [
        Pattern(
            name="upi_vpa",
            regex=r"\b[A-Za-z0-9](?:[A-Za-z0-9._-]{1,48}[A-Za-z0-9])?@[A-Za-z][A-Za-z0-9.-]{1,48}[A-Za-z0-9]\b",
            score=0.6,
        ),
    ]
    CONTEXT = ["upi", "vpa", "virtual payment", "payment id", "handle"]

    def __init__(self):
        super().__init__(
            supported_entity="ACCOUNT_ID",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


class IndianPhoneRecognizer(PatternRecognizer):
    """Indian mobile numbers, including the spaced +91 90000 00001 form."""

    PATTERNS = [
        Pattern(name="in_phone_intl", regex=r"(?:\+91|0091)[ -]?\d{5}[ -]?\d{5}\b", score=0.7),
        Pattern(name="in_phone_local", regex=r"(?<!\d)[6-9]\d{9}(?!\d)", score=0.5),
    ]
    CONTEXT = ["phone", "mobile", "contact", "फोन", "मोबाइल", "ఫోన్", "தொலைபேசி", "ಫೋನ್", "ফোন", "ഫോൺ"]

    def __init__(self):
        super().__init__(
            supported_entity="PHONE_NUMBER",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )


# Unicode blocks for the scripts this deployment handles. A "word" for name
# purposes is a run of letters in one of these blocks (or Latin), because the
# Indic scripts here do not mark case and spaCy/GLiNER give no usable signal.
_INDIC_LETTER = (
    r"\u0900-\u097F"  # Devanagari
    r"\u0980-\u09FF"  # Bengali
    r"\u0A00-\u0A7F"  # Gurmukhi
    r"\u0A80-\u0AFF"  # Gujarati
    r"\u0B00-\u0B7F"  # Odia
    r"\u0B80-\u0BFF"  # Tamil
    r"\u0C00-\u0C7F"  # Telugu
    r"\u0C80-\u0CFF"  # Kannada
    r"\u0D00-\u0D7F"  # Malayalam
)
_NAME_WORD = rf"[{_INDIC_LETTER}A-Za-z][{_INDIC_LETTER}A-Za-z']*"
_NAME_CAPTURE = rf"(?:[A-Za-z]\.[ \t]+){{0,2}}{_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}}"
_IDENTITY_DETAIL = re.compile(
    r"[^.!?;\n।॥:]{0,45}?(?:\bphone\b|\bmobile\b|\bcontact\b|\baadhaar\b|\bMRN\b|"
    r"फोन|आधार|தொலைபேசி|அடையாள|ఫోన్|ఆధార్|గుర్తింపు|ದೂರವಾಣಿ|ಫೋನ್|ಗುರುತ|ফোন|পরিচয়|ഫോൺ|തിരിച്ചറിയൽ)",
    re.IGNORECASE,
)
_CLINICAL_NON_NAMES = {"conscious", "oriented", "advised", "reports", "complains", "feels", "symptoms",
                       "patient", "doctor", "nurse", "improved", "stable", "rest", "tablet", "medicine", "and"}
_NAME_BOUNDARIES = _CLINICAL_NON_NAMES | {
    "ka", "ki", "ko", "hai", "mobile", "phone", "contact", "age", "dob", "is", "was", "has", "gave", "me",
    "well", "better", "take", "reduce", "for", "to", "please", "mg", "years", "old",
    "అండ్", "కదా", "సార్", "గారు", "అండి", "ఇస్", "ఓల్డ్", "చెప్తారు", "ముందుగా", "అంతా", "పప్పు", "వాయిస్", "voice",
    "ఫోన్", "నంబర్", "నెంబర్", "ఏజ్", "పేరు", "का", "की", "को", "फोन", "फ़ोन", "नंबर", "नम्बर", "नाम",
    "தொலைபேசி", "எண்", "அவர்களின்", "ಅವರ", "ಫೋನ್", "ನಂಬರ್", "ಸಂಖ್ಯೆ", "ফোন", "নম্বর", "ഫോൺ", "നമ്പർ",
    "డెమో", "గుర్తింపు", "మెడికల్", "రికార్డు", "டெமோ", "அடையாள", "மற்றும்",
    "ಅವರ", "ಡೆಮೋ", "ಗುರುತಿನ", "ಮತ್ತು", "ഡെമോ", "തിരിച്ചറിയൽ", "ডেমো", "পরিচয়", "और", "aur", "mrn",
    "ఉంటారు", "వచ్చిండు", "వచ్చాడు", "కదండీ", "గారికి", "గారా", "గారు", "sir", "ma'am",
    "क्या", "है", "ఆ", "here",
}
_NAME_QUESTION_WORDS = {"ఏంటి", "ఏంది", "ఏంటమ్మా", "ఏందయ్యా", "ఎట్లా", "ఎవరు", "అండి", "क्या", "कौन"}
_NAME_NARRATIVE_WORDS = {"వస్తుంది", "చెప్తారు", "మారుద్ది", "మర్చిపోయాను",
                         "గుర్తుపెట్టుకోలే", "చెప్పారన్నమాట", "వేపించుకోవాలని", "speaker", "so"}
_NAME_TITLES = r"(?:(?:Dr\.?|doctor|డాక్టర్|डॉक्टर|டாக்டர்|ಡಾಕ್ಟರ್)\s+)"
_OBJECT_NAME_FRAME = re.compile(r"(?:medicine|drug|tablet|test|దాని|మందు|మాత్ర)\s+(?:name|పేరు)\s*(?:is|:)?\s*$",re.I)

# A first-person copula can describe a symptom, occupation, or origin instead
# of a name. Require the adjacent demographic field below as well as rejecting
# these grammatical/predicate words; do not use a closed list of personal names.
_HINDI_INTRO_PREDICATES = frozenset({
    "ठीक", "बीमार", "परेशान", "खुश", "दुखी", "स्वस्थ", "तैयार", "बहुत",
    "डॉक्टर", "मरीज", "मरीज़", "नर्स", "शिक्षक", "अध्यापक", "छात्र", "छात्रा",
    "थका", "थकी", "भूखा", "भूखी", "गर्भवती", "अकेला", "अकेली", "इंजीनियर",
    "से", "में", "पर", "का", "की", "को", "रहा", "रही", "रहता", "रहती",
})
_TELUGU_INTRO_PREDICATES = re.compile(
    r"^(?:(?:రోగి|పేషెంట్|డాక్టర్|నర్సు|వైద్యుడు|వైద్యురాలు|విద్యార్థి)(?:ని|ను|గా)?|"
    r"బాగున్నాను|బాగోలేదు|అనారోగ్యంతో|ఇక్కడ|అక్కడ|ఉంటాను|ఉంటాము|"
    r"(?:చికిత్స|ఆసుపత్రి|ఇంటి|ఒత్తిడి)లో|"
    r".*(?:స్తున్నా(?:ను|ము|ం)|తున్నా(?:ను|ము|ం)|వచ్చా(?:ను|ము|ం)|నుంచి|నుండి|లేను|లేదు|కాదు))$"
)

# Words that introduce a person in a clinical transcript, across the supported
# languages. A name following one of these is captured even when no NER model
# recognises it -- GLiNER2 returns nothing at all for Indic-script names, and
# IndicNER misses names built from common nouns.
_NAME_INTRODUCERS = "|".join([
    # Telugu
    "డాక్టర్", "రోగి", "పేషెంట్", "పేరు", "మరియు రోగి",
    # Hindi / Devanagari
    "डॉक्टर", "मरीज", "मरीज़", "रोगी", "नाम", "पेशेंट",
    # Tamil
    "மருத்துவர்", "நோயாளர்", "பெயர்",
    # Kannada
    "ವೈದ್ಯರು", "ರೋಗಿ", "ಹೆಸರು", "ಡಾಕ್ಟರ್",
    # Bengali
    "ডাক্তার", "রোগী", "নাম",
    # Malayalam
    "ഡോക്ടർ", "രോഗി", "പേര്",
    # Latin / code-switched
    r"patient", r"doctor", r"dr\.?", r"name is", r"my name is", r"naam",
    r"this\s+is", r"దిస్\s*ఇస్",
])


# Telugu's quotative complementizer closes a name value ("called X", "Dr X
# said ..."). It belongs to the sentence, including after a multiword name.
_NAME_BOUNDARIES = _NAME_BOUNDARIES | {"అని"}


class IndicNameContextRecognizer(KeyedGroupRecognizer):
    """Capture names in explicit fields or independently supported clinical frames.

    A generic role label needs gazetteer support or adjacent identification
    details. Sentence/speaker boundaries and ordinary clinical words stop the
    capture, including when a neural model proposes a broader span.
    """

    def __init__(self):
        explicit = "|".join([
            r"\b(?:my\s+)?name(?:\s+is\s+|\s*:\s*)", r"\bnaam\s*(?:is|hai|:)\s*",
            r"పేరు\s*(?:అని|:)?\s*", r"नाम\s*(?:है|:)\s*", r"नाम\s+",
            r"(?:பெயர்|பேரு)\s*:?\s*", r"ಹೆಸರು\s*:?\s*", r"নাম\s*:?\s*", r"പേര്\s*:?\s*",
        ])
        self.explicit_regex = re.compile(rf"(?:{explicit})(?:{_NAME_TITLES})?({_NAME_CAPTURE})", re.I)
        self.question_regex = re.compile(
            rf"(?:what(?:'s|\s+is)\s+your\s+name|(?:మీ|నీ|మీరు|అబ్బాయి|రోగి|పేషెంట్)?\s*(?:పూర్తి[ \t]+)?"
            rf"(?:పేరేంటి|పేరేమిటి|పేరు\s*(?:ఏంటి|ఏమిటి|ఎంటి|ఏంది|ఏంటమ్మా|ఏందయ్యా))|మీ\s*పేరు|"
            rf"(?:உங்களோட\s+)?(?:பேரு|பெயர்)|आपका\s+नाम(?:\s+क्या\s+है)?)"
            rf"\s*[?.:]+\s*(?:my\s+name\s+is\s+|I\s+am\s+|నా[ \t]+పేరు[ \t]*[:=]?[ \t]+)?(?:{_NAME_TITLES})?({_NAME_CAPTURE})",re.I)
        self.honorific_regex = re.compile(
            rf"(?<!\w)({_NAME_WORD})[ \t]+(?:గారు|గారా|గారికి)(?!\w)")
        self.recipient_regex = re.compile(
            rf"\b(?:doctor|dr\.?|nurse)\s+(?:gave|prescribed)\b[^.!?\n]{{1,90}}?\bto\s+({_NAME_CAPTURE})",re.I)
        # Unlisted names need identity evidence, rather than a gazetteer entry.
        self.self_intro_regex = re.compile(
            rf"(?:నేను|ನಾನು|நான்|मैं|હું|\bI\s+am)\s+({_NAME_CAPTURE})\s*,\s*"
            rf"(?:మీ|ನಿಮ್ಮ|உங்கள்|your|आपका|તમારો)\s+"
            rf"(?:వైద్య\s+సహాయక|డాక్టర్|వైద్యురాలు|ವೈದ್ಯ|மருத்துவ|डॉक्टर|doctor|medical\s+assistant)", re.I)
        self.hindi_intro_regex = re.compile(
            rf"(?<![{_INDIC_LETTER}A-Za-z])मैं[ \t]+"
            rf"({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}}?)[ \t]+(?:हूँ|हूं|है)"
            rf"(?=[ \t]*(?:[.!?।॥]|$))"
        )
        self.hindi_identity_followup = re.compile(
            r"^[ \t]*[.!?।॥]+\s*(?:मेरी|मेरा)[ \t]+(?:उम्र|आयु|जन्म[ \t]*तिथि)(?![\u0900-\u097F])"
        )
        self.telugu_intro_regex = re.compile(
            rf"(?<![{_INDIC_LETTER}A-Za-z])నేను[ \t]+"
            rf"({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}}?)(?:[ \t]+(?:ని|ను))?"
            r"(?=[ \t]*(?:[.!?।॥]|$))"
        )
        self.telugu_identity_followup = re.compile(
            r"^[ \t]*[.!?।॥]+\s*నా[ \t]+(?:వయస(?:్స)?ు|పుట్టిన[ \t]+తేదీ)(?![\u0C00-\u0C7F])"
        )
        self.opening_intro_regex = re.compile(
            rf"(?:^|(?<=[.!?।\n]))\s*({_NAME_CAPTURE})[.,।\s]*(?:అ\s+|ఆ[.,]\s*)?"
            rf"(?:ఏం\s+ఇబ్బంది(?:\s+అని)?\s+వచ్చారు|చెప్పండి[.,\s]+ఏమైంది\s+మీకు)", re.I)
        self.prescriber_regex = re.compile(
            rf"(?:டாக்டர்|மருத்துவர்|డాక్టర్|ಡಾಕ್ಟರ್|डॉक्टर)\s+({_NAME_WORD})\s+"
            rf"(?=[^.!?\n]{{0,45}}\bprescrib(?:e|ed|ing)\b)", re.I)
        super().__init__(
            supported_entity="PERSON",
            # The name is captured inside a lookahead so the match itself is
            # zero-width. Consecutive introducers ("డాక్టర్: రోగి ఆరవ్ పరీక్ష")
            # then each fire, and their overlapping captures together cover the
            # whole name instead of the scan consuming the first one.
            regex=rf"(?:{_NAME_INTRODUCERS})[ \t]*[:\-,]?[ \t]+(?=({_NAME_CAPTURE}))",
            # Above spaCy's flat 0.85. This span is anchored on an explicit
            # name-introducer, so it is strictly better evidence than the
            # English backbone guessing at Indic script — and Presidio resolves
            # overlaps by score. At 0.6 spaCy's longer, wronger span
            # ("আরব পরীক্ষা-এর ডেমো" for the name "আরব পরীক্ষা") won the
            # overlap and swallowed the precise one.
            score=0.9,
            name="IndicNameContextRecognizer",
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and "PERSON" not in entities:
            return []
        results = []
        names = load_person_name_gazetteer()
        for pattern, explicit in ((self.explicit_regex, True),(self.question_regex, True),
                                  (self.recipient_regex, True),(self.self_intro_regex, True),
                                  (self.hindi_intro_regex, True),
                                  (self.telugu_intro_regex, True),
                                  (self.opening_intro_regex, True),(self.prescriber_regex, True),
                                  (self.regex, False),
                                  (self.honorific_regex, False)):
            for match in pattern.finditer(text):
                start, end = match.span(1)
                if pattern is self.hindi_intro_regex and (
                    not self.hindi_identity_followup.match(text[match.end():match.end()+100])
                    or any(word in _HINDI_INTRO_PREDICATES for word in match.group(1).split())
                ):
                    continue
                if pattern is self.telugu_intro_regex and (
                    not self.telugu_identity_followup.match(text[match.end():match.end()+100])
                    or any(_TELUGU_INTRO_PREDICATES.fullmatch(word) for word in match.group(1).split())
                ):
                    continue
                if pattern is self.regex:
                    prelude=re.match(r"chart\s+mein\s+",text[start:end],re.I)
                    if prelude:
                        start+=prelude.end()
                # Dots belong to single-letter initials only; a sentence-final
                # stop cannot join names to the next sentence or speaker.
                initial_periods = {start + m.end(1) for m in re.finditer(
                    r"(?:^|[ \t])([A-Za-z])\.(?=[ \t]+[^\W\d_])", text[start:end])}
                for i in range(start, end):
                    if unicodedata.category(text[i]).startswith("P") and text[i] != "'" and i not in initial_periods:
                        end = i
                        break
                surface = text[start:end].rstrip()
                end = start + len(surface)
                if _OBJECT_NAME_FRAME.search(text[max(0,start-70):start]):
                    continue
                # Stop before predicates, identity labels, and politeness
                # particles, including captures longer than two words.
                for word in re.finditer(r"\S+",surface):
                    if word.group().strip(".'’").casefold() in _NAME_BOUNDARIES and (word.start()>0 or pattern is not self.explicit_regex):
                        end = start+word.start()
                        surface = text[start:end].rstrip()
                        end = start+len(surface)
                        break
                words = surface.split()
                strong_value_field = pattern in {self.hindi_intro_regex, self.telugu_intro_regex} or bool(re.search(
                    r"(?:\b(?:patient\s+)?(?:full\s+)?name\s*:|my\s+name\s+is|"
                    r"నా\s*పేరు|పేరు\s*:|பெயர்\s*:)\s*$", text[max(0,start-70):start], re.I))
                if not strong_value_field and words and (words[0].casefold() in _NAME_QUESTION_WORDS or
                                                        words[0].casefold() in _NAME_NARRATIVE_WORDS):
                    continue
                if not words or not words[0][0].isalpha() or (pattern is not self.explicit_regex and
                        (is_suppressed("PERSON", words[0]) or words[0].casefold() in _CLINICAL_NON_NAMES)):
                    continue
                identity_frame = bool(_IDENTITY_DETAIL.match(text[end:end + 65]))
                if identity_frame and not explicit and len(words)>1:
                    for suffix in ('യുടെ','க்கு'):
                        if words[-1].endswith(suffix) and len(words[-1])-len(suffix)>=3:
                            end-=len(suffix)
                            surface=text[start:end]
                            words=surface.split()
                            break
                listed_pair = len(words)>1 and unicodedata.normalize("NFKC",words[-1]).casefold() in names
                capitalized = all(w[0].isascii() and w[0].isupper() for w in words)
                honorific = pattern is self.honorific_regex
                if honorific and (words[0] in {"మరో", "వాళ్ళు", "కలెక్టర్", "టీచర్", "మేడం", "మంత్రి", "ప్రిన్సిపల్", "అన్నా"}
                                  or words[0].casefold() in _NAME_BOUNDARIES):
                    continue
                if not explicit and not identity_frame and not honorific and unicodedata.normalize("NFKC", words[0]).casefold() not in names and not (listed_pair or capitalized):
                    continue
                if not plausible_person_surface(surface,explicit=explicit or identity_frame):
                    continue
                # Drop a known discourse/clinical word at the tail, rather than
                # expanding a supported name into the surrounding sentence.
                # A filler vocabulary is not proof that a plausible surname
                # is invalid in an explicit personal-name field. Retain the
                # whole name there; ordinary stop words still fail plausibility.
                explicit_full_name = (strong_value_field and plausible_person_surface(surface)
                                      and not is_discourse_particle(words[-1]))
                if len(words) > 1 and (words[-1].casefold() in _CLINICAL_NON_NAMES or
                                      (not identity_frame and not explicit_full_name and
                                       (is_suppressed("PERSON", words[-1]) or
                                        not plausible_person_surface(words[-1])))):
                    end = start + len(surface.rsplit(None, 1)[0])
                results.append(RecognizerResult("PERSON", start, end, self.score,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "boundary_evidence": "patient_introduction" if pattern in
                                          {self.self_intro_regex, self.hindi_intro_regex, self.telugu_intro_regex,
                                           self.opening_intro_regex} else
                                          "identity_relation" if pattern is self.prescriber_regex else
                                          "explicit_name" if explicit else
                                          "person_honorific" if honorific else
                                          "identity_details" if identity_frame else "listed_name_after_role"}))
        return list({(r.start, r.end): r for r in reversed(results)}.values())


class TeluguNameCueRecognizer(EntityRecognizer):
    """Names supplied by Telugu identity cues, independent of a name list.

    The cue is evidence of a person's identity; only its value is emitted.
    A detachable case ending is left outside the span. If a spelling change
    makes that ending inseparable, the original token remains intact.
    """

    _WORD = re.compile(rf"(?:[{_INDIC_LETTER}A-Za-z][{_INDIC_LETTER}A-Za-z'’]*)")
    _CUES = (
        (re.compile(r"(?<![\u0C00-\u0C7FA-Za-z])(?:నా|మా|నీ|మీ)\s+(?:అమ్మ|నాన్న|అబ్బాయి|అమ్మాయి|కొడుకు|కూతురు|భర్త|భార్య|తండ్రి|తల్లి)\s+పేరు\s*[:：=-]?\s*"), "explicit_name"),
        (re.compile(r"(?<![\u0C00-\u0C7FA-Za-z])(?:నా|మా|నీ|మీ|తండ్రి|తల్లి|భర్త|భార్య|రోగి|పేషెంట్)\s+పేరు\s*[:：=-]?\s*|(?<![\u0C00-\u0C7FA-Za-z])పేరు\s*[:：]\s*"), "explicit_name"),
        (re.compile(r"(?<![\u0C00-\u0C7FA-Za-z])(?:మా|నా)\s+(?:అమ్మ|నాన్న|అబ్బాయి|అమ్మాయి|కొడుకు|కూతురు|భర్త|భార్య)\s+"), "identity_relation"),
        (re.compile(r"(?<![\u0C00-\u0C7FA-Za-z])(?:నేను|నన్ను)\s+"), "patient_introduction"),
        (re.compile(r"(?<![A-Za-z])(?:naa?|ma|mee)\s+peru\s*(?:is|:|=)?\s+|(?<![A-Za-z])nenu\s+|(?<![A-Za-z])nannu\s+", re.I), "patient_introduction"),
        (re.compile(r"(?<![\u0C00-\u0C7FA-Za-z])(?:శ్రీమతి|శ్రీ|డాక్టర్|డా\.)\s+"), "person_honorific"),
    )
    _PREDICATE = re.compile(
        r"(?:.*(?:తున్నాను|తున్నాడు|తున్నారు|స్తున్నాను|స్తున్నాడు|చేస్తాను|వచ్చాను|ఉంటాను|ఉన్నాను|అంటారు|చెబుతారు|చెప్తారు|కాదు|లేదు)|"
        r"(?:బాగున్నాను|బాగున్నాడు|జ్వరం|అనారోగ్యం|డాక్టర్|వైద్యుడు|వైద్యురాలు|నర్సు|పేషెంట్|రోగి|ఇక్కడ|అక్కడ|చాలా|మంచి|పేరు|వయసు|వయస్సు|ఊరు|నుండి|నుంచి|లో|కు|కి|ని|తో|గారు|అండి|గారూ|ఏంటి|ఏమిటి|ఎంటి|అని|పిలుస్తారు))$"
    )
    _ROMAN_STOP = frozenset({"ni", "nu", "garu", "andi", "peru", "vayasu", "doctor", "nurse", "patient", "fever", "fine", "from", "lo", "ki", "ku", "to", "untanu", "antaru", "matladutunnanu"})
    _PRONOUNS = frozenset({"నేను", "నా", "నాకు", "నన్ను", "మేము", "మా", "మీ", "నీ", "nenu", "naa", "na", "nannu", "ma", "mee"})
    _ROLES = frozenset({"డాక్టర్", "వైద్యుడు", "వైద్యురాలు", "నర్సు", "రోగి", "పేషెంట్", "విద్యార్థి", "టీచర్"})
    _PLACES = frozenset({"ఇల్లు", "ఇంటి", "ఆసుపత్రి", "గది", "గ్రామం", "నగరం", "పట్టణం"})
    # A first-person pronoun or a professional title occurs throughout clinical
    # dialogue.  For those weak frames, require a name-like value AND nearby
    # identity evidence; the cue by itself cannot turn the next clause into PII.
    _NAME_ENDING = re.compile(r"(?:రావు|రెడ్డి|రాజు|కుమార్|కుమారి|లక్ష్మి|దేవ్|ప్రసాద్|నాథ్|మ్మ|య్య)$")
    _IDENTITY_AFTER = re.compile(
        r"^\s*(?:[,.!?]\s*)?(?:(?:వయసు|వయస్సు|ఏజ్|age)\s*[:=]?\s*)?"
        r"(?:[1-9][0-9]?\s*(?:ఏళ్లు|ఏళ్ళు|సంవత్సరాలు|ఇయర్స్|years?|yrs?)(?![\u0c00-\u0c7fA-Za-z])|"
        r"(?:అని\s+(?:పిలుస్తారు|అంటారు))|(?:గారు|గారా|గారూ)(?![\u0c00-\u0c7f]))",
        re.I,
    )
    _VALUE_LABEL = re.compile(r"^(?:నంబర్|నెంబర్|నంబరు|number|name)\s+", re.I)

    def __init__(self):
        super().__init__(supported_entities=["PERSON"], name="TeluguNameCueRecognizer", supported_language="en")

    def load(self) -> None:
        pass

    def _value(self, text: str, start: int, cue: str):
        words = []
        pos = start
        for _ in range(4):
            if words:
                gap = re.match(r"[ \t]+", text[pos:])
                if not gap:
                    break
                pos += gap.end()
            token = self._WORD.match(text, pos)
            if not token:
                break
            word = token.group()
            folded = word.casefold()
            # The cue identifies a frame, not its value. Stop before a verb,
            # duration or symptom so a whole clause cannot become PERSON.
            if implausible_person(word):
                break
            role_root = re.sub(r"(?:గా|ని|ను)$", "", word)
            if (folded in self._PRONOUNS or folded in self._ROMAN_STOP
                    or self._PREDICATE.fullmatch(word) or folded in _NAME_BOUNDARIES
                    or folded in _CLINICAL_NON_NAMES or role_root in self._ROLES):
                break
            residence_tail = re.match(r"[\u200c\u200d]?(?:లో|నుంచి|నుండి|కు|కి)", text[token.end():])
            if cue == "patient_introduction" and (
                word in self._PLACES or
                (word.endswith(("లో", "నుంచి", "నుండి")) and
                 re.match(r"\s*(?:ఉంటాను|ఉంటున్నాను|వచ్చాను|వస్తున్నాను)", text[token.end():])) or
                (residence_tail and re.match(r"\s*(?:ఉంటాను|ఉంటున్నాను|వచ్చాను|వస్తున్నాను)", text[token.end() + residence_tail.end():]))
            ):
                break
            if word.isascii() and word.islower() and cue == "person_honorific" and len(word) < 3:
                break
            token_end = token.end()
            if text[token_end:token_end + 1] == "." and (
                re.match(rf"[ \t]+{self._WORD.pattern}", text[token_end + 1:])
                or (words and len(word) <= 3)
            ):
                token_end += 1
            words.append((pos, token_end))
            pos = token_end
            if cue == "patient_introduction" and re.match(r"\s+(?:అంటారు|అని|మాట్లాడుతున్నాను|ni|nu)\b", text[pos:], re.I):
                break
        if not words:
            return None
        end = words[-1][1]
        surface = text[start:end]
        if implausible_person(surface) or not plausible_person_surface(surface, explicit=True):
            return None
        # Strip an attached grammatical suffix only when its root is a whole
        # Unicode syllable: suffixes here begin with a new Telugu consonant.
        suffix_separated = False
        if cue in {"patient_introduction", "identity_relation", "person_honorific"} and "\u0c00" <= text[end - 1] <= "\u0c7f":
            last_start = words[-1][0]
            last = text[last_start:end]
            stem, suffix = split_telugu_suffix(last, kind="name")
            if suffix:
                end = last_start + len(stem)
                suffix_separated = True
        return end, suffix_separated

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and "PERSON" not in entities:
            return []
        results = []
        for pattern, cue in self._CUES:
            for match in pattern.finditer(text):
                start = match.end()
                if start >= len(text) or _OBJECT_NAME_FRAME.search(text[max(0, match.start()-35):start]):
                    continue
                value = self._value(text, start, cue)
                if value is None:
                    continue
                end, suffix_separated = value
                if cue == "person_honorific":
                    # ASR often inserts a field label between a title and the
                    # actual name (for example, "doctor number <name>").
                    label = self._VALUE_LABEL.match(text[start:end])
                    if label:
                        start += label.end()
                if end <= start or is_suppressed("PERSON", text[start:end]) and cue != "explicit_name":
                    continue
                if cue != "explicit_name":
                    surface = text[start:end]
                    words = surface.split()
                    name_shape = (1 <= len(words) <= 2 and all(len(word) >= 2 for word in words)
                                  and not implausible_person(surface))
                    identity_after = bool(self._IDENTITY_AFTER.match(text[end:end + 45]))
                    name_ending = bool(self._NAME_ENDING.search(words[-1])) if words else False
                    if cue == "patient_introduction" and not (name_shape and (identity_after or name_ending)):
                        continue
                    if cue == "person_honorific" and not (name_shape and (identity_after or name_ending)):
                        continue
                    if cue == "identity_relation" and not (name_shape and (identity_after or name_ending)):
                        continue
                # A named city in an introduction/origin clause is still a
                # place; a weak first-person cue must not relabel it PERSON.
                if cue != "explicit_name" and text[start:end].casefold() in _load_place_gazetteer()[0]:
                    continue
                results.append(RecognizerResult(
                    "PERSON", start, end, .96,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "boundary_evidence": cue, "suffix_separated": suffix_separated},
                ))
        return list({(r.start, r.end): r for r in results}.values())


# Tokens for gazetteer lookup. ``\w`` is wrong here: Python classes the Indic
# vowel signs as combining marks rather than alphanumerics, so ``\w`` splits
# ``ಶೋಭನ`` at its first mark and nothing ever matches. The explicit block ranges
# in ``_INDIC_LETTER`` cover the marks because they span the whole block.
# Unlike ``_NAME_WORD`` this excludes "." — that class allows it for initials
# ("Dr. R. Kumar"), but here it swallows the sentence-final stop and turns
# "Shobhana." into a token that matches nothing.
_TOKEN = re.compile(rf"[{_INDIC_LETTER}A-Za-z][{_INDIC_LETTER}A-Za-z']*", re.UNICODE)

# Case suffixes are matched only on listed Tamil roots in an identifying frame.
# Keep original offsets and redact the root, leaving the grammatical suffix.
_TAMIL_CASE_SUFFIXES = ("வுக்கு", "க்காக", "வுடன்", "விடம்", "வின்", "வால்", "க்கு", "வை")
_TAMIL_NAME_FRAME = re.compile(r"(?:நம்ம|திரு|திருமதி|செல்வி|டாக்டர்|பெயர்)\s*$")


@lru_cache(maxsize=1)
def load_person_name_gazetteer(path: Optional[str] = None) -> frozenset:
    """Load and normalise the personal-name list, once per process."""
    target = Path(path) if path else require_data_path("data/gazetteers/indic_person_names.txt")
    if not target.is_file():
        return frozenset()
    names = set()
    for line in target.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            names.add(unicodedata.normalize("NFKC", entry).casefold())
    return frozenset(names)


class IndicPersonNameGazetteer(EntityRecognizer):
    """Match personal names from a closed list, in any script.

    Neural detectors can miss Indic-script names, including names that double
    as ordinary words or that sit at a code-switch
    boundary. ``ಶೋಭನ`` is the worked example: IndicNER finds it in a plain
    Kannada sentence and misses it in the Kannada-into-Telugu switch where it
    actually occurs, so it survived every run until this recognizer existed.

    Matching is on whole tokens, longest-first, so a listed given name followed
    by a listed surname ("శారదా శర్మ") reports as one span rather than two.
    Scored above spaCy's flat 0.85 so that Presidio's overlap resolution keeps
    this precise span instead of an English model's longer guess around it.
    """

    def __init__(self, gazetteer_path: Optional[str] = None, score: float = 0.9):
        self.names = load_person_name_gazetteer(gazetteer_path)
        self.score = score
        super().__init__(
            supported_entities=["PERSON"],
            name="IndicPersonNameGazetteer",
            supported_language="en",
        )

    def load(self) -> None:  # pragma: no cover - nothing to load lazily
        pass

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        if not self.names or "PERSON" not in (entities or ["PERSON"]):
            return []

        tokens = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text)]
        results: List[RecognizerResult] = []
        i = 0
        while i < len(tokens):
            start, end, surface = tokens[i]
            # Try "given surname" first so the pair reports as a single span.
            if i + 1 < len(tokens):
                nxt_start, nxt_end, nxt_surface = tokens[i + 1]
                pair_gap = text[end:nxt_start]
                if pair_gap.strip() == "" and self._listed(surface) and self._listed(nxt_surface):
                    results.append(self._result(start, nxt_end))
                    i += 2
                    continue
            if self._listed(surface):
                results.append(self._result(start, end))
            elif _TAMIL_NAME_FRAME.search(text[max(0, start - 35):start]):
                for suffix in _TAMIL_CASE_SUFFIXES:
                    if surface.endswith(suffix):
                        root = surface[:-len(suffix)]
                        if len(root) >= 3 and all("\u0b80" <= ch <= "\u0bff" for ch in root) and self._listed(root):
                            result = self._result(start, start + len(root))
                            result.recognition_metadata["boundary_evidence"] = "listed_tamil_case_root"
                            results.append(result)
                            break
            i += 1
        return results

    def _listed(self, token: str) -> bool:
        return unicodedata.normalize("NFKC", token).casefold() in self.names

    def _result(self, start: int, end: int) -> RecognizerResult:
        return RecognizerResult(
            entity_type="PERSON", start=start, end=end, score=self.score,
            recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id}
        )


# Age, short dates and spoken addresses were all at 0% recall against the real
# transcript benchmark: none of the three has a model that emits them. GLiNER's
# 42-label taxonomy has no AGE, IndicNER has only PER/LOC/ORG, and Presidio's
# DateRecognizer does not fire on a bare "12/5". All three are stated in a
# fixed spoken frame in these consultations, so a keyed pattern recovers them.

_AGE_KEYWORD = "|".join([
    r"\bage\s*(?:is|:)?", r"\baged\b", r"ఏజ్\s*(?:ఇస్|is)?", r"ఐటిస్",
    r"వయస(?:్స)?ు", "ವಯಸ್ಸು", "வயது", "বয়স", "വയസ്സ്", "उम्र", "आयु",
])
_AGE_UNIT = "|".join([
    r"(?:years?|yrs?)\s*old\b", r"(?:months?|weeks?|days?)[ \t]+old\b",
    r"ఇయర్స్\s*ఓల్డ్", r"ఏళ్ళు", r"సంవత్సరాలు",
    r"ಇಯರ್ಸ್\s*ಓಲ್ಡ್", r"ಇಯರ್ಸ್\s*ಓಲ್ಡು",
    r"साल", r"वर्ष", r"ವರ್ಷ", r"வயது", r"বছর",
])
_BARE_AGE_YEARS = re.compile(
    r"(?<!\w)(\d{1,3})\s*(?:years?|yrs?|ఇయర్స్|ఇయర్|ಇಯರ್ಸ್)(?!\w)", re.I,
)
_AGE_SELF_FRAME = re.compile(
    r"(?:\bi\s+am|\bi['’]m|\bpatient\s*[:,-]?|నాకు|మాకు|నేను|ఐ\s*యామ్|நான்\s+ஒரு|"
    r"ನಾನು|ಐ\s*ಆಮ್|मैं|मुझे)\s*$", re.I,
)
_AGE_FOLLOWING = re.compile(
    r"^\s*(?:$|[.,;!?]|old\b|female\b|male\b|woman\b|man\b|"
    r"ఓల్డ్|ఫీమేల్|మేల్|సార్|ஆ\s+இருக்கும்போது|ಓಲ್ಡ್|ಹೆಣ್ಣು|ಗಂಡು|and\b|అండ్|ಅಂಡ್)", re.I,
)
_AGE_SEX = re.compile(r"^\s*(?:female\b|male\b|woman\b|man\b|ఫీమేల్|మేల్|ಹೆಣ್ಣು|ಗಂಡು)", re.I)
_AGE_QUESTION = re.compile(
    r"(?:how\s+old\s+are\s+you|what(?:'s|\s+is)\s+your\s+age|మీ\s*(?:వయసు|ఏజ్)\s*(?:ఎంత|ఇప్పుడు)?|"
    r"ఎంత\s*(?:వయసు|వయస్సు)\s*(?:మీకు|నీకు|అంతా)?|(?:వయసు|వయస్సు|ఏజ్)\s*ఎంత\s*(?:మీకు|నీకు)?|"
    r"आपकी\s+(?:उम्र|age)\s*(?:कितनी\s+है)?\s*(?:अभी)?|உங்கள்\s+வயது\s*என்ன|வயசு|ನಿಮ್ಮ\s+ವಯಸ್ಸು\s*ಎಷ್ಟು)"
    r"\s*[?.:,—-]+\s*(?:ఎంత\s*[?.:,]+\s*|I\s+am\s+)?(\d{1,3})(?!\d)",re.I)
_HISTORICAL_AGE = re.compile(
    r"(?:\bage\s*(?:of\s+menarche\s*)?(?:at|of|is|:)?|"
    r"\bmenarche\s*(?:is|at|age|:)?|"
    r"ఏజ్\s*(?:అట్|ఆఫ్|ఇస్))\s*[.:]?\s*(\d{1,3})(?!\d)", re.I,
)
_AGE_AFTER_UNIT = re.compile(r"^[\s.,;:!?]*(?:age\b|ఏజ్|వయసు|ವಯಸ್ಸು|उम्र)", re.I)
_AGE_WITH_SUFFIX = re.compile(
    r"(?<!\w)(\d{1,3})\s*(?:సంవత్సరాల\s+వయసు(?:లో)?|ఇయర్స్\s+ఏజ్|"
    r"సంవత్సరాలు\s+వచ్చాయి|years?\s+of\s+age|ఏజ్)(?!\w)", re.I)
_AGE_ANSWER_WITH_NAME = re.compile(
    rf"(?:పేషెంట్\s+ఏజ్\s*ఎంత)[^.!?\n]{{0,15}}[?]\s*{_NAME_WORD}\s+(\d{{1,3}})\s*(?:years?|ఇయర్స్)(?!\w)", re.I)
_AGE_REPEAT_GAP = re.compile(
    r"^(?:[\s.,;:!?—-]|years?|yrs?|ఇయర్స్|ఇయర్|ಇಯರ್ಸ್|age|ఏజ్|"
    r"వయసు|ఆ|అవును|కాదు|దాకా|వచ్చింది|సార్|sir|yes|no|okay|ok|and|అండ్)*$", re.I,
)


# Telugu age expressions are handled separately from the older multilingual
# number-only rules.  Keep a stated unit in the span, and require a personal
# frame for month counts, which otherwise commonly describe a duration.
_TELUGU_AGE_NUMBER = TELUGU_NUMBER_PATTERN
_TELUGU_AGE_UNIT = r"(?:ఏళ్లు|ఏళ్ళు|ఏళ్ల|ఏళ్ళ|సంవత్సరాలు|సంవత్సరాల|నెలలు|నెలల|yellu|ellu|years?|yrs?|samvatsaralu|nelalu|nelala)"
_TELUGU_AGE_VALUE = re.compile(
    rf"(?<![\w\u0c00-\u0c7f])(?P<number>{_TELUGU_AGE_NUMBER})(?:[ \t]+(?P<unit>{_TELUGU_AGE_UNIT}))?(?![\w\u0c00-\u0c7f])",
    re.I,
)
_TELUGU_AGE_CUE = re.compile(
    r"(?:వయసు|వయస్సు|(?:నా|మా|మీ|నీ)\s+వయసు|నాకు|మాకు|నీకు|మీకు|"
    r"(?:మా|నా)\s+(?:అబ్బాయి|అమ్మాయి|పాప|బాబు|బిడ్డ|భర్త|తండ్రి|అమ్మ)(?:కి|కు)?|"
    r"\b(?:naa|na|ma|maa)\s+vayasu|\b(?:naaku|naku|maaku)\b)\s*[:=]?\s*$",
    re.I,
)
_TELUGU_EXPLICIT_AGE_CUE = re.compile(
    r"(?:వయసు|వయస్సు|ఏజ్|(?:మా|నా)\s+(?:అబ్బాయి|అమ్మాయి|పాప|బాబు|బిడ్డ)(?:కి|కు)?)\s*[:=]?\s*$",
    re.I,
)
_TELUGU_AGE_PERSON_AFTER = re.compile(
    r"^[ \t]*(?:పురుషుడు|మహిళ|వ్యక్తి|వాడిని|దాన్ని|బాబు|పాప|బిడ్డ|అబ్బాయి|అమ్మాయి|male\b|female\b)",
    re.I,
)
_TELUGU_AGE_MEASUREMENT = re.compile(
    r"^[ \t]*(?:నిమిషాలు|నిమిషాల|రోజులు|రోజుల|మి\.?గ్రా|mg\b|ml\b|kg\b|"
    r"డోసు|మాత్రలు|(?:మి\.?లీ))", re.I,
)
_TELUGU_DURATION_AFTER = re.compile(r"^[ \t]*(?:నుంచి|నుండి|పాటు|గా|since\b|for\b)", re.I)


class AgeRecognizer(KeyedGroupRecognizer):
    """A stated age: "my age is 27", "ఐ యామ్ 35 ఇయర్స్ ఓల్డ్", "29 years old"."""

    def __init__(self):
        super().__init__(
            supported_entity="AGE",
            # Either the keyword precedes the number, or a unit follows it.
            regex=rf"(?:(?:{_AGE_KEYWORD})\W{{0,8}}?(\d{{1,3}})\b)"
                  rf"|(?:\b(?<!\d)(\d{{1,3}})\s*(?:{_AGE_UNIT}))",
            score=0.8,
            name="AgeRecognizer",
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and "AGE" not in entities:
            return []
        results = []
        for start, end, evidence in iter_hindi_age_spans(text):
            results.append(RecognizerResult(
                entity_type="AGE", start=start, end=end, score=0.96,
                recognition_metadata={"recognizer_name": self.name,
                                      "recognizer_identifier": self.id,
                                      "source": "rule",
                                      "boundary_evidence": evidence},
            ))
        for match in _TELUGU_AGE_VALUE.finditer(text):
            # English units are shared spellings; apply the Telugu-specific
            # whole-span rule only where Telugu script or a Telugu age cue is present.
            if not re.search(r"[\u0c00-\u0c7f]", text) and not _TELUGU_AGE_CUE.search(text[max(0, match.start() - 70):match.start()]):
                continue
            number = parse_telugu_number(match.group("number"))
            if number is None or not 0 < number <= 120:
                continue
            start, end = match.span()
            before = text[max(0, start - 70):start]
            after = text[end:end + 50]
            cue = bool(_TELUGU_AGE_CUE.search(before))
            explicit_cue = bool(_TELUGU_EXPLICIT_AGE_CUE.search(before))
            person_after = bool(_TELUGU_AGE_PERSON_AFTER.match(after))
            unit = match.group("unit")
            if _TELUGU_AGE_MEASUREMENT.match(after):
                continue
            if unit and _TELUGU_DURATION_AFTER.match(after):
                continue
            if unit:
                is_month = unit.casefold() in {"నెలలు", "నెలల", "nelalu", "nelala"}
                # "to me, two months" usually describes symptom duration;
                # only an age field or a child frame supports infant age.
                if is_month and not (explicit_cue or person_after):
                    continue
                if not is_month and not (cue or person_after):
                    continue
            elif not explicit_cue:
                continue
            if age_is_duration(text, start, end):
                continue
            # The numeric value is the identifier; leaving the unit visible
            # also agrees with the existing AGE span contract.
            if match.group("number").isascii() and match.group("number").isdigit():
                end = match.end("number")
            results.append(RecognizerResult(
                "AGE", start, end, .95,
                recognition_metadata={"recognizer_name": self.name,
                                      "recognizer_identifier": self.id,
                                      "boundary_evidence": "explicit_personal_age"},
            ))
        for match in self.regex.finditer(text):
            # Two alternatives, so take whichever group participated.
            for group in (1, 2):
                span = match.span(group)
                if span[0] < 0:
                    continue
                value = match.group(group)
                if not value or not (0 < int(value) <= 120) or age_is_duration(text,*span):
                    continue
                results.append(RecognizerResult(
                    entity_type="AGE", start=span[0], end=span[1], score=self.score,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id}
                ))
                break
        for match in _BARE_AGE_YEARS.finditer(text):
            start, end = match.span(1)
            after = text[match.end():match.end() + 45]
            before = text[max(0, start - 40):start]
            if not (0 < int(match.group(1)) <= 120) or age_is_duration(text,start,end):
                continue
            if (_AGE_SEX.match(after) or _AGE_AFTER_UNIT.match(after)
                    or has_historical_age_context(text, start)
                    or (_AGE_SELF_FRAME.search(before) and _AGE_FOLLOWING.match(after))):
                results.append(RecognizerResult(
                    entity_type="AGE", start=start, end=end, score=self.score,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "boundary_evidence": "patient_age_frame"},
                ))
        for pattern in (_AGE_WITH_SUFFIX, _AGE_ANSWER_WITH_NAME):
            for match in pattern.finditer(text):
                start, end = match.span(1)
                if 0 < int(match[1]) <= 120 and not re.match(r"\s*(?:mg|kg|ml)\b", text[end:], re.I):
                    results.append(RecognizerResult("AGE", start, end, self.score,
                        recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                              "boundary_evidence": "explicit_age_suffix"}))
        for match in _AGE_QUESTION.finditer(text):
            start,end = match.span(1)
            if 0<int(match.group(1))<=120 and not age_is_duration(text, start, end):
                results.append(RecognizerResult("AGE",start,end,self.score,
                    recognition_metadata={"recognizer_name":self.name,"recognizer_identifier":self.id,
                                          "boundary_evidence":"patient_age_answer"}))
                # A directly adjacent corrected answer may state a different
                # age. Do not cross words or propagate across clinical topics.
                cursor = end
                for _ in range(2):
                    correction = re.match(r"[\s.,?]+(\d{1,3})(?!\d)", text[cursor:end+25])
                    if not correction:
                        break
                    a, b = cursor+correction.start(1), cursor+correction.end(1)
                    if not (0 < int(correction[1]) <= 120) or age_is_duration(text, a, b):
                        break
                    results.append(RecognizerResult("AGE", a, b, self.score,
                        recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                              "boundary_evidence": "corrected_age_answer"}))
                    cursor = b
        for match in _HISTORICAL_AGE.finditer(text):
            start, end = match.span(1)
            if 0 < int(match.group(1)) <= 120 and not age_is_duration(text, start, end):
                results.append(RecognizerResult("AGE", start, end, self.score,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "boundary_evidence": "historical_age"}))
        # Repeat only the same age in the immediately adjacent answer. Never
        # propagate a number throughout an entire transcript or across doses.
        for seed in list(results):
            for match in re.finditer(r"(?<!\d)\d{1,3}(?!\d)",
                                     text[max(0, seed.start-70):seed.end+70]):
                start = max(0, seed.start-70) + match.start()
                end = max(0, seed.start-70) + match.end()
                if match.group() != text[seed.start:seed.end] or (start, end) == (seed.start, seed.end):
                    continue
                gap = text[end:seed.start] if end <= seed.start else text[seed.end:start]
                repeat_gap = gap
                if has_historical_age_context(text, seed.start):
                    repeat_gap = re.sub(r"ல\s+இருந்து\s+இருக்கு", "", repeat_gap)
                if len(gap) <= 65 and _AGE_REPEAT_GAP.fullmatch(repeat_gap) and not age_is_duration(text, start, end):
                    results.append(RecognizerResult("AGE", start, end, self.score,
                        recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                              "boundary_evidence": "repeated_age_answer"}))
        full_expressions = [
            result for result in results
            if (result.recognition_metadata or {}).get("boundary_evidence") == "explicit_personal_age"
            and bool(re.search(r"\s", text[result.start:result.end]))
        ]
        best = {}
        for result in results:
            if any(full.start <= result.start and result.end <= full.end
                   and (full.start, full.end) != (result.start, result.end)
                   for full in full_expressions):
                continue
            key = (result.start, result.end)
            if key not in best or result.score > best[key].score:
                best[key] = result
        return sorted(best.values(), key=lambda r: (r.start, r.end))


class ShortDateRecognizer(PatternRecognizer):
    """A bare ``12/5`` or ``05/20`` — a visit or last-period date.

    Constrained so clinical readings never match: a blood pressure such as
    ``118/76`` is excluded both by the digit-boundary lookarounds and by the
    day/month range check below, which previously sailed through as a date.
    """

    PATTERNS = [
        Pattern(name="short_date", regex=r"(?<![\d/])\d{1,2}/\d{1,2}(?:/\d{2,4})?(?![\d/])", score=0.55),
    ]

    def __init__(self):
        super().__init__(
            # Not DATE_TIME: that label sits in DEFAULT_LABELS_TO_IGNORE and is
            # discarded during taxonomy routing. sensitive_date is the
            # equivalent inside the 42-label contract.
            supported_entity="sensitive_date",
            patterns=self.PATTERNS,
            context=["date", "day", "period", "visit", "రోజు", "తేదీ", "तारीख", "ದಿನಾಂಕ"],
            supported_language="en",
        )
        self.supported_entities.append('date_of_birth')

    def validate_result(self, pattern_text: str):
        return valid_calendar_literal(pattern_text)

    def analyze(self,text,entities,nlp_artifacts=None,**kwargs):
        results = super().analyze(text,entities,nlp_artifacts,**kwargs)
        # Explicit invalid-result filtering also covers analyzer configurations
        # that include zero-score invalid pattern matches.
        accepted=[]
        for r in results:
            if not valid_calendar_literal(text[r.start:r.end]):
                continue
            if re.search(r"(?:\bDOB|date\s*of\s*birth|birth\s*date|जन्म\s*तिथि|పుట్టిన\s*తేదీ)\s*[:=]?\s*$",text[max(0,r.start-65):r.start],re.I):
                r.entity_type='date_of_birth'
                r.recognition_metadata={**(r.recognition_metadata or {}),'recognizer_name':self.name,
                                        'boundary_evidence':'explicit_birth_date'}
            if not entities or r.entity_type in entities:
                accepted.append(r)
        return accepted


@lru_cache(maxsize=1)
def _load_place_gazetteer():
    """Verify and compile the pinned place list once per process."""
    place_file=require_data_path("data/gazetteers/indic_places.txt")
    places={line.split('#',1)[0].strip().casefold() for line in place_file.read_text(encoding='utf-8').splitlines()} if place_file.is_file() else set()
    places.discard("")
    # Known places corroborate travel and locative suffixes. Do not treat
    # every noun after "I" as a place, or redact generic place mentions.
    if places:
        known = "|".join(re.escape(place) for place in sorted(places, key=len, reverse=True))
        # Indic vowel signs are not Python \w; use whole script blocks to
        # avoid matching a city inside a longer unrelated word.
        suffix = r"(?:లోనే|లో|కి|కు|నుంచి|થી|માં|ನಲ್ಲಿ|க்கு|இல்)"
        letter = rf"[{_INDIC_LETTER}A-Za-z]"
        return frozenset(places), re.compile(rf"(?<!{letter})({known})(?=$|\s|[.,!?;।]|[\u200c\u200d]?{suffix}(?!{letter}))", re.I)
    return frozenset(), None


class SpokenPlaceRecognizer(KeyedGroupRecognizer):
    """A place named in the spoken frame "I'm from X" / "my address is X".

    Every LOCATION in the benchmark arrives this way — ``ఐ యామ్ ఫ్రమ్ కూకట్పల్లి``,
    ``I'm from Gachpoli``, ``my address is Kukatpalli`` — and none of the models
    finds them, because the place names are neighbourhoods absent from any
    gazetteer these models were trained on.
    """

    def __init__(self):
        # These frames require a personal residence/origin cue or an explicit
        # address field.  A bare locative ending is not evidence of an address.
        word = r"[\u0C00-\u0C7FA-Za-z][\u0C00-\u0C7FA-Za-z\u200c\u200d]*"
        value = rf"{word}(?:[ \t]+{word}){{0,2}}?"
        field_value = rf"{word}(?:[ \t]+{word}){{0,2}}"
        self.telugu_residence_cue = re.compile(
            rf"(?:నేను|మేము|రోగి|పేషెంట్|మా\s+కుటుంబం)\s+(?P<place>{value})"
            rf"[ \t]*(?P<suffix>[\u200c\u200d]?(?:లోని|లో|కు|కి))"
            rf"[ \t]+(?:ఉంటాను|ఉంటున్నాను|ఉన్నాను|ఉంటాం|ఉంటున్నాం|ఉంటున్నారు|నివసిస్తాను|నివసిస్తున్నాను|నివసిస్తున్నారు)(?![\u0C00-\u0C7F])",
            re.I,
        )
        self.telugu_origin_cue = re.compile(
            rf"(?:నేను|మేము)\s+(?P<place>{value})"
            rf"[ \t]*(?P<suffix>[\u200c\u200d]?(?:నుండి|నుంచి))"
            rf"[ \t]+(?:వచ్చాను|వచ్చాము|వస్తున్నాను|వస్తున్నాం)(?![\u0C00-\u0C7F])",
            re.I,
        )
        self.telugu_destination = re.compile(
            rf"(?:నేను|మేము)\s+(?P<place>{value})"
            rf"[ \t]*(?P<suffix>[\u200c\u200d]?(?:కు|కి))"
            rf"[ \t]+(?:వెళ్లాను|వెళ్ళాను|వెళ్లాం|వెళ్ళాం|వచ్చాను|వచ్చాము)(?![\u0C00-\u0C7F])",
            re.I,
        )
        self.telugu_belonging = re.compile(
            rf"(?:మాది|మా\s+ఊరు|నా\s+ఊరు)\s*[:=]?[ \t]+(?P<place>{field_value})"
            rf"(?=$|[\s,.;!?।॥])", re.I,
        )
        self.telugu_field = re.compile(
            rf"(?:గ్రామం|ఊరు|మండలం|జిల్లా|నగరం|పట్టణం|కాలనీ)"
            rf"[ \t]*[:=]?[ \t]+(?P<place>{field_value})(?=$|[\s,.;!?।॥])", re.I,
        )
        self.telugu_relation = re.compile(
            rf"(?P<place>{value})[ \t]+(?:దగ్గర|ఎదురుగా)(?=$|[\s,.;!?।॥])", re.I,
        )
        self.roman_residence = re.compile(
            r"\b(?:nenu|memu)\s+(?P<place>[A-Za-z][A-Za-z'\-]*(?:[ \t]+[A-Za-z][A-Za-z'\-]*){0,2}?)"
            r"[ \t]+(?:lo|loni|nunchi|nundi)[ \t]+(?:untanu|untunnanu|unnanu|vachanu|vacchanu|vastunnanu)\b",
            re.I,
        )
        self.roman_belonging = re.compile(
            r"\b(?:maadi|madi|maa\s+ooru)\s*[:=]?[ \t]+"
            r"(?P<place>[A-Za-z][A-Za-z'\-]*(?:[ \t]+[A-Za-z][A-Za-z'\-]*){0,2}?)"
            r"(?=$|[\s,.;!?])", re.I,
        )
        self.telugu_address = re.compile(
            r"(?:నా\s+)?(?:చిరునామా|అడ్రస్|విలాసం)[ \t]*[:=]?[ \t]+"
            r"(?P<place>[^.!?;\n।॥]{3,110})", re.I,
        )
        self.telugu_house = re.compile(
            r"(?:ఇంటి|ఇల్లు|హౌస్)[ \t]*(?:నెం\.?|నంబర్|నెంబర్|సంఖ్య)[ \t]*[:=]?[ \t]*"
            r"(?P<place>[0-9\u0C66-\u0C6F]+(?:[-/][0-9\u0C66-\u0C6F]+){0,3}"
            r"(?:[ \t]*,[ \t]*(?:(?:రోడ్|వీధి|కాలనీ)[ \t]*(?:నెం\.?|నంబర్|నెంబర్)?[ \t]*"
            r"[0-9\u0C66-\u0C6F\u0C00-\u0C7FA-Za-z-]+|[\u0C00-\u0C7FA-Za-z][\u0C00-\u0C7FA-Za-z \u200c\u200d-]*)){0,3})",
            re.I,
        )
        self.places, self.place_pattern = _load_place_gazetteer()
        self.address_field = re.compile(
            r"(?:\baddress\s*(?:is|:)?|पता|முகவரி|చిరునామా|విలాసం|ವಿಳಾಸ|ঠিকানা|വിലാസം)\s*:?\s*"
            r"([^.!?;\n।॥:]{1,100}?\b[0-9]{6}\b)", re.IGNORECASE,
        )
        keyword = "|".join([
            r"\bi\s*am\s*from", r"\bi['’]?m\s*from", r"ఐ\s*యామ్\s*ఫ్రమ్",
            r"\bi\s+live\s+(?:in|at)", r"నేను\s*(?:ఉండేది|ఉంటున్నది)",
            r"address\s*(?:is|at|:)?", r"అడ్రస్", r"చిరునామా", r"पता", r"ವಿಳಾಸ",
            r"\bi\s*(?:am|'m)\s*(?:staying|living)\s*(?:in|at)",
        ])
        self.question_field = re.compile(
            rf"(?:from\s+where\s+are\s+you|where\s+are\s+you\s+from|where\s+do\s+you\s+live|"
            rf"ఏ\s*ఊరు\s*మీది|మీది\s*ఏ\s*ఊరు|మీ\s*ఊరు\s*ఎక్కడ|"
            rf"మీరు[ \t]+ఎక్కడ[ \t]+(?:నివసిస్తున్నారు|ఉంటున్నారు|ఉంటారు)|"
            rf"आप\s*कहाँ\s*रहते\s*हैं|તમારું\s*ગામ\s*કયું\s*છે|"
            rf"உங்கள்\s*ஊர்\s*எது|ನಿಮ್ಮ\s*ಊರು\s*ಯಾವುದು)"
            rf"\s*[?.:,—-]+\s*(?:from\s*[?.:,—-]+\s*)?({_NAME_WORD}(?:[ \t]+{_NAME_WORD})?)",re.I)
        self.asr_address = re.compile(rf"\bmy\s+interest\s*(?:is|:)\s*({_NAME_WORD})",re.I)
        # First-person residence + a locative suffix supplies identity context
        # without requiring the place to occur in a closed gazetteer. Capture
        # the original place root and leave the grammatical suffix readable.
        self.telugu_residence = re.compile(
            rf"(?<![{_INDIC_LETTER}A-Za-z])(?:నేను|మేము)[ \t]+"
            rf"({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,2}}?)[\u200c\u200d]?[ \t]*లో[ \t]+"
            rf"(?:నివసి(?:స్తున్నాను|స్తున్నాము|స్తున్నాం|స్తాను|స్తాము)|"
            rf"ఉంటున్నా(?:ను|ము|ం)|ఉంటాను|ఉంటాము)(?![{_INDIC_LETTER}A-Za-z])"
        )
        self.telugu_origin = re.compile(
            rf"(?<![{_INDIC_LETTER}A-Za-z])(?:నేను|మేము)[ \t]+"
            rf"({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,2}}?)[\u200c\u200d]?[ \t]*(?:నుంచి|నుండి)"
            rf"(?:[ \t]+(?:వచ్చా(?:ను|ము|ం)|వస్తున్నా(?:ను|ము|ం))(?=$|[\s.,!?;:।॥])"
            r"|(?=[ \t]*(?:[.!?;:।॥]|$)))"
        )
        self.hindi_residence = re.compile(
            rf"(?<![{_INDIC_LETTER}A-Za-z])(?:मैं|हम)[ \t]+"
            rf"({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,3}}?)[ \t]+"
            rf"(?:से[ \t]+(?:हूँ|हूं|हैं)|में[ \t]+(?:रहता|रहती|रहते)[ \t]+(?:हूँ|हूं|हैं))"
            r"(?=$|[\s.,!?;:।॥])"
        )
        super().__init__(
            # Not LOCATION, for the same reason as the date above: it is ignored
            # during routing. "address" carries the same meaning in the taxonomy.
            supported_entity="address",
            # Deliberately not _NAME_WORD: that permits "." for initials and
            # would swallow the sentence-final stop into the place name.
            regex=rf"(?:{keyword})\s+(?:the\s+|ది\s+)?({_NAME_WORD})",
            score=0.6,
            name="SpokenPlaceRecognizer",
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = super().analyze(text, entities, nlp_artifacts)
        if entities and "address" not in entities:
            return []
        results = [r for r in results if text[r.start:r.end] not in {"चलेगा","ఇక్కడ","చిన్న"}]
        results.extend(self._telugu_cue_results(text))
        for match in self.hindi_residence.finditer(text):
            start, end = match.span(1)
            surface = text[start:end]
            if is_suppressed("LOCATION", surface) or surface in {
                "घर", "यहाँ", "वहाँ", "यहां", "वहां", "गाँव", "गांव", "शहर", "अस्पताल",
                "ऑफिस", "कार्यालय", "स्कूल", "कहीं",
            }:
                continue
            results.append(RecognizerResult("address", start, end, .9,
                recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                      "boundary_evidence": "explicit_residence"}))
        telugu_locations = list(self.telugu_residence.finditer(text)) + list(self.telugu_origin.finditer(text))
        for match in telugu_locations:
            start, end = match.span(1)
            surface = text[start:end]
            # Generic home/room/hospital/town references do not identify a
            # named residence. Multiword named sites remain candidates.
            if is_suppressed("LOCATION", surface) or surface in {
                "ఇంట్", "ఇంటి", "ఇల్లు", "గది", "ఆసుపత్రి", "గ్రామం", "నగరం", "పట్టణం", "ఒత్తిడి",
                "చికిత్స", "పని", "బయట", "ఇక్కడ", "అక్కడ",
            }:
                continue
            results.append(RecognizerResult("address", start, end, .9,
                recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                      "boundary_evidence": "explicit_residence"}))
        # Gazetteer contents are pinned; workers reload them on restart.
        if self.place_pattern is not None:
            for match in self.place_pattern.finditer(text):
                start, end = match.span(1)
                before, after = text[max(0, start-60):start], text[end:end+60]
                personal = bool(
                    re.search(r"(?:నాది|మాది|నా\s+ఊరు|మా\s+ఊరు)[\s,]*$", before)
                    or
                    re.search(r"(?:నేను|మేము|మాది|వాళ్లది|వాళ్ళది)[\s,]*$", before)
                    and re.match(r"(?:లో|కి|కు)(?![A-Za-z])", after)
                    or re.search(r"(?:मैं|हम)[^.!?।\n]{0,35}$", before) and
                    re.match(r"\s+(?:गया|गई|गए|पहुंचा|रहता|रहती|रहते|ट्रैवल\s+किया)(?!\w)", after)
                    or re.search(r"(?:હું|અમે)[\s,]*(?:અહ[\s,]*)?$", before) and
                    re.match(r"(?:થી\s*(?:આવ્યો|આવી|આવ્યા)|\s+(?:જઈને|ગયો|ગઈ|રહું|રહીએ))", after)
                    or re.match(r"\s+(?:నుంచి|నుండి)\s+(?:వస్తున్నాను|వచ్చాను|వచ్చాము|వస్తున్నాం)(?!\w)", after)
                    or re.search(r"\bi\s+(?:went|travelled|traveled|live|stayed|came)\s+(?:to|in|at|from)\s*$", before, re.I)
                )
                repeated = any(
                    r.end <= start and start-r.end <= 45 and
                    text[r.start:r.end].casefold() == match.group(1).casefold() and
                    re.fullmatch(r"[\s.,]*(?:(?:गया|गई|గయా)[\s.,]*)?", text[r.end:start]) and
                    re.match(r"\s*(?:में|లో|in\b)", after, re.I)
                    for r in results
                )
                if personal or repeated:
                    results.append(RecognizerResult("address", start, end, .8,
                        recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                              "boundary_evidence": "personal_place_frame"}))
        for match in self.asr_address.finditer(text):
            if match.group(1).casefold() in self.places:
                start,end=match.span(1)
                results.append(RecognizerResult('address',start,end,.7,
                    recognition_metadata={'recognizer_name':self.name,'recognizer_identifier':self.id,
                                          'boundary_evidence':'asr_address_with_place_support'}))
        for match in self.question_field.finditer(text):
            start,end = match.span(1)
            # Possessive answers such as "theirs is <town>" contain a
            # grammatical prefix, not a multiword place name.
            answer_prefix = re.match(r"(?:వాళ్ళది|వాళ్లది|మాది|నాది)\s+", text[start:end])
            if answer_prefix:
                start += answer_prefix.end()
            value = text[start:end]
            for word in re.finditer(r"\S+", value):
                if word.group().casefold() in {"సార్", "అండి", "sir", "doctor", "गया", "गई"}:
                    end = start + word.start()
                    break
            end = start + len(text[start:end].rstrip())
            value = text[start:end]
            if not value or is_suppressed("LOCATION",value) or value in {"ఇక్కడ","చిన్న","here","there"}:
                continue
            results.append(RecognizerResult("address",start,end,.85,
                recognition_metadata={"recognizer_name":self.name,"recognizer_identifier":self.id,
                                      "boundary_evidence":"address_answer"}))
            # A brief correction/repetition after the answer is evidence only
            # inside this same address Q&A, not every "huh" in a transcript.
            tail = text[end:end+90]
            for correction in re.finditer(rf"^[\s.,]*(?:huh\s*\?|[?])\s*({_NAME_WORD})(?:,\s*({_NAME_WORD}))?",tail,re.I):
                for group in (1,2):
                    a,b = correction.span(group)
                    if a<0 or is_suppressed("LOCATION",tail[a:b]):
                        continue
                    results.append(RecognizerResult("address",end+a,end+b,.8,
                        recognition_metadata={"recognizer_name":self.name,"recognizer_identifier":self.id,
                                              "boundary_evidence":"address_answer_variant"}))
        # Unlisted villages can be recovered from an actual personal workplace
        # or origin discussion plus a local place relation. The suffix alone
        # never makes a word PII.
        village = re.compile(
            rf"(?<![{_INDIC_LETTER}A-Za-z])"
            rf"({_NAME_WORD}(?:[ \t]+{_NAME_WORD})?(?:పల్లి|పేట|పాడు|పురం))"
            rf"(?=$|[\s.,!?;]|లో|కి|కు)")
        for match in village.finditer(text):
            start, end = match.span(1)
            before, after = text[max(0, start-100):start], text[end:end+50]
            personal = re.search(r"(?:పని\s*చేస్తా|టిఫిన్\s*షాప్|మా\s*ఊరు|మీ\s*ఊరు|ఏ\s*ఊరు|వాళ్లది|వాళ్ళది)", before)
            relation = re.match(rf"(?:\s*(?:దగ్గర|మీద|నుంచి)|లో|[.,\s]+{_NAME_WORD}\s+దగ్గర)", after)
            if personal and relation and not is_suppressed("LOCATION", match[1]):
                results.append(RecognizerResult("address", start, end, .9,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "boundary_evidence": "personal_village_relation"}))
        for match in self.address_field.finditer(text):
            start, end = match.span(1)
            surface = text[start:end]
            # A full street address needs both a house/flat number and a postal
            # code. This prevents a clinical clause ending in a number from
            # being treated as an address just because someone says "address".
            if not re.search(r"\d", surface[:-6]):
                continue
            results.append(RecognizerResult("address", start, end, .9,
                recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                      "boundary_evidence": "explicit_address"}))
        cleaned = []
        for result in results:
            boundary = (result.recognition_metadata or {}).get("boundary_evidence")
            if boundary in {"explicit_address", "explicit_house_number"}:
                cleaned.append(result)
                continue
            start, end = normalize_place_span(text, result.start, result.end)
            surface = text[start:end]
            boundary = (result.recognition_metadata or {}).get("boundary_evidence")
            # Normalization can trim "12 hours to Ulavapadu near" down to
            # "hours". Recheck the final value, not only the raw proposal.
            if boundary in {"landmark_relation", "personal_landmark_relation"}:
                last = surface.split()[-1] if surface.split() else ""
                geographic = surface.casefold() in self.places or bool(re.search(
                    r"(?:పల్లి|పేట|పాడు|పురం|నగర్|వాడ|గ్రామం|కోట|కుంట|గూడెం)$", last))
                if not geographic:
                    continue
            if start < end and not implausible_place(surface):
                result.start, result.end = start, end
                cleaned.append(result)
        return cleaned

    def _telugu_cue_results(self, text):
        """Emit place values from bounded Telugu and romanized cue families."""
        generic = {
            "ఇల్లు", "ఇంట్లో", "ఇంటి", "ఆసుపత్రి", "ఆసుపత్రిలో", "హాస్పిటల్",
            "ఇక్కడ", "అక్కడ", "ఎక్కడ", "ఊరు", "గ్రామం", "జిల్లా", "మండలం",
            "ఉంటాను", "ఉంటున్నాను", "వచ్చాను", "నుంచి", "నుండి", "లో", "కి", "కు",
            "నేను", "మేము", "నా", "మా", "నాకు", "మాకు",
            "home", "hospital", "here", "there", "place", "ooru", "intlo",
        }
        results = []
        frames = (
            (self.telugu_residence_cue, "personal_residence", .88),
            (self.telugu_origin_cue, "personal_origin", .88),
            (self.telugu_destination, "personal_destination", .86),
            (self.telugu_belonging, "personal_hometown", .87),
            (self.telugu_field, "explicit_place_field", .9),
            (self.roman_residence, "romanized_personal_residence", .86),
            (self.roman_belonging, "romanized_personal_hometown", .84),
            (self.telugu_address, "explicit_address", .9),
            (self.telugu_house, "explicit_house_number", .9),
        )
        for pattern, evidence, score in frames:
            for match in pattern.finditer(text):
                start, end = match.span("place")
                surface = text[start:end].strip(" \t,;:.\u200c\u200d")
                end = start + len(surface)
                # If the captured location runs into a grammatical predicate,
                # preserve only the first bounded place phrase.
                parts = surface.split()
                for i, part in enumerate(parts):
                    if part.casefold() in generic or part in {"ఉంటాను", "ఉన్నాను", "నుండి", "నుంచి"}:
                        if i == 0:
                            surface = ""
                        else:
                            surface = " ".join(parts[:i])
                        end = start + len(surface)
                        break
                if not surface or surface.casefold() in generic or is_suppressed("LOCATION", surface):
                    continue
                if evidence in {"personal_hometown", "explicit_place_field"}:
                    stem, attached_suffix = split_telugu_suffix(surface, kind="place")
                    if attached_suffix:
                        surface = stem.rstrip()
                        end = start + len(surface)
                if evidence == "explicit_house_number":
                    first_number = re.match(r"[0-9\u0c66-\u0c6f]+", surface)
                    if first_number is None or parse_telugu_number(first_number.group()) is None:
                        continue
                # Sandhi can merge a stem and locative ending (for example a
                # stem altered before లో); keep that entire token rather than
                # asserting an unattested stem boundary.  Separable suffixes
                # and joiners remain outside the reported span.
                suffix = match.groupdict().get("suffix") or ""
                if suffix.endswith("లో") and surface.endswith("ు") and match.start("suffix") == end:
                    end = match.end("suffix")
                elif end < len(text) and text[end] in "\u200c\u200d" and end > start and text[end - 1] == "్":
                    # The joiner belongs to the preceding virama cluster.
                    end += 1
                results.append(RecognizerResult("address", start, end, score,
                    recognition_metadata={"recognizer_name": self.name,
                                          "recognizer_identifier": self.id,
                                          "boundary_evidence": evidence}))
        # A relation word identifies a locality or landmark independently of a
        # gazetteer.  Generic buildings and discourse terms are still excluded.
        for match in self.telugu_relation.finditer(text):
            start, end = match.span("place")
            surface = text[start:end].strip()
            before = text[max(0, start-60):start]
            leading = re.match(r"(?:నేను|మాది|మా|నా)\s+", surface)
            if leading:
                start += leading.end()
                surface = text[start:end]
            if surface.casefold() in generic or is_suppressed("LOCATION", surface):
                continue
            personal = bool(re.search(r"(?:నేను|మాది|మా\s+ఊరు|నా\s+చిరునామా|ఇంటి\s+నెం\.?)\s*$", before))
            # "near" describes anatomy and arbitrary objects as readily as a
            # village.  A landmark relation needs independent geographic
            # evidence: a known place or a conventional locality ending.
            place_root = surface.split()[-1] if surface.split() else ""
            place_shape = bool(re.search(r"(?:పల్లి|పేట|పాడు|పురం|నగర్|వాడ|గ్రామం|కోట|కుంట|గూడెం)$", place_root))
            if not (surface.casefold() in self.places or place_shape):
                continue
            results.append(RecognizerResult("address", start, end, .84 if personal else .76,
                recognition_metadata={"recognizer_name": self.name,
                                      "recognizer_identifier": self.id,
                                      "boundary_evidence": "personal_landmark_relation" if personal else "landmark_relation"}))
        return results


class LabelledHealthIdRecognizer(KeyedGroupRecognizer):
    """An identifier explicitly labelled as a health ID, medical number or ABHA.

    Formats vary by issuer. The field gives type evidence; grouped digits
    alone do not establish a medical ID or certify a valid ABHA number.
    """

    def __init__(self):
        super().__init__(
            supported_entity="MEDICAL_ID",
            regex=(
                r"(?:\b(?:health|medical)[ \t]+(?:id(?:entifier)?|(?:record[ \t]+)?number)\b|"
                r"\babha(?:[ \t]+(?:id|number))?\b|"
                r"(?:मेडिकल|हेल्थ|स्वास्थ्य)[ \t]+(?:आईडी|नंबर|नम्बर|संख्या)(?![\u0900-\u097F])|"
                r"(?<![\u0C00-\u0C7F])(?:హెల్త్|ఆరోగ్య|మెడికల్|వైద్య)"
                r"(?:[ \t]+(?:రికార్డు|రికార్డ్))?[ \t]+"
                r"(?:ఐడి|ఐడీ|నంబర్|నెంబర్|సంఖ్య)(?![\u0C00-\u0C7F]))"
                r"[ \t]*(?:(?:number|no\.?|is|నంబర్|నెంబర్)[ \t]+)?[:=#]?[ \t]*"
                r"(\d+(?:[ \t-]+\d+)*|[A-Z0-9]+(?:[-_][A-Z0-9]+)*)(?![A-Z0-9_-])"
            ),
            score=.95,
            name="LabelledHealthIdRecognizer",
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for result in super().analyze(text, entities, nlp_artifacts):
            value = text[result.start:result.end]
            compact = re.sub(r"[\s_-]", "", value)
            if not 4 <= len(compact) <= 32 or not any(char.isdigit() for char in compact):
                continue
            result.recognition_metadata["boundary_evidence"] = "explicit_health_id"
            results.append(result)
        return results



class ConversationalPhoneRecognizer(EntityRecognizer):
    """Protect digit fragments in bounded, explicit contact-number answers.

    Echoed groups keep their original offsets. They are not silently joined to
    fabricate a normalized phone number, and unrelated quantities are excluded.
    """
    FIELD = re.compile(r"\b(?:phone|mobile|contact)\s*(?:number|no\.?)\b|మొబైల్\s*నెంబర్|ఫోన్[\s\u200c\u200d]*(?:నంబర్|నెంబర్)|"
                       r"(?:मोबाइल|फ़ोन|फोन)\s*(?:नंबर|नम्बर)|தொலைபேசி\s*எண்|ಫೋನ್\s*(?:ಸಂಖ್ಯೆ|ನಂಬರ್)|ফোন\s*নম্বর|ഫോൺ\s*നമ്പർ",re.I)
    VALUE = re.compile(r"(?<!\d)\+?\d{1,15}(?:[ \t-]+\d{1,15})*(?!\d)")
    NEXT = re.compile(r"\b(?:complaint|symptoms?|age|address|date|dob|mrn|pin|otp|blood|sugar|dose|tablets?)\b|"
                      r"వయసు|పేరు|షుగర్|మాత్ర|తేదీ|ఆధార్|उम्र|आयु|नाम|पता|आधार|வயது|பெயர்|வಯಸ್ಸು|ಹೆಸರು|বয়স|নাম|വയസ്സ്|പേര്",re.I)

    def __init__(self):
        super().__init__(supported_entities=["PHONE_NUMBER"],name="ConversationalPhoneRecognizer",supported_language="en")

    def load(self):
        pass

    def analyze(self,text,entities,nlp_artifacts=None):
        if entities and "PHONE_NUMBER" not in entities:
            return []
        results=[]
        for field in self.FIELD.finditer(text):
            answer=text[field.end():field.end()+220]
            next_field=self.NEXT.search(answer)
            if next_field:
                answer=answer[:next_field.start()]
            first_value=self.VALUE.search(answer)
            # A contact label in ordinary prose is not a numeric answer.
            if not first_value or first_value.start()>45:
                continue
            age_value=re.search(r"(\d{1,3})\s*(?:years?|yrs?|ఇయర్స్|ఏళ్ళు|ಸಂವತ್ಸರ|साल)",answer,re.I)
            if age_value:
                preceding=[re.sub(r'\D','',m.group()) for m in self.VALUE.finditer(answer[:age_value.start()])]
                if preceding and all(v==age_value.group(1) for v in preceding):
                    # Echoes of an age do not establish a contact-number answer.
                    continue
            for match in self.VALUE.finditer(answer):
                if re.match(r"\s*(?:mg|ml|kg|%|years?|ఇయర్స్|సంవత్సరాలు|साल)",answer[match.end():],re.I):
                    break
                start,end=field.end()+match.start(),field.end()+match.end()
                results.append(RecognizerResult("PHONE_NUMBER",start,end,.9,
                    recognition_metadata={"recognizer_name":self.name,"recognizer_identifier":self.id,
                                          "boundary_evidence":"explicit_contact_answer",
                                          "partial_identifier":len(re.sub(r"\D","",match.group()))<8}))
        return results


class CredentialFieldRecognizer(EntityRecognizer):
    """Recover opaque values from explicit credential fields, including quotes."""
    FIELD=re.compile(r"\b(password|passphrase|username|secret|api\s*key|access\s*token)\s*(?:is|:|=)\s*",re.I)
    LABELS={'passphrase':'password','api key':'api_key','access token':'access_token'}

    def __init__(self):
        super().__init__(supported_entities=['password','username','secret','api_key','access_token'],
                         name='CredentialFieldRecognizer',supported_language='en')

    def load(self):
        pass

    def analyze(self,text,entities,nlp_artifacts=None):
        results=[]
        for field in self.FIELD.finditer(text):
            word=' '.join(field.group(1).lower().split())
            label=self.LABELS.get(word,word)
            if entities and label not in entities:
                continue
            start=field.end(); value=text[start:]
            if not value:
                continue
            if value[0] in {'"',"'"}:
                stop=re.search(r'(?<!\\)'+re.escape(value[0]),value[1:])
                if not stop:
                    continue
                start+=1;end=start+stop.start()
            else:
                stop=re.search(r"[;\n]|\.(?=\s|$)|\s+(?=(?:password|username|secret|DOB|MRN)\s*[:=])",value,re.I)
                end=field.end()+(stop.start() if stop else len(value))
                end=len(text[:end].rstrip())
            if end>start:
                results.append(RecognizerResult(label,start,end,.95,
                    recognition_metadata={'recognizer_name':self.name,'recognizer_identifier':self.id,
                                          'boundary_evidence':'explicit_credential'}))
        return results


def build_indic_identifier_recognizers() -> List[EntityRecognizer]:
    """Recognizers that work identically in Latin and Indic scripts."""
    return [
        AadhaarRecognizer(),
        PANRecognizer(),
        LabelledPanRecognizer(),
        UPIHandleRecognizer(),
        IndianPhoneRecognizer(),
        LabelledHealthIdRecognizer(),
        ConversationalPhoneRecognizer(),
        CredentialFieldRecognizer(),
        KeyedGroupRecognizer("card_cvv",r"\b(?:CVV|CVC)\s*(?:is|:|=)?\s*(\d{3,4})(?!\d)",.95,"LabelledCvvRecognizer"),
        # A bare PIN/OTP is PII only next to its keyword, so the keyword is
        # matched but excluded from the reported span.
        KeyedGroupRecognizer(
            supported_entity="PASSWORD",
            regex=r"(?:UPI[ ]?)?(?:PIN|OTP|password|passcode|पिन|पासवर्ड|ఓటిపి|பின்|ಪಿನ್|পিন|പിൻ)\W{0,12}?(\d{4,8})\b",
            score=0.75,
            name="UpiPinRecognizer",
        ),
        # MRNs are facility-specific; the labelled form is unambiguous.
        KeyedGroupRecognizer(
            supported_entity="MEDICAL_RECORD_NUMBER",
            regex=(r"(?<!\w)(?:(?:MRN|(?:medical[ \t]+)?record[ \t]+(?:number|no\.?|id)|"
                   r"chart[ \t]+(?:number|no\.?|id)|patient[ \t]+(?:id|identifier)|"
                   r"(?:మెడికల్[ \t]+)?రికార్డు[ \t]+(?:నంబర్|నెంబర్|సంఖ్య)|రోగి[ \t]+గుర్తింపు[ \t]+సంఖ్య|"
                   r"(?:मेडिकल[ \t]+)?रिकॉर्ड[ \t]+(?:नंबर|संख्या))"
                   r"(?:[ \t]*[:=#][ \t]*|[ \t]+(?:is[ \t]+)?)|(?=MRN[-_]?[A-Z0-9]))"
                   r"((?=[A-Z\d/_-]*\d)[A-Z\d]{1,16}(?:[-_/][A-Z\d]{1,12}){0,3})(?![\w/-])"),
            score=0.95,
            name="LabelledMrnRecognizer",
        ),
        IndicNameContextRecognizer(),
        TeluguNameCueRecognizer(),
        IndicPersonNameGazetteer(),
        AgeRecognizer(),
        ShortDateRecognizer(),
        ClinicalCalendarDateRecognizer(),
        SpokenPlaceRecognizer(),
    ]
