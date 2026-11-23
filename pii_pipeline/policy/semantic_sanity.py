"""Cheap type checks for conversational name, place, and account guesses.

These checks require a *value* to look like its claimed entity. A grammatical
cue ("near", "my name", or a locative suffix) is only supporting evidence.
"""
from __future__ import annotations

import re
from functools import lru_cache

from ..telugu_utils import split_telugu_suffix
from ..data_assets import require_data_path

_BODY = frozenset({
    "చెయ్యి", "చేయి", "చేతి", "కాలు", "తల", "పెదవి", "లిప్", "ఛాతీ", "ఛాతి", "మోకాలు", "భుజం", "కడుపు", "వీపు",
    "கை", "கால்", "தலை", "உதடு", "மார்பு", "முழங்கால்", "தோள்", "வயிறு", "முதுகு",
    "ಕೈ", "ಕಾಲು", "ತಲೆ", "ತುಟಿ", "ಎದೆ", "ಮೊಣಕಾಲು", "ಭುಜ", "ಹೊಟ್ಟೆ", "ಬೆನ್ನು",
    "hand", "arm", "leg", "head", "lip", "lips", "chest", "shoulder", "knee", "stomach", "back",
})
_MEDICAL = frozenset({
    "pain", "swelling", "ache", "symptom", "treatment", "medicine", "tablet", "fever",
    "పెయిన్", "నొప్పి", "స్వెల్లింగ్", "వాపు", "గ్యాస్ట్రిక్", "ప్రాబ్లం", "మందు", "చికిత్స",
    "வலி", "வீக்கம்", "மருந்து", "ನೋವು", "ಊತ", "ಔಷಧ",
    # Common conditions remain clinical vocabulary when a model changes its
    # type guess under a different window context.
    "allergy", "thyroid", "అలర్జీ", "థైరాయిడ్", "தைராய்டு", "ಥೈರಾಯ್ಡ್",
})
_TIME = frozenset({
    "ఒక", "టూ", "డేస్", "బ్యాక్", "రోజులు", "నెలలు", "గంటలు", "ఇయర్స్",
    "one", "two", "days", "day", "weeks", "week", "months", "month", "years", "year", "ago", "yesterday", "today",
    "ஒரு", "நாள்", "நாட்கள்", "ದಿನ", "ದಿನಗಳು", "ಒಂದು",
    "పొద్దున", "మధ్యాహ్నం", "రాత్రి", "ఉదయం", "సాయంత్రం",
})
_SPEECH = frozenset({
    "చేసి", "రాసి", "చూసి", "సెట్", "స్టార్ట్", "అన్నారు", "రమ్మ", "ఇస్తా", "ఇస్తాను", "డేట్",
    "ఎవరో", "నీకు", "మీది", "బయట", "ఇక్కడ", "అక్కడ", "దగ్గర", "లో", "వద్ద", "నుంచి",
    "ఇంట్లో", "ఇంట్", "ఇంటి", "ఇల్లు", "ఆసుపత్రి", "హాస్పిటల్",
    "and", "or", "is", "was", "the", "near", "from", "at", "in", "here", "there",
    "అబ్బాయి", "బాయ్", "అత్తమ్మ", "పాప్", "ఎంత", "వయసు", "వయస్సు",
    # Honorifics, pronouns, repeat markers and conjunctions are grammatical
    # evidence, not person/place values. Do not depend on a model's score.
    "ఆయన", "ఆమె", "వాళ్ళు", "వాళ్లు", "మళ్లా", "మళ్ళా", "ma'am", "sir",
    "rather", "than", "రథర్", "దెన్",
})
_TELUGU_VERB = re.compile(r"(?:స్తా(?:ను|రు)?|తున్నా(?:ను|రు|డు)?|తాను|తాడు|తారు|న్నారు|వద్దు|చేసి|రాసి|చూసి)$")
_USERNAME_CUE = re.compile(r"(?:user\s*(?:name|id)|handle|account|profile|login|యూజర్\s*(?:నేమ్|ఐడి)|பயனர்|ಬಳಕೆದಾರ)\s*(?::|is|=)?\s*$", re.I)
_PLACE_PREFIX = re.compile(r"^(?:బయట|outside|near|దగ్గర)\s+", re.I)
_PLACE_STOP = frozenset({"ఎవరో", "నీకు", "నాకు", "ఉంది", "ఉన్నారు", "చెప్పాను", "అన్నారు", "దగ్గర", "వద్ద", "నుంచి", "లో", "near", "from", "there", "here", "pain", "swelling"})
_RELATIVE_DATE = re.compile(
    r"(?:yesterday|today|tomorrow|last\s+(?:night|week|month|year)|"
    r"this\s+(?:week|month|year)|कल|आज|परसों|निन्न|నిన్న|ఈరోజు|రేపు|"
    r"நேற்று|இன்று|நாளை|ನಿನ್ನೆ|ಇಂದು|ನಾಳೆ)", re.I,
)
_PASSWORD_CONTEXT = re.compile(
    r"(?:password|passcode|login\s+(?:code|pin)|పాస్\s*వర్డ్|पासवर्ड|"
    r"கடவுச்சொல்|ಗುಪ್ತಪದ)\s*(?::|is|=)?\s*$", re.I,
)
_ACCOUNT_CONTEXT = re.compile(
    r"(?:account|bank|iban|ఖాతా|అకౌంట్|खाता|बैंक|கணக்கு|வங்கி|ಖಾತೆ|ಬ್ಯಾಂಕ್)"
    r"(?:\s+(?:number|no\.?|id|నంబర్|नंबर|எண்|ಸಂಖ್ಯೆ))?\s*(?::|is|=)?\s*$", re.I,
)
_BIRTH_CONTEXT = re.compile(r"(?:date\s+of\s+birth|born|dob|birthday|పుట్టిన|జన్మ|जन्म|பிறந்த|ಹುಟ್ಟಿದ)", re.I)
_CONTACT_CONTEXT = re.compile(r"(?:phone|mobile|call|contact|ఫోన్|మొబైల్|కాల్|फोन|मोबाइल|தொலைபேசி|ಮೊಬೈಲ್)", re.I)
_US_ID_CONTEXT = re.compile(r"(?:driver.?s?\s+licen[cs]e|health\s+plan|beneficiary|medicare)", re.I)
_PUBLIC_EVENT_CUES = (
    re.compile(r"\b(?:goal|score|referee|penalty|football|cricket|wicket|team|league|stadium|tournament)\b", re.I),
    re.compile(r"\b(?:world\s+cup|championship|commentator|broadcast|play.by.play)\b", re.I),
)
_CLINICAL_CONTEXT = re.compile(
    r"\b(?:doctor|patient|hospital|clinic|symptom|diagnosis|treatment|medicine|pain|fever)\b|"
    r"డాక్టర్|పేషెంట్|నొప్పి|జ్వరం|వైద్య|चिकित्सक|डॉक्टर|दर्द|மருத்துவர்|வலி|ವೈದ್ಯ|ನೋವು",
    re.I,
)


@lru_cache(maxsize=1)
def _known_places() -> frozenset[str]:
    path = require_data_path("data/gazetteers/indic_places.txt")
    return frozenset(line.split("#", 1)[0].strip().casefold()
                     for line in path.read_text(encoding="utf-8").splitlines()) if path.is_file() else frozenset()


def is_known_place(surface: str) -> bool:
    return surface.casefold().strip() in _known_places()


def _words(surface: str) -> list[str]:
    return [word.strip(" \t.,;:!?।'’\"-\u200c\u200d").casefold() for word in surface.split()]


def implausible_person(surface: str) -> bool:
    words = _words(surface)
    if not 1 <= len(words) <= 3 or len(surface) > 55 or any(not word for word in words):
        return True
    # A cue cannot turn a predicate, a duration, or a body symptom into a name.
    if any(word in _BODY | _MEDICAL | _TIME | _SPEECH or _TELUGU_VERB.search(word) for word in words):
        return True
    without_initials = re.sub(r"\b[A-Za-z]\.\s*", "", surface) if len(words) > 1 else surface
    return bool(re.search(r"\d|[.!?;:]", without_initials))


def implausible_place(surface: str) -> bool:
    words = _words(surface)
    if not 1 <= len(words) <= 12 or len(surface) > 120 or any(not word for word in words):
        return True
    # Positional grammar is equally common in anatomy descriptions.
    return any(word in _BODY | _MEDICAL | _TIME | _SPEECH or _TELUGU_VERB.search(word) for word in words)


def normalize_place_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Remove discourse and detachable Telugu locatives from a place guess."""
    while (match := _PLACE_PREFIX.match(text[start:end])) is not None:
        start += match.end()
    tokens = list(re.finditer(r"\S+", text[start:end]))
    if tokens:
        for index, token in enumerate(tokens):
            word = token.group().strip(".,;:!?।")
            if index and word.casefold() in _PLACE_STOP:
                end = start + tokens[index - 1].end()
                break
        for token in tokens:
            if start + token.start() >= end:
                break
            word = token.group().rstrip(".,;:!?।")
            stem, suffix = split_telugu_suffix(word.rstrip("\u200c\u200d"), kind="place")
            if suffix:
                # A locative closes the place phrase; words after it belong
                # to the sentence, even if the model included them.
                end = min(end, start + token.start() + len(stem.rstrip("\u200c\u200d")))
                break
    return start, end


def unsupported_username(text: str, start: int, end: int) -> bool:
    value = text[start:end]
    if value.startswith("@") or _USERNAME_CUE.search(text[max(0, start - 55):start]):
        return False
    # An unlabelled handle has identifier syntax; plain prose does not.
    return not bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{2,31}", value)
                    and re.search(r"[0-9_.-]", value))


def unsupported_ambiguous_model_type(text: str, start: int, end: int, entity_type: str) -> bool:
    """Require independent evidence for types often assigned to ordinary prose.

    A relative day is useful clinical history but is not an identifying date.
    Password/account predictions need field context or strong identifier syntax;
    a model score cannot make an ordinary clinical abbreviation a credential.
    """
    value = text[start:end]
    before = text[max(0, start - 55):start]
    if entity_type in {"SENSITIVE_DATE", "TRANSACTION_DATE"}:
        compact = re.sub(r"[.,;!?]+", " ", value).strip()
        return bool(_RELATIVE_DATE.fullmatch(re.sub(r"\s+", " ", compact)))
    if entity_type == "DATE_OF_BIRTH":
        return not bool(_BIRTH_CONTEXT.search(text[max(0, start - 75):min(len(text), end + 55)]))
    if entity_type == "PASSWORD":
        if _PASSWORD_CONTEXT.search(before):
            return False
        # A standalone token with all four character classes is sufficiently
        # password-like. Ordinary words and medical measurements lack this shape.
        strong_shape = (8 <= len(value) <= 64 and not re.search(r"\s", value)
                        and re.search(r"[a-z]", value) and re.search(r"[A-Z]", value)
                        and re.search(r"\d", value) and re.search(r"[!@#$%^&*?+_-]", value))
        return not bool(strong_shape)
    if entity_type == "ACCOUNT_NUMBER":
        return not bool(_ACCOUNT_CONTEXT.search(before))
    return False


def unsupported_unlabelled_structured_type(text: str, start: int, end: int, recognizer: str) -> bool:
    """Check generic structured detectors against contact/US-ID evidence."""
    value = text[start:end]
    near = text[max(0, start - 70):min(len(text), end + 45)]
    if recognizer == "PhoneRecognizer":
        digits = re.sub(r"\D", "", value)
        if _CONTACT_CONTEXT.search(near):
            return False
        # Dotted small numbers and slash-separated BP readings can resemble
        # phone formats but are not standalone contact identifiers.
        return len(digits) < 8 or "/" in value
    if recognizer in {"UsLicenseRecognizer", "HealthPlanBeneficiaryRecognizer"}:
        return not bool(_US_ID_CONTEXT.search(near))
    return False


def is_public_event_commentary(text: str) -> bool:
    """Identify public play-by-play discourse without patient context.

    Public figures and event locations in such a transcript are not the
    private/patient identifiers this pipeline is tasked with redacting. The
    gate requires several distinct event cues and abstains on clinical text.
    """
    if _CLINICAL_CONTEXT.search(text):
        return False
    first_group = {m.group().casefold() for m in _PUBLIC_EVENT_CUES[0].finditer(text)}
    second_group = {m.group().casefold() for m in _PUBLIC_EVENT_CUES[1].finditer(text)}
    return len(first_group | second_group) >= 4
