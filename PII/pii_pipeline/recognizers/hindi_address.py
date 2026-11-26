"""Cue-backed Hindi residential addresses and personal place statements.

The recognizer deliberately does not use the runtime place gazetteer.  A
street number needs an address cue, and an unnumbered place needs an explicit
residence, origin, or address-field frame.  All offsets refer to the input.
"""

from __future__ import annotations

import re
import unicodedata

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult


_D = r"[0-9\u0966-\u096f]"
_LETTER = r"[A-Za-z\u0900-\u097f]"
_WORD = rf"{_LETTER}[A-Za-z\u0900-\u097f\u200c\u200d]*"
_HOUSE_NUMBER = rf"{_D}{{1,4}}(?:[-/]{_D}{{1,4}}){{0,3}}(?:[-/]?[A-Za-z\u0900-\u097f]{{1,3}})?"
_NUMBER_MARK = r"(?:नं\.?|न\.?|नंबर|नम्बर|संख्या|no\.?|number|#)"
_HOUSE_CUE = r"(?:मकान|घर|फ्लैट|प्लॉट|flat|plot|house)"
_STREET_CUE = r"(?:गली|रोड|मार्ग|सेक्टर|street|road|sector)"
_LOCALITY_CUE = r"(?:मोहल्ला|नगर|कॉलोनी|चौक|चौराहा|ग्राम|गाँव|गांव|पोस्ट|डाकघर|थाना|तहसील|ज़िला|जिला|शहर)"
_PLACE_WORD = rf"{_WORD}(?:[ \t]+{_WORD}){{0,2}}"

_HOUSE = re.compile(
    rf"(?<!{_LETTER})(?:{_HOUSE_CUE})[ \t]*(?:{_NUMBER_MARK}[ \t]*)?"
    rf"(?P<number>{_HOUSE_NUMBER})(?![0-9\u0966-\u096f])",
    re.I,
)
_ROOM = re.compile(
    rf"(?<!{_LETTER})(?:कमरा|कक्ष|room)[ \t]*(?:{_NUMBER_MARK}[ \t]*)?"
    rf"(?P<number>{_HOUSE_NUMBER})(?![0-9\u0966-\u096f])", re.I,
)
_STREET = re.compile(
    rf"(?<!{_LETTER})(?:{_STREET_CUE})[ \t]*(?:{_NUMBER_MARK}[ \t]*)?"
    rf"(?:{_HOUSE_NUMBER}|{_PLACE_WORD})(?!{_LETTER})",
    re.I,
)
_FIELD = re.compile(r"(?<![A-Za-z\u0900-\u097f])(?:पता|निवास(?: स्थान)?|address)[ \t]*[:：=]?[ \t]*", re.I)
_PIN = re.compile(rf"(?<!{_D})[1-9\u0967-\u096f]{_D}{{5}}(?!{_D})")
_DATE = re.compile(rf"^{_D}{{1,2}}[/-]{_D}{{1,2}}[/-]{_D}{{2,4}}$")
_BP = re.compile(rf"^{_D}{{2,3}}/{_D}{{2,3}}$")
_CLAUSE_STOP = re.compile(
    r"(?:[।!?;\n|]|,?[ \t]+(?:और|लेकिन|जबकि|आयु|उम्र|नाम|बीपी|नब्ज़|नब्ज|दवा|खुराक|"
    r"मिनट|मिलीग्राम|एमजी|नर्स|चिकित्सक|डॉक्टर|काम[ \t]+कर|भर्ती|इलाज|पढ़|रहता|रहती|रहते|हूँ|हूं|हैं|है)(?=$|[ \t,।!?;]))",
    re.I,
)
_POSTPOSITION = re.compile(r"[ \t]+(?:में|मे|से|पर|तक|के[ \t]+पास)(?=$|[ \t,।!?;])")
_STRUCTURE = re.compile(
    rf"(?<!{_LETTER})(?:{_STREET_CUE}|{_LOCALITY_CUE})"
    rf"(?:[ \t]+(?:{_NUMBER_MARK})?[ \t]*(?:{_HOUSE_NUMBER}|{_PLACE_WORD}))?",
    re.I,
)
_RESIDENCE = re.compile(
    rf"(?<!{_LETTER})(?:मैं|मै|हम)[ \t]+(?P<place>{_PLACE_WORD})[ \t]+"
    r"(?:में|मे|से)[ \t]+(?:रहता|रहती|रहते|आया|आई|आए|हूँ|हूं|हैं)"
    r"(?:[ \t]+(?:हूँ|हूं|हैं))?(?=$|[ \t,।!?;,])",
    re.I,
)
_THIRD_PERSON_RESIDENCE = re.compile(
    r"[ \t]+(?:में|मे)[ \t]+(?:रहता|रहती|रहते)[ \t]+(?:है|हैं)(?=$|[ \t,।!?;,])",
    re.I,
)
_PERSONAL_ORIGIN = re.compile(
    rf"(?<!{_LETTER})(?:मैं|मै|हम)[ \t]+(?P<place>{_PLACE_WORD})[ \t]+से[ \t]+(?:आया|आई|आए|लौटा|लौटी|लौटे)[ \t]+(?:हूँ|हूं|हैं|है)",
    re.I,
)
_PERSONAL_HOME = re.compile(
    rf"(?<!{_LETTER})(?:मेरा|मेरी|हमारा|हमारी|और)[ \t]+घर[ \t]+(?P<place>{_PLACE_WORD})[ \t]+(?:में|मे)(?=$|[ \t,।!?;])",
    re.I,
)
_RETURNED_TO_PLACE = re.compile(
    rf"(?:से|मैं|मै|हम)[ \t]+(?P<place>{_PLACE_WORD})[ \t]+(?:लौटा|लौटी|लौटे|आया|आई|आए)[ \t]+(?:हूँ|हूं|हैं|है)",
    re.I,
)
_FAMILY_CONTEXT = re.compile(r"(?:माता|पिता|पत्नी|पति|बेटा|बेटी|मरीज़|रोगी)[^।!?;\n]{0,60}$")
_GENERIC = frozenset({
    "घर", "अस्पताल", "हॉस्पिटल", "दुकान", "कार्यालय", "ऑफिस", "कमरा",
    "यहाँ", "यहां", "वहाँ", "वहां", "गाँव", "गांव", "शहर", "बाहर",
})


def _number_valid(surface: str) -> bool:
    """Reject measurements and calendar dates that resemble street numbers."""
    number = surface.strip()
    if _DATE.fullmatch(number) or _BP.fullmatch(number):
        return False
    digits = [unicodedata.digit(char) for char in number if char.isdecimal()]
    return bool(digits) and len(digits) <= 12


def _trim_end(text: str, start: int, end: int) -> int:
    """Keep address punctuation internal, but leave syntax outside the span."""
    while end > start and text[end - 1] in " \t,.-:：" :
        end -= 1
    return end


def _bounded_address(text: str, start: int, minimum_end: int) -> int:
    """Extend a cue through contiguous address components, up to a short clause."""
    limit = min(len(text), start + 160)
    tail = text[minimum_end:limit]
    stop = _CLAUSE_STOP.search(tail)
    if stop:
        limit = minimum_end + stop.start()
    postposition = _POSTPOSITION.search(text[minimum_end:limit])
    if postposition:
        limit = minimum_end + postposition.start()
    candidate = text[start:limit]
    pin = list(_PIN.finditer(candidate))
    if pin:
        return _trim_end(text, start, start + pin[-1].end())
    # The first few comma groups may carry street, locality and city.  A
    # sentence without delimiters is bounded to six trailing words.
    groups = re.split(r",[ \t]*", candidate)
    if len(groups) > 1:
        groups = groups[:4]
        limit = start + len(", ".join(groups))
    else:
        rest = list(re.finditer(r"\S+", text[minimum_end:limit]))
        if len(rest) > 6:
            limit = minimum_end + rest[5].end()
    return _trim_end(text, start, max(minimum_end, limit))


class HindiAddressRecognizer(PatternRecognizer):
    """Recognize Hindi house/street addresses and cue-backed personal places."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity="address",
            name="HindiAddressRecognizer",
            supported_language="en",
            patterns=[Pattern(name="hindi_house", regex=_HOUSE.pattern, score=0.88)],
        )

    def _result(self, start: int, end: int, score: float, evidence: str) -> RecognizerResult:
        return RecognizerResult(
            entity_type="address", start=start, end=end, score=score,
            recognition_metadata={
                "recognizer_name": self.name,
                "recognizer_identifier": self.id,
                "boundary_evidence": evidence,
                "source": "rule",
            },
        )

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and "address" not in entities:
            return []
        results = []
        for match in _HOUSE.finditer(text):
            if not _number_valid(match.group("number")):
                continue
            end = _bounded_address(text, match.start(), match.end())
            results.append(self._result(match.start(), end, .91, "explicit_house_number"))
        for match in _ROOM.finditer(text):
            if not _number_valid(match.group("number")):
                continue
            before = text[max(0, match.start() - 40):match.start()]
            if not re.search(r"(?:पता|निवास|address|मकान|घर|फ्लैट|प्लॉट)\s*[:：=,]?\s*$", before, re.I):
                continue
            end = _bounded_address(text, match.start(), match.end())
            results.append(self._result(match.start(), end, .89, "explicit_address_room"))
        for match in _FIELD.finditer(text):
            start = match.end()
            if start >= len(text):
                continue
            end = _bounded_address(text, start, start)
            value = text[start:end]
            if not value or value.strip().casefold() in _GENERIC:
                continue
            if not (_HOUSE.search(value) or _ROOM.search(value) or _STRUCTURE.search(value) or _PIN.search(value)):
                continue
            results.append(self._result(start, end, .90, "explicit_address_field"))
        for match in _STREET.finditer(text):
            if any(r.start <= match.start() < r.end for r in results):
                continue
            end = _bounded_address(text, match.start(), match.end())
            results.append(self._result(match.start(), end, .83, "explicit_street"))
        for match in _RESIDENCE.finditer(text):
            start, end = match.span("place")
            value = text[start:end].strip().casefold()
            if value in _GENERIC or any(r.start <= start < r.end for r in results):
                continue
            results.append(self._result(start, end, .87, "personal_residence"))
        for pattern in (_PERSONAL_ORIGIN, _PERSONAL_HOME, _RETURNED_TO_PLACE):
            for match in pattern.finditer(text):
                if pattern is _PERSONAL_HOME and text[match.start():match.end()].startswith("और"):
                    before = text[max(0, match.start() - 70):match.start()]
                    if not re.search(r"(?:मैं|मेरा|मेरी|हमारा|हमारी|हम)(?=$|[ \t])", before):
                        continue
                start, end = match.span("place")
                value = text[start:end].strip().casefold()
                if value not in _GENERIC and not any(r.start <= start < r.end for r in results):
                    results.append(self._result(start, end, .87, "personal_origin_or_home"))
        for match in _THIRD_PERSON_RESIDENCE.finditer(text):
            before = text[max(0, match.start() - 90):match.start()]
            if not _FAMILY_CONTEXT.search(before):
                continue
            words = list(re.finditer(rf"{_WORD}", before))
            if not words:
                continue
            last = words[-1]
            start = match.start() - len(before) + last.start()
            end = match.start() - len(before) + last.end()
            if text[start:end].casefold() not in _GENERIC and not any(r.start <= start < r.end for r in results):
                results.append(self._result(start, end, .84, "family_residence"))
        unique = {(r.start, r.end, r.entity_type): r for r in results}
        return list(unique.values())
