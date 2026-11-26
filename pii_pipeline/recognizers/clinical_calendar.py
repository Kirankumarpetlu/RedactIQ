"""Contextual clinical calendar dates; numeric doses and durations stay readable."""
from __future__ import annotations

import re
from datetime import date

from presidio_analyzer import EntityRecognizer, RecognizerResult

from ..policy.context_filters import valid_calendar_literal, has_clinical_year_context


# Literal spellings preserve source offsets; no transliteration or text rewrite.
_MONTH_GROUPS = [
    "January Jan జనవరి जनवरी ஜனவரி ಜನವರಿ",
    "February Feb ఫిబ్రవరి फरवरी பிப்ரவரி ಫೆಬ್ರವರಿ",
    "March Mar మార్చి మార్చ్ मार्च மார்ச் ಮಾರ್ಚ್",
    "April Apr ఏప్రిల్ अप्रैल ஏப்ரல் ಏಪ್ರಿಲ್",
    "May మే मई மே ಮೇ",
    "June Jun జూన్ जून ஜூன் ಜೂನ್",
    "July Jul జూలై जुलाई ஜூலை ಜುಲೈ",
    "August Aug ఆగస్టు अगस्त ஆகஸ்ட் ಆಗಸ್ಟ್",
    "September Sep Sept సెప్టెంబర్ सितंबर செप्टेम्बर செப்டம்பர் ಸೆಪ್ಟೆಂಬರ್",
    "October Oct అక్టోబర్ अक्टूबर அக்டோபர் ಅಕ್ಟೋಬರ್",
    "November Nov నవంబర్ नवंबर நவம்பர் ನವೆಂಬರ್",
    "December Dec డిసెంబర్ दिसंबर டிசம்பர் ಡಿಸெம்பர் ಡಿಸೆಂಬರ್",
]
_MONTHS = {word.casefold(): number for number, group in enumerate(_MONTH_GROUPS, 1)
           for word in group.split()}
_MONTH = "|".join(re.escape(word) for word in sorted(_MONTHS, key=len, reverse=True))
_MONTH_DAY = re.compile(
    rf"(?<!\w)(?P<month>{_MONTH})[\s.-]+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*,?\s*(?P<year>(?:19|20)\d{{2}}))?(?!\d)", re.I,
)
_DAY_MONTH = re.compile(
    rf"(?<!\w)(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[\s.-]*(?P<month>{_MONTH})(?!\w)"
    rf"(?:\s*,?\s*(?P<year>(?:19|20)\d{{2}}))?", re.I,
)
_MONTH_ONLY = re.compile(
    rf"(?<![A-Za-z\u0900-\u0D7F])(?:{_MONTH})"
    rf"(?=(?:లోనే|లో|இல்|ನಲ್ಲಿ)(?![A-Za-z\u0900-\u0D7F])|(?![A-Za-z\u0900-\u0D7F]))", re.I)
_EVENT = re.compile(
    r"\b(?:dob|born|birth|date|visit|appointment|consultation|seen|came|tested|testing|"
    r"admitted|discharged|surgery|operation|lmp|last\s+period|menstrual\s+period|"
    r"treatment\s*(?:started|starts?|began)|last\s+test|test\s*(?:done|was)|"
    r"done\s+by[^.!?\n]{1,60}\bon|"
    r"(?:echo|angio|angiogram|scan|procedure)\s*(?:(?:was|is)\s*)?(?:done|performed)|"
    r"blood\s+reports?|investigations?\s*(?:(?:was|were|is|are)\s*)?(?:done|dated))\b|"
    r"ట్రీట్మెంట్|పీరియడ్|తారీకు|తేదీ|చేయించుకున్న|చూయించుకున్న|వచ్చావు|పుట్టిన|"
    r"ఆపరేషన్|శస్త్రచికిత్స|మందు(?:లు)?\s*(?:వాడ|తీసు|మొదలు|తాగ)|"
    r"బ్లడ్\s*రిపోర్ట్స్|ఇన్వెస్టిగేషన్స్\s*డన్|"
    r"(?:ఎకో|ఎక్కో|ఆంజియో|స్కాన్)\s*(?:డన్|చేయించ)|"
    r"तारीख|जन्म|मासिक|ದಿನಾಂಕ|ಜನ್ಮ|தேதி|பிறந்த|எக்கோ(?:ட)?|அறுவை\s*சிகிச்சை", re.I,
)
_BIRTH_FIELD = re.compile(
    r"(?:\bdob|date\s*of\s*birth|birth\s*date|born\s*(?:on|in)|"
    r"పుట్టిన\s*తేదీ|जन्म\s*तिथि|பிறந்த\s*தேதி|ಜನ್ಮ\s*ದಿನಾಂಕ)\s*[:=]?\s*$", re.I,
)
_VALUE_UNIT = re.compile(
    r"^\s*(?:mg|mcg|ml|kg|gm|mmhg|cm|mm|bpm|%|days?|weeks?|months?|years?|"
    r"డేస్|రోజులు|నెలలు|మంత్స్|ఇయర్స్|दिन|महीने)(?![A-Za-z])", re.I,
)
_DOSE_TOPIC = re.compile(r"\b(?:tablet|capsule|dose|dosage|pulse|bp|weight)\b|టాబ్లెట్|బిపి|బీపీ", re.I)
_DAY_WORD = r"(?:date|డేట్|తారీకు|తేదీ|तारीख|ದಿನಾಂಕ|தேதி)"
_EXPLICIT_DAY = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?!\d)(?:st|nd|rd|th|వ)?\s*{_DAY_WORD}(?:కి)?(?!\w)|"
    rf"{_DAY_WORD}\s*(?:is|on|ఇస్|:|=)?\s*(\d{{1,2}})(?!\d)", re.I,
)
# Spoken days require a calendar word, not just a Telugu number in prose.
_SPOKEN_DAYS = (
    "ముప్పై ఒకటో", "ఇరవై తొమ్మిదో", "ఇరవై ఎనిమిదో", "ఇరవై ఏడో", "ఇరవై ఆరో",
    "ఇరవై ఐదో", "ఇరవై నాలుగో", "ఇరవై మూడో", "ఇరవై రెండో", "ఇరవై ఒకటో",
    "పంతొమ్మిదో", "పద్దెనిమిదో", "పదిహేడో", "పదహారో", "పదిహేనో", "పద్నాలుగో",
    "పదమూడో", "పన్నెండో", "పదకొండో", "ముప్పయ్యో", "ఇరవయ్యో", "పదో",
    "తొమ్మిదో", "తొమ్మిది", "ఎనిమిదో", "ఏడో", "ఆరో", "ఐదో", "నాలుగో", "మూడో", "రెండో", "ఒకటో",
)
_SPOKEN_DAY_VALUE = "|".join(re.escape(day) for day in _SPOKEN_DAYS)
_SPOKEN_DAY = re.compile(rf"(?<!\w)({_SPOKEN_DAY_VALUE})[\s.,]*{_DAY_WORD}(?!\w)", re.I)
_SPOKEN_DAY_ECHO = re.compile(
    rf"(?<![\w\u0c00-\u0c7f])(?P<first>{_SPOKEN_DAY_VALUE})[.!?][ \t]+"
    rf"(?P<second>(?P=first))(?![\w\u0c00-\u0c7f])"
)
_MONTH_SPOKEN_DAY = re.compile(rf"ఈ\s+నెల\s+({_SPOKEN_DAY_VALUE})(?!\w)")
_SPOKEN_DAY_LINK = re.compile(
    rf"(?<!\w)({_SPOKEN_DAY_VALUE})\s+(?:కాబట్టి\s+ఆ\s+డేట్|తర్వాత\s+డేట్)(?!\w)")
_ONSET_YEAR = re.compile(
    r"(?<!\d)((?:19|20)\d{2})(?!\d)\s*(?:నుంచి\s*(?:షుగర్|బీపీ|బీపి|బిపి)\s*(?:కూడా\s*)?ఉంద(?:ా|ి)|"
    r"(?:diabetes|hypertension)\s*(?:started|began))", re.I)
_ENGLISH_ONSET_YEAR = re.compile(
    r"\b(?:diagnosed\s+in|(?:diabetes|hypertension)\s+since)\s+((?:19|20)\d{2})(?!\d)", re.I)
_DAY_ACTION = re.compile(
    r"(?<![\d/])(\d{1,2})(?![\d/])\s*(?:తీసుకున్నావు|వచ్చావు|రాయలేదు|వచ్చాను|"
    r"ఎప్పుడు|came\b|visited\b|when\b)", re.I,
)
_DAY_LIST = re.compile(
    r"(?<!\d)(\d{1,2}(?:\s*,\s*\d{1,2}){1,3})(?!\d)\s*(?:రాశాను|వచ్చావు|visited\b)", re.I,
)
_NUMERIC_DATE = re.compile(r"(?<![\d/])\d{1,2}/\d{1,2}(?:/\d{2,4})?(?![\d/])")
_DAY_REPEAT_GAP = re.compile(
    r"^(?:[\s.,;:!?—-]|ఏంటి|ఎప్పుడు|వన్|ఇయర్|ఆ|అవును|date|డేట్|on|the)*$", re.I,
)


def _numeric_date_fragment(text, start, end):
    return text[start:end].isdigit() and bool(
        re.search(r"\d[/.-]$", text[max(0, start-2):start])
        or re.match(r"[/.-]\d", text[end:end+2])
    )


class ClinicalCalendarDateRecognizer(EntityRecognizer):
    """Emit only date values backed by a calendar/clinical-event context."""

    def __init__(self):
        super().__init__(supported_entities=["sensitive_date", "date_of_birth"],
                         name="ClinicalCalendarDateRecognizer", supported_language="en")

    def load(self):
        pass

    def _result(self, text, start, end, evidence, score=.85):
        label = "date_of_birth" if _BIRTH_FIELD.search(text[max(0, start-65):start]) else "sensitive_date"
        return RecognizerResult(label, start, end, score,
            recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                  "boundary_evidence": evidence})

    @staticmethod
    def _has_context(text, start, end):
        return bool(_EVENT.search(text[max(0, start-100):end+100]) or (
            re.search(r"మేము[ \t]+$", text[max(0, start-15):start])
            and re.match(r"[ \t]+చేశాం[.!?।]", text[end:end+20])
        ))

    def analyze(self, text, entities, nlp_artifacts=None):
        if entities and not set(entities) & set(self.supported_entities):
            return []
        results = []
        for pattern in (_MONTH_DAY, _DAY_MONTH):
            for match in pattern.finditer(text):
                month, day = _MONTHS[match["month"].casefold()], int(match["day"])
                try:
                    date(int(match["year"] or 2000), month, day)
                except ValueError:
                    continue
                if _VALUE_UNIT.match(text[match.end():]) or not self._has_context(text, *match.span()):
                    continue
                results.append(self._result(text, *match.span(), "clinical_month_day", .9))
        for match in _MONTH_ONLY.finditer(text):
            if any(r.start <= match.start() < r.end for r in results):
                continue
            # An invalid or unit-bearing month/day must not fall back to a
            # shorter month-only entity. "May I return?" is not the month May.
            tail = text[match.end():match.end()+30]
            before = text[max(0, match.start()-40):match.start()]
            if re.match(r"\s*(?:weather|climate|festival|temperatures?)\b", tail, re.I):
                continue
            if re.match(r"[\s.-]*\d", tail):
                continue
            if match.group().casefold() in {"may", "మే"} and re.match(
                    r"\s*(?:i\b|we\b|you\b|be\b|need\b|have\b|return\b|బీ)", tail, re.I):
                continue
            if re.search(r"(?:\bdoctor|\bdr\.?|name\s*(?:is|:)|డాక్టర్)\s*$", before, re.I):
                continue
            # Telugu మార్చి also means "having changed". A bare occurrence
            # needs a month/preposition cue, rather than merely nearby treatment.
            if match.group() == "మార్చి" and not (
                    re.search(r"(?:\bin|\bduring|\bmonth|నెల|తేదీ)\s*$", before, re.I)
                    or re.match(r"\s*నెల", tail)):
                continue
            if self._has_context(text, *match.span()):
                results.append(self._result(text, *match.span(), "clinical_month"))
        for match in _EXPLICIT_DAY.finditer(text):
            group = 1 if match.group(1) is not None else 2
            start, end = match.span(group)
            if (1 <= int(match.group(group)) <= 31 and not _VALUE_UNIT.match(text[end:])
                    and text[max(0, start-1):start] != "/" and text[end:end+1] != "/"):
                results.append(self._result(text, start, end, "explicit_calendar_day"))
        for pattern in (_SPOKEN_DAY, _MONTH_SPOKEN_DAY, _SPOKEN_DAY_LINK, _ONSET_YEAR, _ENGLISH_ONSET_YEAR):
            for match in pattern.finditer(text):
                start, end = match.span(1)
                results.append(self._result(text, start, end, "explicit_calendar_day" if
                    pattern in (_SPOKEN_DAY, _MONTH_SPOKEN_DAY, _SPOKEN_DAY_LINK) else "clinical_onset_year"))
        for match in re.finditer(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text):
            if has_clinical_year_context(text, *match.span()):
                results.append(self._result(text, *match.span(), "clinical_onset_year", .9))

        # A number is a spoken visit-day only inside a nearby date discussion,
        # with an immediate visit verb and no intervening dosage topic.
        # Reject fragments before creating anchors, so a partial DOB also
        # cannot provide evidence for surrounding bare numbers.
        results = [r for r in results if not _numeric_date_fragment(text, r.start, r.end)]
        anchors = [(r.start, r.end) for r in results]
        anchors.extend(match.span() for match in _NUMERIC_DATE.finditer(text)
                       if valid_calendar_literal(match.group()) and not _VALUE_UNIT.match(text[match.end():])
                       and not _DOSE_TOPIC.search(text[max(0, match.start()-20):match.start()]))

        def anchored(start, end):
            for a, b in anchors:
                gap = text[min(end, b):max(start, a)]
                if min(abs(start-a), abs(end-b)) <= 200 and not _DOSE_TOPIC.search(gap):
                    return True
            return False

        for match in _DAY_ACTION.finditer(text):
            start, end = match.span(1)
            if 1 <= int(match[1]) <= 31 and anchored(start, end):
                results.append(self._result(text, start, end, "spoken_visit_day"))
        for match in _DAY_LIST.finditer(text):
            if not anchored(*match.span(1)):
                continue
            for number in re.finditer(r"\d+", match[1]):
                if 1 <= int(number.group()) <= 31:
                    start, end = match.start(1)+number.start(), match.start(1)+number.end()
                    results.append(self._result(text, start, end, "spoken_visit_day_list"))
        for seed in list(results):
            if seed.recognition_metadata["boundary_evidence"] != "explicit_calendar_day":
                continue
            low = max(0, seed.start-65)
            if not text[seed.start:seed.end].isdigit():
                for match in re.finditer(r"(?<!\w)" + re.escape(text[seed.start:seed.end]) + r"(?!\w)",
                                         text[low:seed.end+65]):
                    start, end = low+match.start(), low+match.end()
                    if (start, end) == (seed.start, seed.end):
                        continue
                    gap = text[end:seed.start] if end <= seed.start else text[seed.end:start]
                    if len(gap) <= 60 and _DAY_REPEAT_GAP.fullmatch(gap):
                        results.append(self._result(text, start, end, "repeated_calendar_day"))
            for match in re.finditer(r"(?<!\d)\d{1,2}(?!\d)", text[low:seed.end+65]):
                start, end = low+match.start(), low+match.end()
                if match.group() != text[seed.start:seed.end] or (start, end) == (seed.start, seed.end):
                    continue
                gap = text[end:seed.start] if end <= seed.start else text[seed.end:start]
                if len(gap) <= 60 and _DAY_REPEAT_GAP.fullmatch(gap) and not _VALUE_UNIT.match(text[end:]):
                    results.append(self._result(text, start, end, "repeated_calendar_day"))
        for match in _SPOKEN_DAY_ECHO.finditer(text):
            if any(r.recognition_metadata["boundary_evidence"] == "explicit_calendar_day"
                   and text[r.start:r.end] == match["first"]
                   and abs(r.start - match.start()) <= 160
                   for r in results):
                for group in ("first", "second"):
                    results.append(self._result(text, *match.span(group), "repeated_calendar_day"))
        # A spoken day must not steal part of a complete numeric date from
        # ShortDateRecognizer. This includes hyphen/dot formats, not just
        # slash dates: e.g. "birth date 04-04-1994" is one date, not two days.
        results = [r for r in results if not _numeric_date_fragment(text, r.start, r.end)]
        unique = {(r.start, r.end, r.entity_type): r for r in results}
        return sorted((r for r in unique.values() if not entities or r.entity_type in entities),
                      key=lambda r: (r.start, r.end))
