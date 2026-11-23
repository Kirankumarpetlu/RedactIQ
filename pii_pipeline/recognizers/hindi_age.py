"""Cue validated Hindi age spans in original Python string offsets."""
from __future__ import annotations

import re
from typing import Iterator

from ..hindi_utils import iter_hindi_number_spans


_BOUNDARY = r"(?=$|[\s,.;:!?।॥|])"
_YEAR = re.compile(r"^\s*(?:साल|वर्ष|बरस|वर्षीय|saal|sal|varsh|baras)" + _BOUNDARY, re.I)
_SHORT = re.compile(r"^\s*(?:महीन(?:े|ा)|माह|दिन|mahine|maah|din)" + _BOUNDARY, re.I)
_BARE = re.compile(r"^\s*(?:का|की|के|ka|ki|ke)\s+(?:हूँ|हूं|है|हैं|हैँ|था|थी|थे|hoon|hun|hai)" + _BOUNDARY, re.I)
_STRONG_BEFORE = re.compile(r"(?<![A-Za-z])(?:उम्र|आयु|वय(?:स|स्क)|umar|aayu|age)\s*[:=]?\s*(?:(?:लगभग|करीब|क़रीब|तक़रीबन|lagbhag|karib)\s*)?$", re.I)
_PERSON_BEFORE = re.compile(
    r"(?:मैं|मै|मेरी|मेरा|मेरे|main|mai|meri|mera|mere|bachcha|bachchi|beta|beti|बच्चा|बच्ची|बेटा|बेटी|पुत्र|पुत्री|लड़का|लड़की|महिला|पुरुष|"
    r"पत्नी|पति|मरीज़|मरीज|रोगी|व्यक्ति|शिशु|नवजात)"
    r"(?:[\s\u0900-\u097f]{0,24})$"
)
_PERSON_AFTER = re.compile(r"^\s*(?:का|की|के|ka|ki|ke)?\s*(?:हूँ|हूं|है|हैं|हैँ|था|थी|थे|hai|hoon|hun|"
                           r"महिला|पुरुष|बच्चा|बच्ची|लड़का|लड़की|व्यक्ति|मरीज़|मरीज)" + _BOUNDARY, re.I)
_NON_AGE_AFTER = re.compile(
    r"^\s*(?:मिनट|घंटे?|सेकंड|मिलीग्राम|मिग्रा|मि\.ग्रा|मिलीलीटर|लीटर|एमएल|"
    r"mg|mcg|ml|kg|gm|mmhg|bpm|%|प्रतिशत|रुपये|रूपये|₹|पृष्ठ|पेज|"
    r"कमरा|वार्ड|बेड|मंजिल|फ्लोर)" + _BOUNDARY, re.I
)
_NON_AGE_BEFORE = re.compile(
    r"(?:कमरा|वार्ड|बेड|पेज|पृष्ठ|मंजिल|फ्लोर|बीपी|नब्ज़|नब्ज|दवा|खुराक|"
    r"मात्रा|कीमत|राशि|कुल|नंबर|नं\.?|संख्या)\s*[:=#]?\s*$"
)
_DURATION_AFTER = re.compile(r"^\s*(?:से|तक|के\s+लिए|पहले|बाद|की\s+गर्भवती)" + _BOUNDARY)


def iter_hindi_age_spans(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield ``(start, end, evidence)`` for personal age expressions.

    A stated year unit plus person context, an explicit age field, or a bare
    ``का/की`` answer is required. Durations, measurements and item counts are
    rejected before a candidate reaches Presidio or the shared policy.
    """
    for start, number_end, value in iter_hindi_number_spans(text):
        before = re.split(r"[।॥|!?\n]", text[max(0, start - 65):start])[-1]
        after = text[number_end:number_end + 48]
        if _NON_AGE_AFTER.match(after) or _NON_AGE_BEFORE.search(before):
            continue
        strong = bool(_STRONG_BEFORE.search(before))
        person = bool(_PERSON_BEFORE.search(before))
        year = _YEAR.match(after)
        short = _SHORT.match(after)
        bare = _BARE.match(after)
        if year:
            suffix = after[year.end():]
            if _DURATION_AFTER.match(suffix):
                continue
            if not (strong or person or _PERSON_AFTER.match(suffix) or "वर्षीय" in year.group()):
                continue
            if isinstance(value, float) and value not in (1.5, 2.5):
                continue
            end = number_end + year.end()
            ending = re.match(r"^\s+(?:का|की|के)" + _BOUNDARY, text[end:])
            if ending:
                end += ending.end()
            yield start, end, "explicit_hindi_age_unit"
        elif short:
            suffix = after[short.end():]
            if not strong and not (person and _PERSON_AFTER.match(suffix)):
                continue
            if _DURATION_AFTER.match(suffix):
                continue
            end = number_end + short.end()
            ending = re.match(r"^\s+(?:का|की|के)" + _BOUNDARY, text[end:])
            if ending:
                end += ending.end()
            yield start, end, "explicit_hindi_child_age"
        elif strong or (bare and person):
            if isinstance(value, float):
                continue
            yield start, number_end, "explicit_hindi_age_cue"
