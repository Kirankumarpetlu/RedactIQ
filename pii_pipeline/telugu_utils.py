"""Shared Telugu text helpers for recognizers and IndicNER enrichment.

These helpers operate on original Python string offsets. A suffix is removed
only when the proposed cut is an akshara boundary. Sandhi that changes the
stem's spelling is deliberately left intact.
"""
from __future__ import annotations

import re
import unicodedata


TELUGU_NUMBER_WORDS = {
    "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5,
    "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9,
    "పది": 10, "పదకొండు": 11, "పన్నెండు": 12, "పదమూడు": 13,
    "పద్నాలుగు": 14, "పదిహేను": 15, "పదహారు": 16,
    "పదిహేడు": 17, "పద్దెనిమిది": 18, "పందొమ్మిది": 19,
    "ఇరవై": 20, "ముప్పై": 30, "నలభై": 40, "యాభై": 50,
    "అరవై": 60, "డెబ్బై": 70, "ఎనభై": 80, "తొంభై": 90,
    "వంద": 100,
}

_ONES = tuple(word for word, number in TELUGU_NUMBER_WORDS.items() if number < 10)
_TENS = tuple(word for word, number in TELUGU_NUMBER_WORDS.items() if number >= 20)
_TEENS = tuple(word for word, number in TELUGU_NUMBER_WORDS.items() if 10 <= number < 20)
TELUGU_NUMBER_PATTERN = (
    rf"(?:\d{{1,3}}|(?:{'|'.join(map(re.escape, _TENS))})"
    rf"(?:[ \t]+(?:{'|'.join(map(re.escape, _ONES))}))?"
    rf"|{'|'.join(map(re.escape, _TEENS))}"
    rf"|{'|'.join(map(re.escape, _ONES))})"
)


def parse_telugu_number(value: str) -> int | None:
    """Parse Unicode decimal digits or a supported Telugu one/tens phrase."""
    if value.isdecimal():
        try:
            return int(value)
        except ValueError:
            return None
    words = value.split()
    if len(words) == 1:
        return TELUGU_NUMBER_WORDS.get(words[0])
    if len(words) == 2:
        tens = TELUGU_NUMBER_WORDS.get(words[0], 0)
        ones = TELUGU_NUMBER_WORDS.get(words[1], 0)
        if tens >= 20 and tens % 10 == 0 and 0 < ones < 10:
            return tens + ones
    return None


# Longest endings first. This is a morphology table, never a list of names or
# places. A grammatical ending that changes the stem is not split by this table.
TELUGU_NAME_SUFFIXES = (
    "గారితో", "గారికి", "గారిని", "గారి", "గారు",
    "యొక్క", "నుండి", "నుంచి", "నికి", "తో", "కి", "కు", "ని", "ను",
)
TELUGU_PLACE_SUFFIXES = (
    "నుండి", "నుంచి", "లోని", "లోనే", "యొక్క", "లో", "కు", "కి", "తో",
)
TELUGU_HONORIFICS = ("శ్రీమతి", "డాక్టర్", "శ్రీ", "డా.")
_JOINERS = frozenset({"\u200c", "\u200d"})
_VIRAMA = "\u0c4d"


def is_akshara_boundary(text: str, index: int) -> bool:
    """Conservative Telugu cluster boundary in Python code-point offsets."""
    if not 0 <= index <= len(text):
        return False
    if index in (0, len(text)):
        return True
    previous, following = text[index - 1], text[index]
    # ZWNJ closes a conjunct: the boundary after it is safe before a new
    # consonant. ZWJ keeps the consonant in the same shaping cluster.
    if previous == _VIRAMA or previous == "\u200d":
        return False
    if following in _JOINERS or unicodedata.category(following) in {"Mn", "Mc", "Me"}:
        return False
    return True


def split_telugu_suffix(surface: str, kind: str = "name") -> tuple[str, str]:
    """Return stem and detachable ending, or the unsplit original surface.

    The minimum stem length only guards trivial one-code-point false cuts. It
    does not assert that the stem is present in any gazetteer.
    """
    if kind not in {"name", "place"}:
        raise ValueError("kind must be 'name' or 'place'")
    suffixes = TELUGU_NAME_SUFFIXES if kind == "name" else TELUGU_PLACE_SUFFIXES
    for suffix in suffixes:
        if not surface.endswith(suffix):
            continue
        boundary = len(surface) - len(suffix)
        # A final ఉ before attached లో can reflect a changed nominative stem.
        # Without an independent stem boundary, retain the whole sandhi token.
        if kind == "place" and suffix == "లో" and surface[:boundary].endswith("ు"):
            return surface, ""
        if boundary >= 2 and is_akshara_boundary(surface, boundary):
            return surface[:boundary], suffix
    return surface, ""


def trim_telugu_honorifics(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim standalone leading titles and trailing గారు from a proposed span."""
    if not 0 <= start <= end <= len(text):
        raise ValueError("invalid span")
    while start < end:
        leading = re.match(r"\s*(?:" + "|".join(map(re.escape, TELUGU_HONORIFICS)) + r")\s+", text[start:end])
        if not leading:
            break
        start += leading.end()
    trailing = re.search(r"\s+గారు\s*$", text[start:end])
    if trailing:
        end = start + trailing.start()
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
