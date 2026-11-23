"""Shared Hindi cardinal numbers used by detector rules.

Spellings are intentionally explicit: Hindi cardinals below 100 are irregular.
The alternatives in ``NUMBER_VARIANTS`` need native-speaker review.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterator


_ONE_TO_NINETY_NINE = (
    "एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
    "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस "
    "इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस "
    "इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
    "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ "
    "इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर "
    "इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी अस्सी "
    "इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे "
    "इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_ONE_TO_NINETY_NINE) == 99

HINDI_NUMBER_WORDS: dict[int, str] = {
    **{n: word for n, word in enumerate(_ONE_TO_NINETY_NINE, 1)},
    100: "सौ",
    **{n: f"एक सौ {_ONE_TO_NINETY_NINE[n - 101]}" for n in range(101, 121)},
}

# Alternate written forms occur across reference counting charts and speech.
NUMBER_VARIANTS: dict[int, tuple[str, ...]] = {
    5: ("पांच",), 6: ("छः", "छ:"), 15: ("पन्द्रह",),
    28: ("अठाईस", "अठ्ठाईस"), 29: ("उन्नतीस",), 39: ("उनचालीस",),
    43: ("तियालीस",), 45: ("पैतालीस",), 53: ("तिरेपन",),
    63: ("तिरेसठ",), 67: ("सतसठ",), 78: ("अठत्तर",),
    79: ("उन्यासी",), 98: ("अठानवे",), 100: ("एक सौ",),
}


def _fold_number_word(word: str) -> str:
    """Fold common orthographic variants while preserving input offsets."""
    word = unicodedata.normalize("NFC", word).replace("\u200c", "").replace("\u200d", "")
    word = word.replace("ँ", "ं")
    word = word.replace("़", "")
    return " ".join(word.split())


_WORD_TO_NUMBER = {
    _fold_number_word(form): number
    for number, canonical in HINDI_NUMBER_WORDS.items()
    for form in (canonical, *NUMBER_VARIANTS.get(number, ()))
}
for _number in range(101, 121):
    _WORD_TO_NUMBER[_fold_number_word(f"सौ {_ONE_TO_NINETY_NINE[_number - 101]}")] = _number
_WORD_TO_NUMBER.update({_fold_number_word("डेढ़"): 1.5, _fold_number_word("ढाई"): 2.5})

# Transliteration is a matching aid for romanized Hindi. It is derived from
# the same 1–120 written forms, rather than maintaining a second number list.
_ROMAN_CONSONANTS = dict(zip(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
    "k kh g gh ng ch chh j jh ny t th d dh n t th d dh n p ph b bh m y r l v sh sh s h l".split(),
))
_ROMAN_VOWELS = dict(zip("अआइईउऊएऐओऔऋ", "a aa i ee u oo e ai o au ri".split()))
_ROMAN_MATRAS = dict(zip("ािीुूेैोौृ", "a i i u u e ai o au ri".split()))


def _romanize_cardinal(word: str, *, short_vowels: bool = False) -> str:
    out = []
    chars = (unicodedata.normalize("NFC", word) if short_vowels else _fold_number_word(word))
    vowels = {**_ROMAN_VOWELS, **({"आ": "a", "ई": "i", "ऊ": "u"} if short_vowels else {})}
    for index, char in enumerate(chars):
        if char in _ROMAN_CONSONANTS:
            out.append(_ROMAN_CONSONANTS[char])
            following = chars[index + 1] if index + 1 < len(chars) else ""
            if following in _ROMAN_MATRAS:
                continue
            if following == "्":
                continue
            if following and unicodedata.category(following).startswith("L"):
                out.append("a")
        elif char in _ROMAN_MATRAS:
            out.append(_ROMAN_MATRAS[char])
        elif char in vowels:
            out.append(vowels[char])
        elif char in "ंँ":
            out.append("n")
        elif char == "ः":
            out.append("h")
        elif char not in "़्\u200c\u200d":
            out.append(char)
    return "".join(out)


_ROMAN_TO_NUMBER = {
    rendering: number
    for number, canonical in HINDI_NUMBER_WORDS.items()
    for form in (canonical, *NUMBER_VARIANTS.get(number, ()))
    for rendering in (_romanize_cardinal(form), _romanize_cardinal(form, short_vowels=True))
}


def parse_hindi_number(surface: str) -> int | float | None:
    """Parse a cardinal aged 1–120, including mixed ASCII/Devanagari digits."""
    stripped = surface.strip()
    if not stripped:
        return None
    if all(unicodedata.category(ch) == "Nd" for ch in stripped):
        value = int("".join(str(unicodedata.decimal(ch)) for ch in stripped))
        return value if 1 <= value <= 120 else None
    return _WORD_TO_NUMBER.get(_fold_number_word(stripped), _ROMAN_TO_NUMBER.get(stripped.casefold()))


_FORMS = sorted(
    {form for number, canonical in HINDI_NUMBER_WORDS.items()
     for form in (canonical, *NUMBER_VARIANTS.get(number, ()))}
    | {f"सौ {_ONE_TO_NINETY_NINE[n - 101]}" for n in range(101, 121)}
    | {"डेढ़", "ढाई"}, key=len, reverse=True,
)
_NUMBER_RE = re.compile(
    r"(?<![\u0900-\u0963\u0966-\u097f\w])(?:[0-9\u0966-\u096f]{1,3}|"
    + "|".join(re.escape(form).replace(r"\ ", r"\s+") for form in _FORMS)
    + r")(?=$|[\s,.;:!?।॥|])"
)
_ROMAN_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(form).replace(r"\ ", r"\s+")
                                  for form in sorted(_ROMAN_TO_NUMBER, key=len, reverse=True))
    + r")(?![A-Za-z])", re.I,
)


def iter_hindi_number_spans(text: str) -> Iterator[tuple[int, int, int | float]]:
    """Yield original-codepoint spans and values for Hindi numbers."""
    for match in _NUMBER_RE.finditer(text):
        value = parse_hindi_number(match.group())
        if value is not None:
            yield match.start(), match.end(), value
    for match in _ROMAN_NUMBER_RE.finditer(text):
        value = parse_hindi_number(match.group())
        if value is not None:
            yield match.start(), match.end(), value
