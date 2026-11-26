"""Cue-backed Hindi person and organization recognizers.

These rules use grammatical evidence, not a list of private names or places.
They retain Python code-point offsets and leave titles and postpositions outside
the reported span. Model proposals remain separate evidence in the pipeline.
"""
from __future__ import annotations

import re
from typing import List, Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


_DEV = r"\u0900-\u0963\u0970-\u097f"
_HINDI_SCRIPT = re.compile(r"[\u0900-\u097f]")
_ROMAN_HINDI = re.compile(r"\b(?:mera|meri|mere)\s+naam\b|\b(?:main|mai)\s+\w+\s+(?:hoon|hun)\b", re.I)
_WORD = rf"[{_DEV}A-Za-z][{_DEV}A-Za-z'’\u200c\u200d]*"
_NAME_ATOM = rf"(?:[{_DEV}]{{1,3}}\.|[A-Za-z]\.|{_WORD})"
_TITLE = re.compile(r"^(?:(?:श्री(?:मती)?|सुश्री|कुमारी|डॉ\.?|डॉक्टर|प्रो\.?|Dr\.?)\s+)", re.I)
_SUFFIX = re.compile(r"(?:[\s-]*(?:जी|साहब|साहिबा|भाई|बहन|दीदी))$", re.I)
_NAME_STOP = frozenset({
    "है", "हैं", "हूँ", "हूं", "हुं", "का", "की", "के", "को", "से", "में", "मे", "पर",
    "ने", "तक", "और", "लेकिन", "जो", "क्या", "कौन", "नहीं", "नही", "मेरी", "मेरा", "मेरे",
    "आपका", "उनका", "रहता", "रहती", "रहते", "आया", "आई", "आईं", "आए", "काम", "उम्र", "आयु",
    "साल", "वर्ष", "ठीक", "थका", "बीमार", "खुश", "तैयार", "घर", "अस्पताल", "दुकान",
    "सूची", "दवा", "इलाज", "कंपनी", "मिनट", "मिलीग्राम", "नाम", "पता", "विवरण",
    "जी", "साहब", "साहिबा", "भाई", "बहन", "दीदी", "लिखा", "लिखी", "लिखे", "दर्ज", "hai", "hoon", "hun", "hu", "mein", "main", "mera", "meri", "naam", "umar", "kaam",
})
_ORG_GENERIC = frozenset({
    "अस्पताल", "हॉस्पिटल", "क्लिनिक", "कॉलेज", "विश्वविद्यालय", "विद्यालय", "संस्थान",
    "बैंक", "कंपनी", "दुकान", "ऑफिस", "कार्यालय", "स्कूल", "hospital", "clinic",
    "college", "university", "bank", "company", "school",
})
_ORG_SUFFIX = re.compile(
    r"(?:अस्पताल|हॉस्पिटल|क्लिनिक|मेडिकल\s+कॉलेज|कॉलेज|विश्वविद्यालय|विद्यालय|संस्थान|"
    r"बैंक|कंपनी|लिमिटेड|प्राइवेट\s+लिमिटेड|स्कूल|hospital|clinic|college|university|bank|company|ltd\.?|school)$",
    re.I,
)
_PERSON_CUE = re.compile(
    r"(?:मेरा\s+नाम|मेरी\s+नाम|(?:पिता|माता|पति|पत्नी|बेटा|बेटी|पुत्र|पुत्री|"
    r"सुपुत्र|सुपुत्री)\s+का\s+नाम|(?:रोगी|मरीज़|मरीज|पेशेंट)\s+का\s+नाम|"
    r"(?<![\u0900-\u097f])नाम|\bmera\s+naam\b|\bmeri\s+naam\b|\bname\b)\s*[:：-]?\s*",
    re.I,
)
_RELATION_CUE = re.compile(
    r"(?:मेरे|मेरी|मेरा|उनके|उनकी|उसके)?\s*(?:पिता|माता|पति|पत्नी|बेटा|बेटी|"
    r"पुत्र|पुत्री|सुपुत्र|सुपुत्री|धर्मपत्नी)\s+(?:श्री\s+)?",
    re.I,
)
_SELF_CUE = re.compile(r"(?:मैं|मै|मॆ|\bmain\b|\bmai\b)\s+", re.I)
_HONORIFIC = re.compile(r"(?:श्री|श्रीमती|सुश्री|कुमारी|डॉ\.?|डॉक्टर|प्रो\.?)\s+", re.I)
_RELATIONAL = re.compile(r"(?:पुत्र|सुपुत्र|पुत्री|सुपुत्री|पत्नी|धर्मपत्नी)\s+(?:श्री\s+)?", re.I)
_BY_CUE = re.compile(r"(?:द्वारा|c/o)\s*[:：-]?\s*", re.I)
_NAME_END = re.compile(r"^(?:\s*(?:है|हैं|हूँ|हूं|हुं|थी|था|थे|की|का|के|को|से|ने|आया|आई|"
                       r"रहता|रहती|रहते|है\s+और|hai|hoon|hun|hu)\b|[।|,;.!?]|$)", re.I)
_ORG_RELATION = re.compile(
    r"^\s*(?:में|मे|से|के\s+पास)\s+(?:काम\s+कर(?:ता|ती|ते)?|नौकरी|भर्ती|इलाज|"
    r"पढ़(?:ता|ती|ते)?|पढ(?:ता|ती|ते)?|जॉब|job|work|admitted|stud(?:y|ies))",
    re.I,
)
_ORG_NAMED_PRELUDE = re.compile(r"(?:कंपनी|बीमा\s+कंपनी|अस्पताल|क्लिनिक|विद्यालय|कॉलेज)\s*[:：]\s*$", re.I)
_GENERAL_FACT = re.compile(r"\bWHO\b|दिशानिर्देश|निर्देशों|खबर|समाचार", re.I)
_MEDICAL_CONTEXT = re.compile(r"(?:मुझे|मेरा|मेरी|मरीज़|मरीज|रोगी|patient|मेरे|उनको|उसे)\s*$", re.I)
_NON_NAME_PREDICATE = re.compile(
    r"(?:^|\s)(?:रहा|रही|रहे|रहोगे|हो|होते|जाइए|बोल(?:ता|ती|ते|तोय)?|"
    r"कर(?:ता|ती|ते)?|थोड़ा|बहुत|वही|है|हैं|am|having|will|some)(?=$|\s)", re.I)


def _token_spans(text: str, start: int, limit: int = 5) -> list[tuple[int, int]]:
    """Read adjacent name/organization words without crossing punctuation."""
    out = []
    cursor = start
    for _ in range(limit):
        gap = re.match(r"\s*", text[cursor:])
        cursor += gap.end() if gap else 0
        match = re.match(_NAME_ATOM, text[cursor:], re.I)
        if not match:
            break
        end = cursor + match.end()
        out.append((cursor, end))
        cursor = end
    return out


def _name_after(text: str, start: int, *, self_intro: bool = False) -> Optional[tuple[int, int]]:
    title = _TITLE.match(text[start:])
    if title:
        start += title.end()
    tokens = _token_spans(text, start, 4)
    if not tokens:
        return None
    accepted = []
    for a, b in tokens:
        word = text[a:b].casefold()
        if word in _NAME_STOP or any(ch.isdigit() for ch in word):
            break
        accepted.append((a, b))
    if not accepted:
        return None
    # A self-introduction without a field label needs its copula close by.
    if self_intro:
        valid = None
        for a, b in accepted:
            if re.match(r"\s*(?:(?:है|हैं|हूँ|हूं|हुं|hai|hoon|hun|hu)(?![\u0900-\u097fA-Za-z])|बोल\s+रह(?:ा|ी|े)\s+(?:हूँ|हूं|हैं|है))", text[b:], re.I):
                valid = (accepted[0][0], b)
                break
        return valid
    # A residence/origin phrase after a complete two-token name is a separate
    # place value, even when punctuation is absent in dictated notes.
    if len(accepted) >= 3 and re.match(r"\s*(?:में|मे|से)(?![\u0900-\u097fA-Za-z])", text[accepted[-1][1]:]):
        accepted.pop()
    return accepted[0][0], accepted[-1][1]


class HindiNameRecognizer(EntityRecognizer):
    """Recognize a bounded name following identity, relation or title evidence."""

    def __init__(self):
        super().__init__(supported_entities=["PERSON"], supported_language="en", name="HindiNameRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None) -> List[RecognizerResult]:
        if entities and "PERSON" not in entities:
            return []
        if not (_HINDI_SCRIPT.search(text) or _ROMAN_HINDI.search(text)):
            return []
        proposals: dict[tuple[int, int], RecognizerResult] = {}
        for pattern, evidence, score, is_self in (
            (_PERSON_CUE, "explicit_name", .94, False),
            (_RELATION_CUE, "identity_relation", .92, False),
            (_RELATIONAL, "identity_relation", .92, False),
            (_BY_CUE, "identity_relation", .92, False),
            (_HONORIFIC, "person_honorific", .89, False),
            (_SELF_CUE, "patient_introduction", .91, True),
        ):
            for cue in pattern.finditer(text):
                # A question or generic field mention has no bounded name value.
                found = _name_after(text, cue.end(), self_intro=is_self)
                if not found:
                    continue
                start, end = found
                suffix = _SUFFIX.search(text[start:end])
                if suffix:
                    end = start + suffix.start()
                if start >= end:
                    continue
                # Honorifics and relative words can also be part of a generic
                # phrase. Require at least one lexical name token after them.
                value = text[start:end].strip()
                if value.casefold() in _NAME_STOP or value.casefold() in _ORG_GENERIC or _ORG_SUFFIX.search(value):
                    continue
                # First-person copulas and professional titles introduce
                # predicates as often as names in dictated medical speech.
                # A weak cue needs a compact noun phrase without auxiliaries;
                # an explicit name field can still contain longer names.
                if evidence != "explicit_name" and (len(value.split()) > 2
                                                    or _NON_NAME_PREDICATE.search(value)
                                                    or value in {"श्री", "श्रीमती", "सुश्री"}):
                    continue
                result = RecognizerResult("PERSON", start, end, score,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "source": "rule", "boundary_evidence": evidence})
                key = (start, end)
                if key not in proposals or score > proposals[key].score:
                    proposals[key] = result
        return list(proposals.values())


class HindiOrganizationRecognizer(EntityRecognizer):
    """A named institution tied to work, study, treatment or insurance."""

    def __init__(self):
        super().__init__(supported_entities=["organization"], supported_language="en", name="HindiOrganizationRecognizer")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts: Optional[NlpArtifacts] = None) -> List[RecognizerResult]:
        if entities and "organization" not in entities:
            return []
        if not _HINDI_SCRIPT.search(text):
            return []
        results = []
        # Candidate starts are bounded by clauses, fields and personal pronouns.
        # The relation following the organization is mandatory; a suffix alone
        # cannot turn a general mention into a personal identifier.
        for relation in re.finditer(r"(?:में|मे|से|के\s+पास)\s+(?:काम|नौकरी|भर्ती|इलाज|पढ़|पढ|जॉब|job|work|admitted|stud)", text, re.I):
            if not _ORG_RELATION.match(text[relation.start():]):
                continue
            left = max(0, relation.start() - 85)
            prefix = text[left:relation.start()]
            split = list(re.finditer(r"[।|,;!?\n:]|(?:मैं|मै|मुझे|मेरे|मेरी|मेरा|हम|रोगी|मरीज़|मरीज|patient)\s+", prefix, re.I))
            start = left + (split[-1].end() if split else 0)
            candidate = text[start:relation.start()].strip()
            start += len(text[start:relation.start()]) - len(text[start:relation.start()].lstrip())
            if not candidate or len(candidate) > 65 or _GENERAL_FACT.search(candidate):
                continue
            words = candidate.split()
            if len(words) > 5 or any(word.casefold() in {"और", "लेकिन", "कि", "इस", "उस", "यह", "वह"} for word in words):
                continue
            if candidate.casefold() in _ORG_GENERIC:
                continue
            has_suffix = bool(_ORG_SUFFIX.search(candidate))
            # Without an organization suffix, a personal work cue still gives
            # useful evidence, but require a compact proper-name surface.
            if not has_suffix and (len(words) > 2 or any(w.casefold() in _NAME_STOP for w in words)):
                continue
            end = relation.start()
            while end > start and text[end - 1].isspace():
                end -= 1
            if end <= start:
                continue
            results.append(RecognizerResult("organization", start, end, .91 if has_suffix else .86,
                recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                      "source": "rule", "boundary_evidence": "personal_organization_relation",
                                      "suffix_evidence": has_suffix}))
        # Field and employment frames put the relationship before the value.
        for field in re.finditer(r"(?:मेरी|मेरा|हमारी|हमारा)?\s*(?:नौकरी|विद्यालय|स्कूल|कॉलेज|विश्वविद्यालय|अस्पताल|कंपनी)\s*(?:का\s+नाम)?\s*[:：]?\s*", text, re.I):
            left = text[field.start():field.end()]
            if not (re.search(r"नौकरी", left) or re.search(r"का\s+नाम", left)):
                continue
            tokens = _token_spans(text, field.end(), 5)
            if not tokens:
                continue
            for start, end in tokens:
                candidate = text[tokens[0][0]:end]
                if not _ORG_SUFFIX.search(candidate) or candidate.casefold() in _ORG_GENERIC:
                    continue
                after = text[end:]
                if not re.match(r"\s*(?:में|मे|है|लिख[ाीे]|दर्ज)(?=$|[\s।,;!?])", after):
                    continue
                results.append(RecognizerResult("organization", tokens[0][0], end, .91,
                    recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id,
                                          "source": "rule", "boundary_evidence": "personal_organization_field",
                                          "suffix_evidence": True}))
                break
        return results
