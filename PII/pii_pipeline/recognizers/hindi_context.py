"""Hindi cue enhancement for validated, rule-produced Presidio candidates.

English spaCy lemmas do not recover Hindi cue words. This enhancer only adds
bounded contextual support to dedicated Hindi recognizers; it cannot turn a
generic Presidio number, date, or spaCy guess into PII.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right

from ..hindi_normalization import normalize_with_offsets


_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_CUES = {
    "PERSON": re.compile(r"(?:नाम|पिता|माता|पति|पत्नी|पुत्री?|सुपुत्री?|श्री|श्रीमती|सुश्री|कुमारी|डॉक्टर|डॉ)", re.I),
    "AGE": re.compile(r"(?:उम्र|आयु|वर्षीय|साल|वर्ष|बरस|महीन|माह|दिन)", re.I),
    "address": re.compile(r"(?:पता|मकान|घर|फ्लैट|प्लॉट|गली|मोहल्ला|नगर|कॉलोनी|सेक्टर|वार्ड|रोड|मार्ग|गाव|गांव|जिला|तहसील|पिन|शहर|रहत|रहती|रहते)", re.I),
    "organization": re.compile(r"(?:कंपनी|काम|नौकरी|भर्ती|इलाज|पढ|अस्पताल|हॉस्पिटल|क्लिनिक|कॉलेज|विश्वविद्यालय|विद्यालय|संस्थान|बैंक|लिमिटेड)", re.I),
}
_LABELS = {
    "PERSON": "PERSON", "NAME": "PERSON", "person": "PERSON",
    "AGE": "AGE", "age": "AGE",
    "address": "address", "ADDRESS": "address", "LOCATION": "address",
    "organization": "organization", "ORGANIZATION": "organization", "ORG": "organization",
}


def enhance_hindi_results(text: str, results: list, *, boost: float = 0.12,
                          minimum_score: float = 0.6, window: int = 48) -> list:
    """Add a capped score lift for a nearby Hindi cue in the same clause.

    These defaults are provisional and require synthetic dev/held-out
    calibration before production use. The recognizer must already have
    validated and emitted a candidate.
    Original offsets and surfaces are never altered. No-op on non-Devanagari.
    """
    if not _DEVANAGARI.search(text):
        return results
    normalized = normalize_with_offsets(text)
    for result in results:
        metadata = result.recognition_metadata or {}
        name = str(metadata.get("recognizer_name", ""))
        label = _LABELS.get(result.entity_type)
        if not name.startswith("Hindi") or label is None or not 0 <= result.start < result.end <= len(text):
            continue
        cue = _CUES[label]
        # Map the original interval to normalized indices. Folds can change
        # string length, so original offsets cannot index normalized text.
        before = bisect_right(normalized.ends, result.start)
        after = bisect_left(normalized.starts, result.end)
        left = normalized.text[max(0, before - window):before]
        right = normalized.text[after:min(len(normalized.text), after + window)]
        left = re.split(r"[।|.!?\n]", left)[-1]
        right = re.split(r"[।|.!?\n]", right)[0]
        if not (cue.search(left) or cue.search(right)):
            continue
        result.score = min(1.0, max(result.score + boost, minimum_score))
        metadata = dict(metadata)
        metadata["hindi_context_cue"] = True
        metadata["hindi_context_boost"] = boost
        result.recognition_metadata = metadata
    return results
