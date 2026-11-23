"""Evidence-based postprocessing of raw Devanagari IndicNER proposals.

Every output is descended from a model proposal. This module never manufactures
an entity from a cue. Offsets always refer to the original Python string.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


DEVANAGARI = re.compile(r"[\u0900-\u097f]")
NAME_LEFT = re.compile(r"(?:नाम\s*[:：]?|मेरा\s+नाम|(?:मेरे|मेरी|मेरा)\s+(?:पिता|माता|पति|पत्नी|बेटा|बेटी|भाई|बहन)|(?:पिता|माता|पति)\s+का\s+नाम|(?:पुत्र|पुत्री|सुपुत्र|सुपुत्री|पत्नी|धर्मपत्नी)\s+(?:श्री\s*)?|(?:श्री|श्रीमती|सुश्री|कुमारी|डॉ\.?|डॉक्टर|प्र\.?)\s+|मैं\s+)$")
PLACE_LEFT = re.compile(r"(?:\bपता\s*[:：]?|(?:ग्राम|गाँव|गांव|शहर|जिला|ज़िला|तहसील|मोहल्ला|कॉलोनी|सेक्टर|गली|रोड|मार्ग)\s+)$")
PLACE_RIGHT = re.compile(r"^\s*(?:में|से|का|की|के|पर|के\s+पास)\s*(?:रहत[ाीे]|आय[ाीे]|काम|नौकरी|पढ़त[ाीे]|हूँ|हूं|है|हैं|था|थी|है\b)")
ORG_RIGHT = re.compile(r"^\s*(?:में|से)\s*(?:काम|नौकरी|भर्ती|इलाज|पढ़त[ाीे]|पढत[ाीे]|भेजा|उपचार)")
ORG_LEFT = re.compile(r"(?:मेरी|मेरा|मेरे|मुझे|मैं|मै|हमारा|रोगी|मरीज़|मरीज|बच्चा|बच्ची)\s+(?:नौकरी|कंपनी|बीमा|इलाज|भर्ती|पढ़ाई|शिक्षा|उपचार)\s*$")
ORG_SUFFIX = re.compile(r"(?:अस्पताल|हॉस्पिटल|क्लिनिक|मेडिकल\s+कॉलेज|कॉलेज|विश्वविद्यालय|विद्यालय|संस्थान|बैंक|लिमिटेड|प्राइवेट\s+लिमिटेड)$")
ORG_SUFFIX_AFTER = re.compile(r"^\s+(?:अस्पताल|हॉस्पिटल|क्लिनिक|मेडिकल\s+कॉलेज|कॉलेज|विश्वविद्यालय|विद्यालय|संस्थान|बैंक|लिमिटेड)\b")
GENERIC_ORG = frozenset({"अस्पताल", "हॉस्पिटल", "क्लिनिक", "कॉलेज", "विद्यालय", "विश्वविद्यालय", "संस्थान", "बैंक", "कंपनी"})
POSTPOSITION = re.compile(r"(?:[-\s]+)(?:जी|साहब|साहिबा|भाई|बहन|दीदी|में|का|की|के|को|से|पर|ने|तक)$")
LEADING_HONORIFIC = re.compile(r"^(?:श्री|श्रीमती|सुश्री|कुमारी|डॉ\.?|डॉक्टर|प्रो\.?)\s+")
TRAILING_PUNCT = re.compile(r"[\s,;:!?।॥|.]+$")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(text))


def _normalized(text: str) -> str:
    # This is the same reversible matching view used by the Hindi recognizers.
    # Its offsets are deliberately not used to return model spans: the model
    # spans already refer to original text after chunk offset mapping.
    try:
        from ..hindi_normalization import normalize_with_offsets
    except ImportError:
        return text
    return normalize_with_offsets(text).text


def _boundary(text: str, pos: int) -> bool:
    if pos <= 0 or pos >= len(text):
        return True
    after = text[pos]
    before = text[pos - 1]
    return not (unicodedata.category(after).startswith("M") or after in "\u200c\u200d" or before in "\u094d\u200c\u200d")


def _action(entity: dict[str, Any], name: str) -> None:
    entity["enrichment_actions"] = list(dict.fromkeys([*(entity.get("enrichment_actions") or []), name]))


def repair_boundary(text: str, source: Mapping[str, Any]) -> dict[str, Any]:
    entity = dict(source)
    start, end = entity["start"], entity["end"]
    label = entity["entity_type"].upper()
    if not (0 <= start < end <= len(text)):
        return entity
    while start > 0 and not _boundary(text, start):
        start -= 1
    while end < len(text) and not _boundary(text, end):
        end += 1
    surface = text[start:end]
    if label == "PERSON":
        prefix = LEADING_HONORIFIC.match(surface)
        if prefix:
            start += prefix.end()
            surface = text[start:end]
    trimmed = TRAILING_PUNCT.sub("", surface)
    if label in {"PERSON", "LOCATION"}:
        trimmed = POSTPOSITION.sub("", trimmed)
    end = start + len(trimmed)
    if start < end and _boundary(text, start) and _boundary(text, end):
        if (start, end) != (entity["start"], entity["end"]):
            entity["start"], entity["end"], entity["text"] = start, end, text[start:end]
            _action(entity, "boundary_repaired")
    return entity


def enrich_candidates(text: str, raw: list[dict[str, Any]], *, options: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return enriched proposals and explicit enrichment rejections.

    The strict non-cue gate remains the pipeline's existing min_confidence.
    Cue thresholds are independently configurable and apply only after evidence.
    """
    if not has_devanagari(text) or not (options or {}).get("enabled", True):
        return raw, []
    opts = options or {}
    threshold = opts.get("confidence_thresholds") or {}
    output: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for original in raw:
        if not has_devanagari(str(original.get("text", ""))):
            output.append(original)
            continue
        entity = dict(original)
        metadata = dict(entity.get("recognition_metadata") or {})
        metadata["raw_indic_score"] = original.get("score")
        metadata["raw_acceptance_threshold"] = metadata.get("acceptance_threshold")
        entity["recognition_metadata"] = metadata
        if opts.get("boundary_repair", True):
            entity = repair_boundary(text, entity)
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= len(text):
            continue
        label = entity["entity_type"].upper()
        left = text[max(0, start - 48):start]
        right = text[end:end + 48]
        raw_right = right
        surface = text[start:end]
        if opts.get("normalization", True):
            left, right, surface = map(_normalized, (left, right, surface))
        if opts.get("type_reconciliation", True):
            if label in {"PERSON", "LOCATION"} and (ORG_SUFFIX_AFTER.match(right) or ORG_SUFFIX.search(surface)):
                label = "ORGANIZATION"
                _action(entity, "type_reconciled")
            elif label == "PERSON" and PLACE_RIGHT.match(right) and not NAME_LEFT.search(left):
                label = "LOCATION"
                _action(entity, "type_reconciled")
            elif label == "LOCATION" and NAME_LEFT.search(left):
                label = "PERSON"
                _action(entity, "type_reconciled")
        if label == "ORGANIZATION" and opts.get("organization_suffix", True):
            suffix = ORG_SUFFIX_AFTER.match(raw_right)
            if suffix and _boundary(text, end + suffix.end()):
                entity["end"] = end + suffix.end()
                entity["text"] = text[start:entity["end"]]
                end = entity["end"]
                surface = entity["text"]
                right = text[end:end + 48]
                _action(entity, "boundary_repaired")
        cue = False
        if label == "PERSON":
            cue = bool(NAME_LEFT.search(left) or NAME_LEFT.search(left.rstrip()))
            if cue and re.search(r"मैं\s+$", left):
                cue = bool(re.match(r"^\s*(?:हूँ|हूं|हैं|है|,\s*\d|और\s*\d)", right))
        elif label == "LOCATION":
            cue = bool(PLACE_LEFT.search(left) or PLACE_RIGHT.match(right))
        elif label == "ORGANIZATION":
            cue = bool((ORG_RIGHT.match(right) or ORG_LEFT.search(left))
                       and surface.strip() not in GENERIC_ORG)
            if not cue:
                entity["enrichment_rejection_reason"] = "organization_without_personal_evidence"
                rejected.append(entity)
                continue
        entity["entity_type"] = label
        metadata = dict(entity.get("recognition_metadata") or {})
        metadata["hindi_cue_evidence"] = cue
        if cue and opts.get("confidence_recalibration", True):
            floor = float(threshold.get(f"{label.lower()}:cue", {"PERSON": .45, "LOCATION": .55, "ORGANIZATION": .55}.get(label, .95)))
            metadata["enrichment_threshold"] = floor
            metadata["acceptance_threshold"] = floor
            _action(entity, "cue_threshold")
        metadata["enrichment_actions"] = list(entity.get("enrichment_actions") or [])
        metadata["source"] = entity.get("source", "indic_ner")
        entity["recognition_metadata"] = metadata
        output.append(entity)
    return output, rejected
