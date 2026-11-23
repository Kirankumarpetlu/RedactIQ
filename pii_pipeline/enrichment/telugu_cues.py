"""Telugu cue evidence for existing IndicNER PERSON and LOCATION proposals.

This module never creates a detection.  It attaches contextual evidence to raw
IndicNER proposals after their offsets have been mapped to the original text.
Thresholds are provisional until a reviewed Telugu development split exists.
The caller owns the final confidence gate and the shared entity policy.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


_TELUGU = re.compile(r"[\u0c00-\u0c7f]")
_NAME_BEFORE = (
    ("explicit_name", re.compile(r"(?:నా|మా|మీ|తండ్రి|తల్లి|భర్త|భార్య|రోగి)\s+పేరు\s*[:：=-]?\s*$|పేరు\s*[:：=-]\s*$")),
    ("person_honorific", re.compile(r"(?:శ్రీమతి|శ్రీ|డాక్టర్|డా\.)\s+$")),
    ("personal_introduction", re.compile(r"(?:నేను|నన్ను)\s+$")),
    ("identity_relation", re.compile(r"(?:మా|నా)\s+(?:అమ్మ|నాన్న|అబ్బాయి|అమ్మాయి|కొడుకు|కూతురు|భర్త|భార్య)\s+$")),
)
_NAME_AFTER = re.compile(
    r"^\s*(?:[\u200c\u200d]?(?:ని|ను|కి|కు|తో))?\s*"
    r"(?:(?:అంటారు|మాట్లాడుతున్నాను|అని|గారు)(?:\s|[.,;!?]|$)|[.,;!?]|$)"
)
_PLACE_BEFORE = (
    ("explicit_address", re.compile(r"(?:చిరునామా|అడ్రస్|ఇంటి\s+నెం\.?|రోడ్\s+నెం\.?)\s*[:：=-]?\s*$")),
    ("administrative_place", re.compile(r"(?:గ్రామం|మండలం|జిల్లా|నగరం|కాలనీ)\s*[:：=-]?\s*$")),
    ("personal_hometown", re.compile(r"మాది\s+$")),
    ("origin_or_residence", re.compile(r"నేను\s+$")),
)
_PLACE_AFTER = re.compile(
    r"^\s*(?:[\u200c\u200d]?(?:లోని|లో|నుండి|నుంచి|కు|కి)|\s+(?:లో|నుండి|నుంచి))?"
    r"\s*(?:ఉంటాను|ఉంటున్నాను|వచ్చాను|వస్తున్నాను|దగ్గర|ఎదురుగా)(?:\s|[.,;!?]|$)"
)
_ADDRESS_COMPOUND_AFTER = re.compile(r"^\s*(?:నగర్|నగరం|కాలనీ|వీధి|రోడ్|గ్రామం)(?:\s|[.,;!?]|$)")

# These are grammatical/clinical categories, not a closed list of names or
# places.  Contextual cues still cannot make a pronoun or a generic location PII.
_NON_VALUES = frozenset({
    "నేను", "నా", "నాకు", "నన్ను", "మేము", "మా", "మీ", "పేరు",
    "డాక్టర్", "వైద్యుడు", "రోగి", "నర్సు", "అమ్మ", "నాన్న",
    "ఇల్లు", "ఇంట్లో", "ఆసుపత్రి", "ఆసుపత్రిలో", "గది", "గ్రామం",
    "మండలం", "జిల్లా", "నగరం", "కాలనీ", "బాగున్నాను", "జ్వరం",
})
_PRONOUNS = frozenset({"నేను", "నా", "నాకు", "నన్ను", "మేము", "మా", "మీ"})
_GENERIC_PLACES = frozenset({"ఇల్లు", "ఇంట్లో", "ఆసుపత్రి", "ఆసుపత్రిలో", "గది", "గ్రామం", "మండలం", "జిల్లా", "నగరం", "కాలనీ"})
_TYPE_KEYS = {"PERSON": "person", "LOCATION": "location"}
DEFAULT_CUE_THRESHOLDS: dict[str, float] = {}


def has_telugu(text: str) -> bool:
    return bool(_TELUGU.search(text))


def cue_evidence(text: str, start: int, end: int, *,
                 normalized: Any | None = None) -> dict[str, str | None]:
    """Return bounded cue classes adjacent to an existing proposal.

    The first-person pronoun is weak evidence: a trailing identity or residence
    predicate must also fit.  Explicit field and honorific cues stand alone.
    """
    if not 0 <= start < end <= len(text) or not has_telugu(text):
        return {"name": None, "place": None}
    view = text
    if normalized is not None:
        # NormalizedText.origins holds an original half-open interval for each
        # normalized code point.  Locate the corresponding proposal interval;
        # no normalized offset is ever written back to the entity.
        covered = [i for i, (origin_start, origin_end) in enumerate(normalized.origins)
                   if origin_start < end and origin_end > start]
        if not covered:
            return {"name": None, "place": None}
        start, end = covered[0], covered[-1] + 1
        view = normalized.text
    before = view[max(0, start - 70):start]
    after = view[end:min(len(view), end + 40)]
    name = next((kind for kind, pattern in _NAME_BEFORE if pattern.search(before)), None)
    place = next((kind for kind, pattern in _PLACE_BEFORE if pattern.search(before)), None)
    if name == "personal_introduction" and not _NAME_AFTER.match(after):
        name = None
    if place == "origin_or_residence" and not _PLACE_AFTER.match(after):
        place = None
    if _ADDRESS_COMPOUND_AFTER.match(after):
        place = "address_compound"
    elif _PLACE_AFTER.match(after):
        place = place or "origin_or_residence"
    return {"name": name, "place": place}


def proposed_threshold(entity_type: str, evidence: Mapping[str, str | None],
                       *, options: Mapping[str, Any] | None = None,
                       baseline: float = 0.95) -> float:
    """Suggest a cue-specific gate; no-cue proposals keep the caller's gate.

    The threshold must be confirmed on reviewed development data before it is
    described as calibrated.  An unrelated cue never lowers a type's gate.
    """
    key = _TYPE_KEYS.get(entity_type.upper())
    if key is None or not evidence.get("name" if key == "person" else "place"):
        return baseline
    options = options or {}
    thresholds = options.get("cue_thresholds", DEFAULT_CUE_THRESHOLDS)
    proposed = float(thresholds.get(key, baseline))
    return max(0.0, min(1.0, proposed))


def enrich_cues(text: str, entities: Sequence[Mapping[str, Any]], *,
                options: Mapping[str, Any] | None = None,
                normalized: Any | None = None) -> list[dict[str, Any]]:
    """Annotate raw IndicNER proposals without changing source or raw score.

    Every input proposal is returned, including proposed rejections, so the
    caller can preserve them in decision_trace before applying its final gate.
    `enrichment_rejection_reason` is advisory until the caller acts on it.
    """
    if not has_telugu(text):
        return [dict(entity) for entity in entities]
    options = options or {}
    baseline = float(options.get("baseline_threshold", 0.95))
    enable_cues = bool(options.get("cue_evidence", True))
    enable_types = bool(options.get("type_reconciliation", True))
    enable_common = bool(options.get("common_noun_rejection", True))
    enable_gate = bool(options.get("cue_thresholds_enabled", True))
    score_lift = float(options.get("cue_score_lift", 0.0))
    out: list[dict[str, Any]] = []
    for original in entities:
        entity = dict(original)
        start, end = entity.get("start"), entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            out.append(entity)
            continue
        surface = text[start:end]
        label = str(entity.get("entity_type", "")).upper()
        if label not in _TYPE_KEYS:
            out.append(entity)
            continue
        evidence = cue_evidence(text, start, end, normalized=normalized) if enable_cues else {"name": None, "place": None}
        actions = list(entity.get("enrichment_actions", ()))
        metadata = dict(entity.get("recognition_metadata") or {})
        metadata.setdefault("raw_indic_score", entity.get("score"))
        metadata.setdefault("raw_indic_type", label)
        if evidence["name"] or evidence["place"]:
            metadata["telugu_cue_evidence"] = evidence
            actions.append("cue_evidence")
        if enable_types:
            # An explicit address frame or a following street/locality marker
            # outweighs the model's person label.  Reverse reconciliation is
            # limited to explicit name/honorific frames.
            if label == "PERSON" and evidence["place"] in {
                "explicit_address", "administrative_place", "address_compound", "origin_or_residence"
            } and not evidence["name"]:
                label = "LOCATION"
                actions.append("type_reconciled")
            elif label == "LOCATION" and evidence["name"] in {"explicit_name", "person_honorific"} and not evidence["place"]:
                label = "PERSON"
                actions.append("type_reconciled")
        entity["entity_type"] = label
        non_value = surface.strip().casefold()
        contextual_non_value = (
            non_value in _PRONOUNS
            or label == "LOCATION" and non_value in _GENERIC_PLACES
            or non_value in _NON_VALUES and not (
                label == "PERSON" and evidence["name"] in {"explicit_name", "person_honorific"}
                or label == "LOCATION" and evidence["place"] in {"explicit_address", "administrative_place"}
            )
        )
        if enable_common and contextual_non_value:
            entity["enrichment_rejection_reason"] = "telugu_non_value_in_context"
            actions.append("non_value_rejected")
        if enable_gate:
            threshold = proposed_threshold(label, evidence, options=options, baseline=baseline)
            metadata["enrichment_threshold"] = threshold
            if threshold < baseline:
                actions.append("cue_threshold")
        if score_lift > 0 and "cue_boost" not in actions and evidence["name" if label == "PERSON" else "place"]:
            entity["score"] = min(1.0, float(entity["score"]) + score_lift)
            actions.append("cue_boost")
        entity["recognition_metadata"] = metadata
        entity["enrichment_actions"] = list(dict.fromkeys(actions))
        out.append(entity)
    return out
