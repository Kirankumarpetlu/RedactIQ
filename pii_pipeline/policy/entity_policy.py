"""Shared validation, acceptance and deterministic overlap selection.

Every entry point uses the same applied spans. Original labels, recognizers and
source metadata survive normalization; rejected candidates carry safe reasons.
"""
from __future__ import annotations

from .. import json_codec as json
import math
import re
import unicodedata
from numbers import Integral
from typing import Any, Iterable, Mapping

from .indic_filters import (
    is_clinical_measurement,
    is_english_model_hallucination,
    is_suppressed,
    suppression_enabled,
)
from .context_filters import (
    contextual_rejection_reason,
    has_explicit_credential_context,
    has_explicit_identifier_context,
    has_person_identity_context,
    plausible_person_surface,
    has_clinical_speech_context,
)
from .semantic_sanity import (implausible_person, implausible_place, is_known_place,
                              normalize_place_span, unsupported_username,
                              unsupported_ambiguous_model_type,
                              unsupported_unlabelled_structured_type,
                              is_public_event_commentary)
from ..taxonomy import canonical_entity_type

_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)


def _name_word_char(char: str) -> bool:
    """Keep letters, combining marks and Indic joiners in one name word."""
    return unicodedata.category(char)[0] in {"L", "M"} or char in "\u200c\u200d"


def _expand_name_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend a detected name fragment to the edges of its containing words."""
    if _name_word_char(text[start]):
        while start > 0 and _name_word_char(text[start - 1]):
            start -= 1
    if _name_word_char(text[end - 1]):
        while end < len(text) and _name_word_char(text[end]):
            end += 1
    return start, end


def _inside_telugu_akshara(text: str, offset: int) -> bool:
    """Whether a code-point boundary cuts a Telugu combining or virama cluster.

    This local check avoids segmenting the entire input for each proposal. A
    vowel sign or joiner belongs to the preceding base; a consonant after a
    virama belongs to the same akshara.
    """
    if offset <= 0 or offset >= len(text):
        return False
    left, right = text[offset - 1], text[offset]
    telugu = lambda ch: "\u0c00" <= ch <= "\u0c7f"
    return bool(
        (telugu(right) and unicodedata.category(right).startswith("M") and telugu(left))
        or (right in "\u200c\u200d" and telugu(left))
        or (left in "\u0c4d\u200d" and telugu(right))
        or (left == "\u200c" and telugu(right) and unicodedata.category(right).startswith("M"))
    )


def _telugu_akshara_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while _inside_telugu_akshara(text, start):
        start -= 1
    while _inside_telugu_akshara(text, end):
        end += 1
    return start, end


def _signature(word):
    """Use matching positions in aligned Indic blocks; never guess a spelling.

    This intentionally handles only exact aligned-letter repetitions. Names
    with different phonetic spellings, Latin letters or mixed scripts stay
    untouched. Unicode normalization is for comparison, not source offsets.
    """
    letters, scripts, signature = 0, set(), []
    for char in unicodedata.normalize("NFKC", word):
        if unicodedata.category(char)[0] not in {"L", "M"}:
            return None
        block = next((base for base in _BLOCKS if base <= ord(char) < base+128), None)
        if block is None:
            return None
        scripts.add(block)
        letters += unicodedata.category(char).startswith("L")
        signature.append(ord(char)-block)
    if len(scripts) != 1 or letters < 2:
        return None
    return tuple(signature), next(iter(scripts))


def repeated_script_name_parts(surface: str) -> list[tuple[int, int]]:
    """Split two exact Indic-script renderings of the same one/two-word name.

    A different given name/surname across scripts is kept as a complete name.
    Every returned offset is relative to the original, unchanged surface.
    """
    words = list(re.finditer(r"\S+", surface))
    if len(words) not in {2, 4}:
        return []
    signatures = [_signature(word.group()) for word in words]
    if any(value is None for value in signatures):
        return []
    half = len(words)//2
    first, second = signatures[:half], signatures[half:]
    if (len({value[1] for value in first}) != 1 or len({value[1] for value in second}) != 1
            or first[0][1] == second[0][1]
            or [value[0] for value in first] != [value[0] for value in second]):
        return []
    return [(words[0].start(), words[half-1].end()), (words[half].start(), words[-1].end())]



def recognizer_name(result: Any) -> str:
    metadata = getattr(result, "recognition_metadata", None) or {}
    explanation = getattr(result, "analysis_explanation", None)
    return str(metadata.get("recognizer_name") or getattr(explanation, "recognizer", "") or "")


def entity_from_result(result: Any, source: str) -> dict[str, Any]:
    return {
        "entity_type": result.entity_type, "start": result.start, "end": result.end,
        "score": result.score, "source": source, "taxonomy_status": source,
        "recognizer": recognizer_name(result),
        "recognition_metadata": dict(getattr(result, "recognition_metadata", None) or {}),
    }


def parse_semantic_tags(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse imported tags without silently swallowing malformed data."""
    if raw is None or raw == "":
        return [], []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return [], [{"source": "gemma_tag", "reason": "invalid_tag_json"}]
    if not isinstance(raw, list):
        return [], [{"source": "gemma_tag", "reason": "tags_not_a_list"}]
    tags, issues = [], []
    for index, tag in enumerate(raw):
        if not isinstance(tag, Mapping):
            issues.append({"source": "gemma_tag", "index": index, "reason": "tag_not_an_object"})
            continue
        # Imported evidence cannot supply policy-owned provenance or ranking
        # metadata. Keep only the annotation fields in the external contract.
        candidate = {key: tag[key] for key in ("start", "end", "score", "text") if key in tag}
        candidate["entity_type"] = tag.get("tag") or tag.get("entity_type")
        candidate.setdefault("score", 0.85)
        candidate.update(source="gemma_tag", taxonomy_status="gemma_tag", recognizer="ImportedSemanticTag")
        tags.append(candidate)
    return tags, issues


def _evidence_priority(entity: Mapping[str, Any]) -> int:
    # Explicit validated rule provenance takes precedence over unsupported
    # neural boundaries. Scores within a tier remain ranking signals, not
    # interchangeable calibrated probabilities across detectors.
    name = str(entity.get("recognizer", ""))
    boundary = entity.get("recognition_metadata", {}).get("boundary_evidence")
    if entity.get("_policy_supported_containment"):
        return 3
    if name == "ConversationalPhoneRecognizer" and boundary == "explicit_contact_answer":
        return 3
    if name == "CredentialFieldRecognizer" and boundary == "explicit_credential":
        return 3
    if name == "LabelledCvvRecognizer":
        return 3
    if name == "LabelledHealthIdRecognizer" and boundary == "explicit_health_id":
        return 3
    if name == "LabelledPanRecognizer" and boundary == "explicit_tax_id":
        return 3
    if name == "LabelledIndianPinRecognizer" and boundary == "explicit_address":
        return 3
    if name == "DeciderAddressField" and boundary == "explicit_address_field":
        return 3
    if name == "HindiAddressRecognizer" and boundary in {
        "explicit_house_number", "explicit_address_room", "explicit_address_field", "explicit_street",
        "personal_residence", "family_residence", "personal_origin_or_home",
    }:
        return 3
    if name == "HindiOrganizationRecognizer" and boundary == "personal_organization_field":
        return 3
    if name in {'LabelledMrnRecognizer','UpiPinRecognizer'} or boundary in {'explicit_government_id','explicit_birth_date'}:
        return 3
    if name == "SpokenPlaceRecognizer" and boundary in {"personal_place_frame", "landmark_relation"}:
        return 2
    if name in {"IndicNameContextRecognizer", "TeluguNameCueRecognizer", "SpokenPlaceRecognizer"} and boundary in {
        "explicit_name", "identity_details", "identity_relation", "patient_introduction", "explicit_address",
        "personal_village_relation", "explicit_residence", "personal_residence", "personal_origin",
        "personal_hometown", "explicit_place_field", "romanized_personal_residence",
        "romanized_personal_hometown", "explicit_house_number", "personal_landmark_relation",
        "personal_destination",
    }:
        # A first-person cue alone is weaker than a validated place value.
        if name == "TeluguNameCueRecognizer" and boundary == "patient_introduction":
            return 1
        return 3
    if name == "ClinicalCalendarDateRecognizer" and boundary in {"clinical_onset_year", "clinical_month_day"}:
        return 3
    if name == "IndicPersonNameGazetteer":
        return 2
    if name == "UPIHandleRecognizer" and entity.get("score", 0) >= 0.9:
        # Context-enhanced whole-handle evidence beats a model selecting only
        # the account-name fragment of the same payment identifier.
        return 2
    if name in {"IndicNameContextRecognizer", "TeluguNameCueRecognizer", "SpokenPlaceRecognizer", "AgeRecognizer", "ShortDateRecognizer",
                "ClinicalCalendarDateRecognizer"}:
        return 1
    return 0


def candidate_script(surface: str) -> str:
    counts = {}
    blocks = ((0x0900,0x097F,"devanagari"),(0x0980,0x09FF,"bengali"),
              (0x0A00,0x0A7F,"gurmukhi"),(0x0A80,0x0AFF,"gujarati"),(0x0B00,0x0B7F,"odia"),
              (0x0B80,0x0BFF,"tamil"),(0x0C00,0x0C7F,"telugu"),
              (0x0C80,0x0CFF,"kannada"),(0x0D00,0x0D7F,"malayalam"))
    for char in surface:
        if not char.isalpha():
            continue
        script = next((name for low,high,name in blocks if low<=ord(char)<=high),"latin" if char.isascii() else "other")
        counts[script] = counts.get(script,0)+1
    return max(counts,key=counts.get) if counts else "numeric"


def _spacy_needs_independent_name_support(text, entity):
    if (not suppression_enabled() or "spacy" not in str(entity.get("recognizer", "")).casefold()
            or entity["canonical_entity_type"] != "PERSON"):
        return False
    if (entity.get("recognition_metadata") or {}).get("decider_verified") is True:
        return False
    if has_person_identity_context(text, entity):
        return False
    local = text[max(0, entity["start"]-100):entity["end"]+100]
    has_indic = any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0D7F for ch in local)
    # A public name in sports commentary is still a name. Mixed-language text
    # alone is not negative identity evidence for an otherwise Latin span.
    return has_indic and (any(ch.isalpha() and 0x0900 <= ord(ch) <= 0x0D7F for ch in entity["text"])
                          or has_clinical_speech_context(text, entity["start"], entity["end"]))


def acceptance_threshold(entity: Mapping[str, Any], surface: str, minimum: float, thresholds: Mapping[str,float] | None) -> float:
    """Optional reviewed source/type/script gates; empty mapping preserves defaults."""
    if not thresholds:
        return minimum
    if not isinstance(thresholds,Mapping) or any(not isinstance(k,str) or isinstance(v,bool) or
            not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for k,v in thresholds.items()):
        raise ValueError("Acceptance thresholds require string keys and finite scores between 0 and 1")
    label = canonical_entity_type(str(entity.get("entity_type",""))).casefold()
    recognizer = str(entity.get("recognizer","")).casefold()
    script = candidate_script(surface)
    normalized = {key.casefold():value for key,value in thresholds.items()}
    for key in (f"{recognizer}:{label}:{script}",f"{recognizer}:{label}",f"{label}:{script}",recognizer,label,"default"):
        if key in normalized:
            value = float(normalized[key])
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError("Acceptance thresholds must be finite values between 0 and 1")
            return max(minimum,value)
    return minimum


def resolve_entities(
    text: str, candidates: Iterable[Mapping[str, Any]], min_score: float = 0.0,
    *, score_thresholds: Mapping[str,float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return exactly the validated non-overlapping spans to apply.

    Prefer explicit boundary evidence, then confidence, then a shorter span.
    Filtering happens before overlap selection, including for imported tags.
    """
    if not math.isfinite(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be finite and between 0 and 1")
    valid, rejected = [], []
    public_commentary = is_public_event_commentary(text)
    for index, raw in enumerate(candidates):
        reason = None
        if not isinstance(raw, Mapping):
            rejected.append({"index": index, "reason": "entity_not_an_object"})
            continue
        entity = dict(raw)
        entity.pop("_policy_supported_containment",None)
        label, start, end = entity.get("entity_type"), entity.get("start"), entity.get("end")
        if not isinstance(label, str) or not label.strip() or label.strip().upper() == "UNKNOWN":
            reason = "invalid_entity_type"
        elif any(isinstance(offset, bool) or not isinstance(offset, Integral) for offset in (start, end)):
            reason = "invalid_offsets"
        elif not 0 <= start < end <= len(text):
            reason = "span_out_of_bounds"
        try:
            if isinstance(entity.get("score"), bool):
                raise ValueError
            score = float(entity.get("score", 0.0))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError
        except (TypeError, ValueError):
            score = 0.0
            reason = reason or "invalid_score"
        if reason is None:
            surface = text[start:end]
            explicit_address_value = (canonical_entity_type(label) == "LOCATION"
                and (entity.get("recognition_metadata") or {}).get("boundary_evidence") in
                {"explicit_address", "explicit_house_number", "explicit_address_room", "explicit_address_field", "explicit_street"})
            if "text" in entity and entity["text"] != surface:
                reason = "text_offset_mismatch"
            elif canonical_entity_type(label) == "LOCATION" and not explicit_address_value:
                new_start, new_end = normalize_place_span(text, start, end)
                if new_start < new_end and (new_start, new_end) != (start, end):
                    entity["recognition_metadata"] = {**(entity.get("recognition_metadata") or {}),
                        "normalized_place_from": [int(start), int(end)]}
                    start, end = new_start, new_end
                    surface = text[start:end]
                    entity.update(start=start, end=end, text=surface)
            elif (canonical_entity_type(label) == "PERSON" and entity.get("source") != "gemma_tag"
                  and entity.get("recognizer") != "TeluguNameCueRecognizer"
                  # The gazetteer has already separated a listed Tamil name
                  # from an attached case suffix. Expanding to the whole token
                  # would undo that validated boundary and reduce exact recall.
                  and not (entity.get("recognizer") == "IndicPersonNameGazetteer"
                           and (entity.get("recognition_metadata") or {}).get("boundary_evidence") == "listed_tamil_case_root")
                  and not (entity.get("recognizer") == "IndicNameContextRecognizer"
                           and (entity.get("recognition_metadata") or {}).get("suffix_separated"))):
                expanded_start, expanded_end = _expand_name_word_boundaries(text, start, end)
                if (expanded_start, expanded_end) != (start, end):
                    metadata = entity.get("recognition_metadata")
                    entity["recognition_metadata"] = {
                        **(metadata if isinstance(metadata, Mapping) else {}),
                        "expanded_name_word_from": [int(start), int(end)],
                    }
                    start, end = expanded_start, expanded_end
                    surface = text[start:end]
                    entity.update(start=start, end=end, text=surface)
            if reason is None:
                if canonical_entity_type(label) == "PERSON" and entity.get("source") != "gemma_tag":
                    # Courtesy titles close a spoken name but are not part of
                    # its identifier. Trim before sanity checks so "X ma'am"
                    # protects X while a standalone honorific is still rejected.
                    title = re.search(r"\s+(?:sir|ma['’]am|గారు|గారూ)\s*$", surface, re.I)
                    if title and title.start() > 0:
                        old_end = end
                        end = start + title.start()
                        surface = text[start:end]
                        entity.update(end=end, text=surface)
                        entity["recognition_metadata"] = {**(entity.get("recognition_metadata") or {}),
                            "trimmed_person_honorific_from": [int(start), int(old_end)]}
                # A Telugu locative is a detachable grammatical ending even
                # when the place root ends in virama (హైదరాబాద్ + లో).
                place_suffix = (canonical_entity_type(label) == "LOCATION" and
                    text[end:end + 1] in "\u200c\u200dల" and
                    bool(re.match(r"[\u200c\u200d]?లో(?:$|[\s.,!?;])", text[end:])))
                safe_start, safe_end = (start, end) if place_suffix else _telugu_akshara_bounds(text, start, end)
                if (safe_start, safe_end) != (start, end):
                    if entity.get("source") == "gemma_tag":
                        reason = "akshara_split"
                    else:
                        entity["recognition_metadata"] = {
                            **(entity.get("recognition_metadata") or {}),
                            "akshara_expanded_from": [int(start), int(end)],
                        }
                        start, end = safe_start, safe_end
                        surface = text[start:end]
                        entity.update(start=start, end=end, text=surface)
            if reason is None and score < acceptance_threshold(entity,surface,min_score,score_thresholds):
                reason = "below_min_score"
            elif reason is None and canonical_entity_type(label) == "PERSON" and (
                    entity.get("recognizer") in {"TeluguNameCueRecognizer", "IndicNameContextRecognizer",
                                                 "GLiNER2PIIPrimary", "pplx_decider", "pplx_decider_indic"}
                    and implausible_person(surface)):
                reason = "implausible_person_value"
            elif (reason is None and canonical_entity_type(label) == "PERSON"
                  and entity.get("recognizer") == "TeluguNameCueRecognizer"
                  and (entity.get("recognition_metadata") or {}).get("boundary_evidence") != "explicit_name"
                  and is_known_place(surface)):
                reason = "known_place_not_cue_name"
            elif reason is None and canonical_entity_type(label) == "LOCATION" and not explicit_address_value and implausible_place(surface):
                reason = "implausible_place_value"
            elif reason is None and is_english_model_hallucination(str(entity.get("recognizer", "")), label, surface) and not (
                    canonical_entity_type(label) == "PERSON" and has_person_identity_context(text, entity)):
                reason = "unsupported_spacy_script"
            elif reason is None and is_suppressed(label, surface) and not (canonical_entity_type(label)=="PERSON" and has_person_identity_context(text,entity)):
                reason = "known_non_name"
            elif (reason is None and canonical_entity_type(label) == "PERSON"
                  and entity.get("recognizer") in {"GLiNER2PIIPrimary", "IndicPersonNameGazetteer"}
                  and any("\u0900" <= char <= "\u097f" for char in surface)
                  and not has_person_identity_context(text, entity)):
                reason = "hindi_name_without_identity_evidence"
            elif reason is None and (context_reason := contextual_rejection_reason(text, entity)):
                reason = context_reason
            elif reason is None and label.casefold() == "username" and entity.get("recognizer") == "GLiNER2PIIPrimary" and unsupported_username(text, start, end):
                reason = "unsupported_username_syntax"
            elif (reason is None and entity.get("recognizer") == "GLiNER2PIIPrimary"
                  and unsupported_ambiguous_model_type(text, start, end, canonical_entity_type(label))):
                reason = "unsupported_ambiguous_model_type"
            elif (reason is None and unsupported_unlabelled_structured_type(
                    text, start, end, str(entity.get("recognizer", "")))):
                reason = "unsupported_unlabelled_structured_type"
            elif reason is None and is_clinical_measurement(label, text, start, end) and not (has_explicit_credential_context(text, entity) or has_explicit_identifier_context(text,entity)):
                reason = "clinical_measurement"
            if reason is None and public_commentary and canonical_entity_type(label) in {"PERSON", "LOCATION"}:
                boundary = (entity.get("recognition_metadata") or {}).get("boundary_evidence")
                personal_place = boundary in {"personal_residence", "personal_origin", "personal_hometown",
                                              "address_answer", "explicit_place_field", "explicit_address"}
                private_identity = (canonical_entity_type(label) == "PERSON"
                                    and has_person_identity_context(text, entity))
                if not private_identity and not (canonical_entity_type(label) == "LOCATION"
                                                 and (explicit_address_value or personal_place)):
                    reason = "public_commentary_not_private_identifier"
        if reason:
            rejected.append({"index": index, "source": entity.get("source", "pipeline"), "reason": reason,
                             "recognizer": str(entity.get("recognizer","")), "entity_type": label,
                             "start": int(start) if isinstance(start,Integral) and not isinstance(start,bool) else None,
                             "end": int(end) if isinstance(end,Integral) and not isinstance(end,bool) else None,
                             "score": score,
                             "text": text[start:end] if isinstance(start,Integral) and isinstance(end,Integral) and 0<=start<end<=len(text) else None})
            continue
        entity.update(entity_type=label.strip(), original_entity_type=entity.get("original_entity_type", label.strip()),
                      canonical_entity_type=canonical_entity_type(label), start=int(start), end=int(end), score=score,
                      text=text[start:end])
        entity.setdefault("source", entity.get("taxonomy_status", "pipeline"))
        entity.setdefault("taxonomy_status", entity["source"])
        if not isinstance(entity.get("recognition_metadata", {}), Mapping):
            entity["recognition_metadata"] = {}
        if "supporting_evidence" in entity:
            entity["supporting_evidence"] = list(entity["supporting_evidence"]) if isinstance(entity["supporting_evidence"], list) else []
        parts = repeated_script_name_parts(entity["text"]) if entity["canonical_entity_type"] == "PERSON" else []
        if not parts:
            valid.append(entity)
            continue
        # Apply the same acceptance gates to each original-text occurrence.
        # This also prevents the full-name containment step from joining the
        # two renderings back together. Detector provenance is preserved.
        for left, right in parts:
            part = {**entity, "start": entity["start"]+left, "end": entity["start"]+right,
                    "text": entity["text"][left:right],
                    "recognition_metadata": {**entity.get("recognition_metadata", {}),
                        "split_repeated_name_from": [entity["start"], entity["end"]]}}
            part_reason = None
            if score < acceptance_threshold(part, part["text"], min_score, score_thresholds):
                part_reason = "below_min_score"
            elif is_suppressed(label, part["text"]) and not has_person_identity_context(text, part):
                part_reason = "known_non_name"
            else:
                part_reason = contextual_rejection_reason(text, part)
            if part_reason:
                rejected.append({"index": index, "source": part["source"], "reason": part_reason,
                                 "recognizer": part.get("recognizer", ""), "entity_type": label,
                                 "start": part["start"], "end": part["end"], "score": score, "text": part["text"]})
            else:
                valid.append(part)
    # English-only spaCy can emit Latin or mixed-script clinical clauses at its
    # fixed 0.85 score. Before overlap selection, require identity context or an
    # independently validated same-boundary name proposal. Do not let a raw,
    # already-rejected model proposal validate the spaCy candidate.
    unsupported = []
    for entity in valid:
        if not _spacy_needs_independent_name_support(text, entity):
            continue
        # A name repeated just after its validated introduction remains an
        # identity even without repeating "my name is". Use exact text after
        # whitespace/case comparison; never propagate to a different word.
        repeated_identity = max((other for other in valid if other is not entity
            and other["canonical_entity_type"] == "PERSON"
            and 0 <= entity["start"]-other["end"] <= 80
            and " ".join(other["text"].casefold().split()) == " ".join(entity["text"].casefold().split())
            and has_person_identity_context(text, other)),
            key=lambda other: (other["end"], other["start"]), default=None)
        if repeated_identity is not None:
            entity["recognition_metadata"] = {**entity.get("recognition_metadata", {}),
                "supported_name_repetition": {"start": repeated_identity["start"], "end": repeated_identity["end"]}}
            continue
        words = entity["text"].split()
        complete_latin_name = (len(words) >= 2 and candidate_script(entity["text"]) == "latin"
            and plausible_person_surface(entity["text"])
            and sum(word[0].isupper() for word in words) >= 2
            and all(word[0].isupper() or word.casefold() in {"de", "del", "van", "von", "da"}
                    for word in words))
        corroborated = any(other is not entity and other["canonical_entity_type"] == "PERSON"
            and "spacy" not in str(other.get("recognizer", "")).casefold()
            and other.get("recognizer") in {"GLiNER2PIIPrimary", "IndicNER", "IndicNameContextRecognizer",
                                             "pplx_decider", "pplx_decider_indic"}
            and ((other["start"], other["end"]) == (entity["start"], entity["end"])
                 or complete_latin_name and entity["start"] <= other["start"] < other["end"] <= entity["end"])
            for other in valid)
        if not corroborated:
            unsupported.append(entity)
            rejected.append({"source": entity["source"], "recognizer": entity.get("recognizer", ""),
                "entity_type": entity["entity_type"], "reason": "unsupported_spacy_mixed_context",
                "start": entity["start"], "end": entity["end"], "score": entity["score"],
                "text": entity["text"]})
    unsupported_ids = {id(entity) for entity in unsupported}
    valid = [entity for entity in valid if id(entity) not in unsupported_ids]
    # A supported surname/given-name fragment must not destroy a plausible
    # complete name. Promote validated containment; never union arbitrary
    # overlaps or a clinical clause such as "Ananya feels well".
    for full in valid:
        if full["canonical_entity_type"]!="PERSON" or len(full["text"].split())<2:
            continue
        if not plausible_person_surface(full["text"]):
            continue
        fragments = [fragment for fragment in valid if fragment is not full and
                     fragment["canonical_entity_type"]=="PERSON" and
                     full["start"]<=fragment["start"]<fragment["end"]<=full["end"] and
                     (full["start"],full["end"])!=(fragment["start"],fragment["end"]) and
                     fragment.get("recognizer") in {"IndicPersonNameGazetteer","IndicNameContextRecognizer"}]
        if fragments:
            full["_policy_supported_containment"] = True
            full["recognition_metadata"] = {**full.get("recognition_metadata",{}),
                "containment_evidence": "supported_full_name"}
            full.setdefault("supporting_evidence",[]).extend(
                {"source":f["source"],"recognizer":f.get("recognizer",""),"entity_type":f["entity_type"],
                 "score":f["score"],"start":f["start"],"end":f["end"],"evidence_kind":"contained_name_fragment"}
                for f in sorted(fragments,key=lambda f:(f["start"],f["end"],f.get("recognizer",""),f["entity_type"])))
    ordered = sorted(valid, key=lambda e: (
        -_evidence_priority(e), -e["score"], e["end"] - e["start"],
        e["start"], e["end"], e["canonical_entity_type"], e["entity_type"],
        str(e.get("recognizer", "")), str(e["source"]),
    ))
    accepted = []
    for entity in ordered:
        duplicates = [other for other in accepted if (entity["start"], entity["end"], entity["canonical_entity_type"]) ==
                      (other["start"], other["end"], other["canonical_entity_type"])]
        if duplicates:
            other = duplicates[0]
            evidence = other.setdefault("supporting_evidence", [])
            evidence.append({"source": entity["source"], "recognizer": entity.get("recognizer", ""),
                             "entity_type": entity["entity_type"], "score": entity["score"],
                             "recognition_metadata": entity.get("recognition_metadata", {})})
        elif any(entity["start"] < other["end"] and other["start"] < entity["end"] for other in accepted):
            rejected.append({"source": entity["source"], "recognizer": entity.get("recognizer", ""),
                             "entity_type": entity["entity_type"], "reason": "overlapping_candidate",
                             "start": entity["start"], "end": entity["end"]})
        else:
            accepted.append(entity)
    for entity in accepted:
        entity.pop("_policy_supported_containment",None)
    return sorted(accepted, key=lambda e: (e["start"], e["end"])), rejected
