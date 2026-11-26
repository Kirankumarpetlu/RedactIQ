"""Lossless Telugu normalization and conservative IndicNER boundary repair.

All public span coordinates refer to the original Python string. Normalization
is a matching view only: callers must always slice the original text.
"""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any, Mapping, Sequence

from ..telugu_utils import is_akshara_boundary, trim_telugu_honorifics


_JOINERS = frozenset("\u200c\u200d")
_TRIM = frozenset(" \t\r\n,;:!?()[]{}\"'“”‘’।॥")


@dataclass(frozen=True)
class NormalizedText:
    """NFC matching view and original intervals for each normalized code point.

    ``origins[i]`` is the original half-open interval that produced ``text[i]``.
    A removed joiner is included when a mapped span crosses its position.
    """

    text: str
    origins: tuple[tuple[int, int], ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start <= end <= len(self.text):
            raise ValueError("Normalized span is out of bounds")
        if start == end:
            position = self.origins[start][0] if start < len(self.origins) else (
                self.origins[-1][1] if self.origins else 0
            )
            return position, position
        return self.origins[start][0], self.origins[end - 1][1]


def normalize_with_offsets(text: str) -> NormalizedText:
    """Build an NFC view without joiners, retaining original offset intervals.

    Segmentation at base characters lets NFC compose vowel signs without
    requiring an unreliable diff between entire Unicode strings.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    output: list[str] = []
    origins: list[tuple[int, int]] = []
    pending: list[str] = []
    pending_start = 0
    pending_end = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        for char in unicodedata.normalize("NFC", "".join(pending)):
            output.append(char)
            origins.append((pending_start, pending_end))
        pending = []

    for index, char in enumerate(text):
        if char in _JOINERS:
            # Joiners affect rendering but are ignored in the matching view.
            # The next mapped character begins after the joiner; a span across
            # it still maps to the exact original slice.
            flush()
            continue
        if unicodedata.category(char).startswith("M") and pending:
            pending.append(char)
            pending_end = index + 1
            continue
        flush()
        pending = [char]
        pending_start = index
        pending_end = index + 1
    flush()
    return NormalizedText("".join(output), tuple(origins))


def _token_char(char: str) -> bool:
    return char.isalnum() or unicodedata.category(char).startswith("M") or char in _JOINERS


def _snap_start(text: str, start: int) -> int:
    while start > 0 and not is_akshara_boundary(text, start):
        start -= 1
    return start


def _snap_end(text: str, end: int) -> int:
    while end < len(text) and not is_akshara_boundary(text, end):
        end += 1
    return end


def repair_boundaries(text: str, entity: Mapping[str, Any]) -> dict[str, Any]:
    """Repair a raw model proposal while preserving its label and confidence.

    Partial-token predictions expand to the token edge. Honorifics and loose
    punctuation are removed. Suffix trimming belongs to the shared Telugu
    utilities and is applied by the caller after this repair.
    """
    result = dict(entity)
    start, end = result.get("start"), result.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        return result
    original = (start, end)
    start = _snap_start(text, start)
    end = _snap_end(text, end)
    while start > 0 and _token_char(text[start - 1]) and _token_char(text[start]):
        start = _snap_start(text, start - 1)
    while end < len(text) and _token_char(text[end - 1]) and _token_char(text[end]):
        end = _snap_end(text, end + 1)
    while start < end and text[start] in _TRIM:
        start += 1
    while start < end and text[end - 1] in _TRIM:
        end -= 1
    start, end = trim_telugu_honorifics(text, start, end)
    start, end = _snap_start(text, start), _snap_end(text, end)
    if start >= end:
        return result
    result["start"], result["end"] = start, end
    result["text"] = text[start:end]
    if (start, end) != original:
        actions = list(result.get("enrichment_actions") or [])
        if "boundary_repaired" not in actions:
            actions.append("boundary_repaired")
        result["enrichment_actions"] = actions
    return result


def merge_fragments(text: str, entities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent same-label pieces when at least one is an initial.

    Arbitrary neighboring full words may be separate people or places; those
    need cue evidence and are deliberately left for the caller to reconcile.
    """
    def is_initial(surface: str) -> bool:
        value = surface.rstrip(".")
        units = sum(is_akshara_boundary(value, i) for i in range(1, len(value) + 1))
        return bool(value) and (units == 1 or (surface.endswith(".") and units <= 2))

    ordered = sorted((dict(item) for item in entities), key=lambda e: (e.get("start", -1), e.get("end", -1)))
    merged: list[dict[str, Any]] = []
    for item in ordered:
        if not merged:
            merged.append(item)
            continue
        prev = merged[-1]
        gap = text[prev["end"]:item["start"]] if 0 <= prev.get("end", -1) <= item.get("start", -1) <= len(text) else None
        left = text[prev["start"]:prev["end"]] if 0 <= prev.get("start", -1) <= prev.get("end", -1) <= len(text) else ""
        right = text[item["start"]:item["end"]] if 0 <= item.get("start", -1) <= item.get("end", -1) <= len(text) else ""
        initial = is_initial(left) or is_initial(right)
        if (gap is not None and gap.isspace() and len(gap) <= 2 and initial
                and prev.get("entity_type") == item.get("entity_type")
                and left and right):
            prev["end"] = item["end"]
            prev["text"] = text[prev["start"]:prev["end"]]
            prev["score"] = max(float(prev.get("score", 0)), float(item.get("score", 0)))
            actions = list(prev.get("enrichment_actions") or [])
            if "fragments_merged" not in actions:
                actions.append("fragments_merged")
            prev["enrichment_actions"] = actions
        else:
            merged.append(item)
    return merged
