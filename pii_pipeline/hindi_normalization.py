"""Matching-only Hindi normalization with original Python offset recovery.

The normalized string must never be used as the redaction surface.  Every
reported interval is mapped back to complete original grapheme clusters.
"""
from __future__ import annotations

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class NormalizedHindi:
    text: str
    # For every normalized character, the complete original cluster interval.
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        if not 0 <= start < end <= len(self.text):
            raise ValueError("Normalized span is outside the text")
        return self.starts[start], self.ends[end - 1]


def _clusters(text: str):
    start = 0
    for index, char in enumerate(text):
        if (index and not unicodedata.category(char).startswith("M")
                and char not in "\u200c\u200d"
                and text[index - 1] not in "\u200c\u200d\u094d"):
            yield start, index
            start = index
    if text:
        yield start, len(text)


def normalize_with_offsets(text: str) -> NormalizedHindi:
    """Fold Hindi spelling variants while retaining complete original spans.

    NFC is applied per grapheme.  Nukta and chandrabindu/anusvara are folded
    only for matching, joiners are removed, and Devanagari digits become ASCII.
    Unchanged Latin or other-script text remains byte-for-byte identical.
    """
    pieces: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for start, end in _clusters(text):
        cluster = unicodedata.normalize("NFC", text[start:end])
        cluster = cluster.replace("\u093c", "").replace("\u0901", "\u0902")
        cluster = cluster.replace("\u200c", "").replace("\u200d", "")
        cluster = "".join(str(unicodedata.decimal(c)) if "\u0966" <= c <= "\u096f" else c for c in cluster)
        pieces.append(cluster)
        starts.extend([start] * len(cluster))
        ends.extend([end] * len(cluster))
    return NormalizedHindi("".join(pieces), tuple(starts), tuple(ends))
