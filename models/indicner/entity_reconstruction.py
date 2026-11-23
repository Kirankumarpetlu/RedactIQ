"""
entity_reconstruction.py

Hybrid Hierarchical Entity Reconstruction for IndicNER / Indic-language NER.

Pipeline:

    Raw tokenizer subword predictions
                |
                v
    Unicode-aware word segmentation
                |
                v
    Subword -> word alignment
                |
                v
    Weighted entity-type voting per word
                |
                v
    BIO-aware cross-word merging
                |
                v
    Final clean entity spans

Designed for Indic scripts where tokenizer subword offsets may split
visible words around Unicode combining marks.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# CONFIG
# ============================================================

LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",

    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "GPE": "LOCATION",

    "ORG": "ORGANIZATION",
    "ORGANIZATION": "ORGANIZATION",
}

# Minimum word-level confidence.
# Keep this relatively low initially because IndicNER can produce
# weak subword confidence inside an otherwise strongly supported word.
MIN_WORD_CONFIDENCE = 0.35

# Confidence blend:
# 0.70 = weighted model confidence
# 0.30 = agreement among subwords
ALPHA = 0.70

# Characters that split words.
BOUNDARY_CHARS = set(
    " \t\n\r"
    ".,!?;:"
    "\"'()[]{}<>"
    "|/\\"
    "—–-"
    "।॥"
    "‘’“”…"
)

ZERO_WIDTH_JOINERS = {
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
}

NON_BREAKING_CATEGORIES = {
    "Mn",  # Nonspacing mark
    "Mc",  # Spacing combining mark
    "Me",  # Enclosing mark
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SubwordPrediction:
    text: str
    start: int
    end: int
    label: str
    confidence: float
    probabilities: Optional[dict[str, float]] = None

    @property
    def prefix(self) -> str:
        """
        B-PER -> B
        I-PER -> I
        O     -> O
        """
        if self.label == "O":
            return "O"

        if "-" in self.label:
            return self.label.split("-", 1)[0]

        return "B"

    @property
    def raw_entity_type(self) -> str:
        """
        B-PER -> PER
        I-LOC -> LOC
        O     -> O
        """
        if self.label == "O":
            return "O"

        if "-" in self.label:
            return self.label.split("-", 1)[1]

        return self.label

    @property
    def entity_type(self) -> str:
        """
        Normalize model labels.

        PER -> PERSON
        LOC -> LOCATION
        ORG -> ORGANIZATION
        """
        raw = self.raw_entity_type

        if raw == "O":
            return "O"

        return LABEL_MAP.get(raw, raw)


@dataclass
class WordSpan:
    text: str
    start: int
    end: int

    subwords: list[SubwordPrediction] = field(
        default_factory=list
    )


@dataclass
class WordPrediction:
    text: str
    start: int
    end: int

    entity_type: str
    confidence: float

    bio_prefixes: list[str] = field(
        default_factory=list
    )
    probability_margin: float = 0.0
    outside_probability: float = 0.0


@dataclass
class Entity:
    text: str
    start: int
    end: int

    entity_type: str
    confidence: float
    probability_margin: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "type": self.entity_type,
            "confidence": round(self.confidence, 4),
            "probability_margin": round(self.probability_margin, 4),
            "start": self.start,
            "end": self.end,
        }

    def to_presidio_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "score": round(self.confidence, 4),
            "text": self.text,
        }


# ============================================================
# UNICODE HELPERS
# ============================================================

def is_non_breaking_char(char: str) -> bool:
    """
    Telugu/Indic combining marks must not create word boundaries.
    """

    if char in ZERO_WIDTH_JOINERS:
        return True

    category = unicodedata.category(char)

    if category in NON_BREAKING_CATEGORIES:
        return True

    return False


def is_boundary_char(char: str) -> bool:
    """
    Returns True only for actual visible word boundaries.
    """

    if is_non_breaking_char(char):
        return False

    return char in BOUNDARY_CHARS or unicodedata.category(char).startswith("P")


# ============================================================
# STEP 1: ORIGINAL WORD SEGMENTATION
# ============================================================

def segment_words(text: str) -> list[WordSpan]:
    """
    Segment the ORIGINAL text into word spans.

    Important:
    We use original text boundaries, not tokenizer pieces.
    """

    words = []

    i = 0
    n = len(text)

    while i < n:

        # Skip whitespace / punctuation boundaries.
        while i < n and is_boundary_char(text[i]):
            i += 1

        if i >= n:
            break

        start = i

        # Continue until an actual word boundary.
        while i < n and not is_boundary_char(text[i]):
            i += 1

        end = i

        if end > start:

            words.append(
                WordSpan(
                    text=text[start:end],
                    start=start,
                    end=end,
                )
            )

    return words


# ============================================================
# STEP 2: OFFSET OVERLAP
# ============================================================

def overlap_length(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:

    return max(
        0,
        min(end_a, end_b)
        -
        max(start_a, start_b)
    )


# ============================================================
# STEP 3: ALIGN SUBWORDS TO ORIGINAL WORDS
# ============================================================

def align_subwords_to_words(
    subwords: list[SubwordPrediction],
    words: list[WordSpan],
) -> None:
    """
    Attach every subword prediction to the original word(s)
    that overlap its character offsets.

    O predictions are also retained for debugging.
    """

    if not words:
        return

    sorted_subwords = sorted(
        subwords,
        key=lambda item: (
            item.start,
            item.end
        )
    )

    for subword in sorted_subwords:

        for word in words:

            if word.end <= subword.start:
                continue

            if word.start >= subword.end:
                break

            overlap = overlap_length(
                subword.start,
                subword.end,
                word.start,
                word.end,
            )

            if overlap > 0:
                word.subwords.append(subword)


# ============================================================
# STEP 4: WORD-LEVEL ENTITY DECISION
# ============================================================

def decide_word_label(
    word: WordSpan,
    min_word_margin: float = 0.0,
) -> Optional[WordPrediction]:
    """Vote over every covered character, including outside (O) evidence.

    Confidence is the winning type's mean probability over the whole word;
    partial coverage and contrary evidence cannot inflate it to near 1.0.
    Tokenizers may repeat offsets for combining marks: average those votes.
    """
    votes = [{} for _ in range(word.end - word.start)]
    counts = [0 for _ in votes]
    for sw in word.subwords:
        probabilities = sw.probabilities or ({"O": 1.0} if sw.label == "O" else
                                             {sw.label: sw.confidence, "O": 1.0 - sw.confidence})
        by_type = {}
        for label, probability in probabilities.items():
            raw_type = label.split("-", 1)[-1]
            entity_type = LABEL_MAP.get(raw_type, raw_type)
            by_type[entity_type] = by_type.get(entity_type, 0.0) + probability
        for position in range(max(word.start, sw.start), min(word.end, sw.end)):
            i = position - word.start
            counts[i] += 1
            for entity_type, probability in by_type.items():
                votes[i][entity_type] = votes[i].get(entity_type, 0.0) + probability
    scores = {"O": 0.0}
    for vote, count in zip(votes, counts):
        if not count:
            scores["O"] += 1.0
        else:
            for entity_type, score in vote.items():
                scores[entity_type] = scores.get(entity_type, 0.0) + score / count
    winner = max(scores, key=scores.get)
    confidence = scores[winner] / max(len(votes), 1)
    alternative = max((value for label, value in scores.items() if label != winner), default=0.0)
    margin = confidence - alternative / max(len(votes), 1)
    if winner == "O" or confidence < MIN_WORD_CONFIDENCE or margin < min_word_margin:
        return None
    return WordPrediction(word.text, word.start, word.end, winner, confidence,
                          [sw.prefix for sw in sorted(word.subwords, key=lambda sw: (sw.start, sw.end))
                           if sw.entity_type == winner], margin, scores.get("O", 0.0) / max(len(votes), 1))


# ============================================================
# STEP 5: CROSS-WORD MERGE DECISION
# ============================================================

def should_merge_words(
    previous: WordPrediction,
    current: WordPrediction,
    text: str,
) -> bool:
    """
    Decide whether two word-level entities should become one
    multi-word entity.
    """

    if previous.entity_type != current.entity_type:
        return False

    gap = text[
        previous.end:current.start
    ]

    # Never merge across punctuation.
    if not gap:
        return bool(current.bio_prefixes and current.bio_prefixes[0] == "I")

    # Only whitespace is allowed between words.
    if not gap.isspace():
        return False

    # Prevent merging across multiple words / large whitespace.
    if len(gap) > 3:
        return False

    # A new B at the beginning of a word starts a separate entity. Repeated
    # B subwords within one word are handled by the word-level vote above.
    return bool(current.bio_prefixes and current.bio_prefixes[0] == "I")


# ============================================================
# STEP 6: MERGE WORDS INTO FINAL ENTITIES
# ============================================================

def merge_word_predictions(
    word_predictions: list[WordPrediction],
    text: str,
) -> list[Entity]:

    if not word_predictions:
        return []

    word_predictions = sorted(
        word_predictions,
        key=lambda item: (
            item.start,
            item.end,
        )
    )

    entities = []

    cluster = [
        word_predictions[0]
    ]

    def flush_cluster(
        current_cluster: list[WordPrediction],
    ) -> Entity:

        start = current_cluster[0].start
        end = current_cluster[-1].end

        entity_type = (
            current_cluster[0].entity_type
        )

        # A strong adjacent word cannot hide a weak word at the span boundary.
        confidence = min(word.confidence for word in current_cluster)

        return Entity(
            # ALWAYS slice from original text.
            text=text[start:end],
            start=start,
            end=end,
            entity_type=entity_type,
            confidence=round(
                confidence,
                4,
            ),
            probability_margin=min(word.probability_margin for word in current_cluster),
        )

    for current in word_predictions[1:]:

        previous = cluster[-1]

        if should_merge_words(
            previous,
            current,
            text,
        ):

            cluster.append(current)

        else:

            entities.append(
                flush_cluster(cluster)
            )

            cluster = [current]

    entities.append(
        flush_cluster(cluster)
    )

    return entities


# ============================================================
# MAIN RECONSTRUCTOR
# ============================================================

class EntityReconstructor:
    """
    Hybrid reconstruction entry point.

    Example:

        reconstructor = EntityReconstructor()

        result = reconstructor.reconstruct(
            text,
            raw_predictions,
        )

        result["subword_predictions"]
        result["word_predictions"]
        result["entities"]
    """

    def __init__(self, min_word_margin: float = 0.0):
        from pii_pipeline.models.indic_scoring import validate_inference_controls
        _, _, self.min_word_margin = validate_inference_controls(min_word_margin=min_word_margin)

    def reconstruct(
        self,
        text: str,
        subword_predictions: list[
            dict | SubwordPrediction
        ],
    ) -> dict:

        # Normalize input.
        subwords = self._normalize_input(
            subword_predictions
        )

        # Step 1:
        # Segment original text into words.
        words = segment_words(text)

        # Step 2:
        # Align raw model predictions to words.
        align_subwords_to_words(
            subwords,
            words,
        )

        # Step 3:
        # Decide entity type for each word.
        word_predictions = []

        for word in words:

            prediction = decide_word_label(word, self.min_word_margin)

            if prediction is not None:
                word_predictions.append(
                    prediction
                )

        # Step 4:
        # Merge adjacent entity words.
        entities = merge_word_predictions(
            word_predictions,
            text,
        )

        return {

            # Raw model predictions.
            "subword_predictions": [
                {
                    "text": sw.text,
                    "start": sw.start,
                    "end": sw.end,
                    "label": sw.label,
                    "confidence": round(
                        sw.confidence,
                        4,
                    ),
                    "probabilities": sw.probabilities,
                }
                for sw in subwords
            ],

            # Hybrid word-level decisions.
            "word_predictions": [
                {
                    "text": word.text,
                    "start": word.start,
                    "end": word.end,
                    "type": word.entity_type,
                    "confidence": word.confidence,
                    "probability_margin": word.probability_margin,
                    "outside_probability": word.outside_probability,
                    "bio_prefixes": (
                        word.bio_prefixes
                    ),
                }
                for word in word_predictions
            ],

            # Final entities.
            "entities": [
                entity.to_dict()
                for entity in entities
            ],
        }

    @staticmethod
    def _normalize_input(
        predictions: list[
            dict | SubwordPrediction
        ],
    ) -> list[SubwordPrediction]:

        normalized = []

        for item in predictions:

            if isinstance(
                item,
                SubwordPrediction,
            ):

                normalized.append(item)

            else:

                normalized.append(
                    SubwordPrediction(
                        text=item.get(
                            "text",
                            item.get(
                                "token",
                                "",
                            ),
                        ),

                        start=int(
                            item["start"]
                        ),

                        end=int(
                            item["end"]
                        ),

                        label=item["label"],
                        probabilities=item.get("probabilities"),

                        confidence=float(
                            item.get(
                                "confidence",
                                item.get(
                                    "score",
                                    0.0,
                                ),
                            )
                        ),
                    )
                )

        return normalized
