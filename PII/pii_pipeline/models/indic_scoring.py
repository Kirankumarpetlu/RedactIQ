"""Inference-only IndicNER controls; no learned parameters or model loading."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def validate_inference_controls(strategy="argmax", temperature=1.0, min_word_margin=0.0):
    if not isinstance(strategy, str) or strategy not in {"argmax", "constrained_bio"}:
        raise ValueError("IndicNER decoding_strategy must be argmax or constrained_bio")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not .01 <= temperature <= 100):
        raise ValueError("IndicNER temperature must be finite and between 0.01 and 100")
    if (isinstance(min_word_margin, bool) or not isinstance(min_word_margin, (int, float))
            or not math.isfinite(min_word_margin) or not 0 <= min_word_margin <= 1):
        raise ValueError("IndicNER min_word_margin must be finite and between 0 and 1")
    return strategy, float(temperature), float(min_word_margin)


def validate_confidence_thresholds(values):
    if not isinstance(values, Mapping):
        raise ValueError("IndicNER confidence_thresholds must be a mapping")
    scripts = {"devanagari", "bengali", "gujarati", "gurmukhi", "odia", "tamil",
               "telugu", "kannada", "malayalam", "latin", "other", "numeric"}
    result = {}
    for key, value in values.items():
        parts = key.casefold().split(":") if isinstance(key, str) else []
        if (len(parts) not in {1, 2} or parts[0] not in {"person", "location", "organization"}
                or len(parts) == 2 and parts[1] not in scripts):
            raise ValueError("IndicNER confidence_thresholds keys must be entity or entity:script")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("IndicNER confidence thresholds must be finite and between 0 and 1")
        normalized = ":".join(parts)
        if normalized in result:
            raise ValueError("Duplicate normalized IndicNER confidence threshold")
        result[normalized] = float(value)
    return result


def decode_probabilities(probabilities: Sequence[Sequence[float]], id2label: Mapping[int, str],
                         strategy="argmax") -> list[int]:
    """Viterbi with fixed BIO legality constraints, or unchanged token argmax.

    Transition scores are zero for valid paths and negative infinity otherwise.
    There is no trained CRF. Call only on real tokens, excluding padding/specials.
    """
    if strategy not in {"argmax", "constrained_bio"}:
        raise ValueError("Unknown IndicNER decoding_strategy")
    if not probabilities:
        return []
    labels = [id2label[i] for i in range(len(probabilities[0]))]
    if strategy == "argmax":
        return [max(range(len(row)), key=row.__getitem__) for row in probabilities]

    def legal(previous, current):
        if not current.startswith("I-"):
            return True
        return (previous is not None and previous.startswith(("B-", "I-"))
                and previous[2:] == current[2:])

    emissions = [[math.log(max(float(p), 1e-300)) for p in row] for row in probabilities]
    scores = [emissions[0][i] if legal(None, label) else -math.inf
              for i, label in enumerate(labels)]
    backpointers = []
    for row in emissions[1:]:
        current_scores, previous_indices = [], []
        for index, label in enumerate(labels):
            allowed = [i for i, previous in enumerate(labels) if legal(previous, label)]
            previous_index = max(allowed, key=scores.__getitem__)
            current_scores.append(scores[previous_index] + row[index])
            previous_indices.append(previous_index)
        scores = current_scores
        backpointers.append(previous_indices)
    index = max(range(len(scores)), key=scores.__getitem__)
    path = [index]
    for previous_indices in reversed(backpointers):
        index = previous_indices[index]
        path.append(index)
    return list(reversed(path))
