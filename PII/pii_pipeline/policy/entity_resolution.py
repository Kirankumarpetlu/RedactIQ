"""Deterministic entity ordering and exact-duplicate resolution."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


EntityWithSource = Tuple[Any, str]


def sort_and_deduplicate_entities(candidates: Sequence[EntityWithSource]) -> List[EntityWithSource]:
    """Keep the highest-scoring exact typed span and retain its source metadata."""

    selected: Dict[Tuple[str, int, int], EntityWithSource] = {}
    for entity, source in candidates:
        key = (str(entity.entity_type), int(entity.start), int(entity.end))
        existing = selected.get(key)
        if existing is None or float(entity.score) > float(existing[0].score):
            selected[key] = (entity, source)
    return sorted(
        selected.values(),
        key=lambda item: (
            int(item[0].start),
            int(item[0].end),
            -float(item[0].score),
            str(item[0].entity_type),
            item[1],
        ),
    )
