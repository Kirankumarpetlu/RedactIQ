"""Portable GGUF GLiNER2 adapter based on the supplied reference runtime."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "gliner2-pii-portable"


class PortableGGUFGLiNER:
    """Run the supplied CPU-only GGUF GLiNER2 model with safe text windowing."""

    def __init__(
        self,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        threshold: float = 0.95,
        window_chars: int = 1500,
        stride_chars: int = 1200,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not 0 < stride_chars <= window_chars:
            raise ValueError("stride_chars must be greater than 0 and no greater than window_chars")
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.window_chars = window_chars
        self.stride_chars = stride_chars
        self._reference = self._load_reference()
        self.labels = tuple(self._reference.SPEC["labels"])
        self._reference.attention = self._attention

    def _load_reference(self) -> Any:
        runner_path = self.model_dir / "runner.py"
        model_path = self.model_dir / "gliner2-pii-f16.gguf"
        tokenizer_path = self.model_dir / "tokenizer.json"
        schema_path = self.model_dir / "schema_prefix.json"
        missing = [path.name for path in (runner_path, model_path, tokenizer_path, schema_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Portable GGUF model is incomplete: {', '.join(missing)}")
        spec = importlib.util.spec_from_file_location("portable_gliner2_gguf_reference", runner_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load portable GGUF reference: {runner_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _attention(self, hidden: np.ndarray, layer: int, buckets: np.ndarray, position_embedding: np.ndarray) -> np.ndarray:
        reference = self._reference
        prefix = f"encoder.encoder.layer.{layer}.attention"
        query, key, value = (
            reference.lin(hidden, f"{prefix}.self.{name}_proj")
            for name in ("query", "key", "value")
        )
        position_query = reference.lin(position_embedding, f"{prefix}.self.query_proj")
        position_key = reference.lin(position_embedding, f"{prefix}.self.key_proj")
        scale = math.sqrt(reference.HEAD_DIM * 3)
        output = np.zeros_like(hidden)
        token_indices = np.arange(len(hidden))[:, None]

        for head in range(reference.HEADS):
            head_slice = slice(head * reference.HEAD_DIM, (head + 1) * reference.HEAD_DIM)
            query_head = query[:, head_slice]
            key_head = key[:, head_slice]
            value_head = value[:, head_slice]
            position_query_head = position_query[:, head_slice]
            position_key_head = position_key[:, head_slice]
            word_to_word = query_head @ key_head.T
            word_to_position_all = query_head @ position_key_head.T
            position_to_word_all = key_head @ position_query_head.T
            position_indices = np.clip(buckets + reference.BUCKETS, 0, position_key_head.shape[0] - 1)
            reverse_position_indices = np.clip(-buckets + reference.BUCKETS, 0, position_query_head.shape[0] - 1)
            word_to_position = word_to_position_all[token_indices, position_indices]
            position_to_word = position_to_word_all[token_indices, reverse_position_indices]
            output[:, head_slice] = reference.softmax(
                (word_to_word + word_to_position + position_to_word.T) / scale
            ) @ value_head

        output = reference.lin(output, f"{prefix}.output.dense")
        return reference.ln(output + hidden, f"{prefix}.output.LayerNorm")

    @staticmethod
    def _overlaps(left: MappingEntity, right: MappingEntity) -> bool:
        return left["start"] < right["end"] and right["start"] < left["end"]

    def _window_entities(self, text: str, offset: int) -> List[Dict[str, Any]]:
        raw_entities = self._reference.extract(text, threshold=self.threshold)
        return [
            {
                "entity_type": entity["label"],
                "start": int(entity["start"]) + offset,
                "end": int(entity["end"]) + offset,
                "score": float(entity["score"]),
            }
            for entity in raw_entities
        ]

    def extract(self, text: str, labels: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """Return non-overlapping GGUF entities for the model's baked-in label set."""

        if not text:
            return []
        wanted = set(labels) if labels is not None else set(self.labels)
        candidates: List[Dict[str, Any]] = []
        for start in range(0, len(text), self.stride_chars):
            end = min(start + self.window_chars, len(text))
            candidates.extend(self._window_entities(text[start:end], start))
            if end == len(text):
                break

        candidates = [entity for entity in candidates if entity["entity_type"] in wanted]
        selected: List[Dict[str, Any]] = []
        for candidate in sorted(
            candidates,
            key=lambda entity: (-entity["score"], entity["start"], entity["end"], entity["entity_type"]),
        ):
            if any(
                candidate["entity_type"] == selected_entity["entity_type"]
                and self._overlaps(candidate, selected_entity)
                for selected_entity in selected
            ):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda entity: (entity["start"], entity["end"], entity["entity_type"]))


MappingEntity = Dict[str, Any]
