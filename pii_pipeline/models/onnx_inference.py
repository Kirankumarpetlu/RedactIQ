"""
ONNX Runtime inference for IndicNER (full ONNX).
GLiNER2 hybrid encoder is not yet stable - using PyTorch for now.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .. import json_codec as json

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    logger.warning("onnxruntime not available")


class GLiNER2ONNXHybrid:
    """Run GLiNER's transformer encoder in ONNX Runtime and decode in PyTorch."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        threshold: float = 0.5,
        onnx_encoder_path: Optional[str] = None,
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")

        from ..model_cache import ModelCache

        self.model_name = model_name
        self.threshold = threshold
        self.model = ModelCache.get_or_load_gliner(model_name, device)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.encoder_path = Path(
            onnx_encoder_path
            or Path(__file__).resolve().parents[2] / "onnx_models" / "gliner_primary" / "encoder.onnx"
        )
        if not self.encoder_path.is_file():
            raise FileNotFoundError(f"GLiNER primary ONNX encoder not found: {self.encoder_path}")

        self.session = ort.InferenceSession(
            str(self.encoder_path), providers=self._get_providers(device)
        )
        self.input_names = {input_meta.name for input_meta in self.session.get_inputs()}
        self.output_names = [output_meta.name for output_meta in self.session.get_outputs()]
        required_inputs = {"input_ids", "attention_mask"}
        if not required_inputs.issubset(self.input_names):
            raise RuntimeError(
                "GLiNER primary ONNX encoder has incompatible inputs: "
                f"{sorted(self.input_names)}"
            )
        if "last_hidden_state" not in self.output_names:
            raise RuntimeError(
                "GLiNER primary ONNX encoder has incompatible outputs: "
                f"{self.output_names}"
            )

    @staticmethod
    def _get_providers(device: str) -> List[str]:
        available = set(ort.get_available_providers())
        if device.startswith("cuda") and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.session.run(
            ["last_hidden_state"],
            {
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int64, copy=False),
                "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64, copy=False),
            },
        )
        return torch.from_numpy(outputs[0]).to(device=self.device, dtype=self.dtype)

    def extract_entities(
        self,
        text: str,
        entity_types: List[str],
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not text or not entity_types:
            return {"entities": {}}

        schema = self.model.create_schema().entities(entity_types)
        schema_dict = schema.build()
        classification_tasks = [
            item["task"] for item in schema_dict.get("classifications", [])
        ]
        metadata = {
            "field_metadata": schema._field_metadata,
            "entity_metadata": schema._entity_metadata,
            "relation_metadata": getattr(schema, "_relation_metadata", {}),
            "field_orders": schema._field_orders,
            "entity_order": schema._entity_order,
            "relation_order": getattr(schema, "_relation_order", []),
            "classification_tasks": classification_tasks,
        }

        self.model.processor.change_mode(is_training=False)
        batch = self.model.processor.collate_fn_inference(
            [(text, schema)], max_len=max_len
        ).to(self.device, self.dtype if self.dtype != torch.float32 else None)
        token_embeddings = self._encode(batch.input_ids, batch.attention_mask)
        all_token_embs, all_schema_embs = self.model.processor.extract_embeddings_from_batch(
            token_embeddings, batch.input_ids, batch
        )
        span_info = None
        if any(task_type != "classifications" for task_type in batch.task_types[0]) and all_token_embs[0].numel() > 0:
            span_info = self.model.compute_span_rep_batched([all_token_embs[0]])[0]

        result = self.model._extract_sample(
            token_embs=all_token_embs[0],
            schema_embs=all_schema_embs[0],
            schema_tokens_list=batch.schema_tokens_list[0],
            task_types=batch.task_types[0],
            text_tokens=batch.text_tokens[0],
            original_text=batch.original_texts[0],
            schema=batch.original_schemas[0],
            start_mapping=batch.start_mappings[0],
            end_mapping=batch.end_mappings[0],
            threshold=self.threshold if threshold is None else threshold,
            metadata=metadata,
            include_confidence=include_confidence,
            include_spans=include_spans,
            span_info=span_info,
        )
        return self.model.format_results(
            result,
            include_confidence=include_confidence,
            requested_relations=metadata["relation_order"],
            classification_tasks=classification_tasks,
        )


class IndicNERONNX:
    """Full ONNX Runtime wrapper for IndicNER with BIO tag grouping."""
    
    # BIO tag to entity type mapping for IndicNER
    BIO_TO_ENTITY = {
        "B-PER": "PERSON", "I-PER": "PERSON",
        "B-ORG": "ORGANIZATION", "I-ORG": "ORGANIZATION",
        "B-Loc": "LOCATION", "I-Loc": "LOCATION",
        "B-DATE": "DATE", "I-DATE": "DATE",
        "B-TIME": "TIME", "I-TIME": "TIME",
        "B-MONEY": "MONEY", "I-MONEY": "MONEY",
        "B-PERCENT": "PERCENT", "I-PERCENT": "PERCENT",
        "B-EMAIL": "EMAIL", "I-EMAIL": "EMAIL",
        "B-PHONE": "PHONE_NUMBER", "I-PHONE": "PHONE_NUMBER",
        "B-URL": "URL", "I-URL": "URL",
        "B-ID": "ID_NUMBER", "I-ID": "ID_NUMBER",
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        if not ORT_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")
        
        if model_path is None:
            model_path = str(Path(__file__).resolve().parents[2] / "onnx_models" / "indicner.onnx")
        
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"IndicNER ONNX model not found: {self.model_path}")
        self.device = device
        
        providers = self._get_providers(device)
        logger.info(f"Loading IndicNER ONNX: {model_path} on {providers}")
        
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Load tokenizer
        from transformers import AutoTokenizer
        model_dir = Path(__file__).resolve().parents[2] / "models" / "indicner" / "models" / "IndicNER"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Label mapping from IndicNER config
        config_path = model_dir / "config.json"
        self.id2label = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                if "id2label" in config:
                    self.id2label = {int(k): v for k, v in config["id2label"].items()}
        
        logger.info(f"IndicNER ONNX loaded: inputs={self.input_names}, outputs={self.output_names}")
    
    def _get_providers(self, device: str) -> List[str]:
        if device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    def _bio_to_entity(self, bio_tag: str) -> Optional[str]:
        """Convert BIO tag to entity type."""
        if bio_tag == "O":
            return None
        return self.BIO_TO_ENTITY.get(bio_tag, bio_tag)
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Run IndicNER prediction on text with BIO tag grouping."""
        if not text:
            return []
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        offset_mapping = inputs["offset_mapping"][0].tolist()
        
        # Run ONNX inference
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logits = self.session.run(self.output_names, ort_inputs)[0]  # (1, seq, num_labels)
        
        # Decode measured probabilities through the native reconstruction code.
        from models.indicner.entity_reconstruction import EntityReconstructor
        from .indic_scoring import decode_probabilities, validate_inference_controls
        strategy, temperature, margin = validate_inference_controls(
            getattr(self, "decoding_strategy", "argmax"), getattr(self, "temperature", 1.0),
            getattr(self, "min_word_margin", 0.0))
        scaled_logits = logits[0] / temperature
        stable_logits = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
        probabilities = np.exp(stable_logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        real = [(offset, distribution) for token, offset, distribution in zip(tokens, offset_mapping, probabilities)
                if token not in self.tokenizer.all_special_tokens and offset[0] < offset[1]]
        predictions = decode_probabilities([distribution.tolist() for _, distribution in real], self.id2label, strategy)
        raw_preds = []
        for (offset, distribution), pred_idx in zip(real, predictions):
            raw_preds.append({
                "text": text[offset[0]:offset[1]], "start": int(offset[0]), "end": int(offset[1]),
                "label": self.id2label[int(pred_idx)], "confidence": float(distribution[pred_idx]),
                "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(distribution)},
            })
        reconstruction = EntityReconstructor(min_word_margin=margin).reconstruct(text, raw_preds)
        self.last_diagnostics = {key: reconstruction[key] for key in ("subword_predictions", "word_predictions")}
        entities = reconstruction["entities"]
        return [{"text": e["text"], "start": e["start"], "end": e["end"],
                 "label": e["type"], "confidence": e["confidence"],
                 "probability_margin": e["probability_margin"]} for e in entities]

    def _get_providers(self, device: str) -> List[str]:
        if device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]


def get_onnx_indicner(**kwargs) -> 'IndicNERONNX':
    """Factory for IndicNER ONNX inference."""
    return IndicNERONNX(**kwargs)
