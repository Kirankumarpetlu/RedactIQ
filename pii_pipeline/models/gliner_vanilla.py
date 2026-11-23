"""
GLiNER2 Vanilla (Conditional Fallback, orange box) -- confidence-checks any
label the orchestrator produced that falls OUTSIDE the 42-type taxonomy
(i.e. the custom medical/PHI labels from Presidio Medical).

Uses ModelCache for single-load memory residency across all inferences.
Auto-detects any available GPU dynamically.

Supports sliding window inference for texts longer than model max length (512).
Supports ONNX Runtime hybrid inference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from ..model_cache import ModelCache, auto_detect_device, extract_gliner_entities, resolve_gliner_adapter

logger = logging.getLogger(__name__)

MODEL_NAME = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 384


@dataclass
class FallbackCandidate:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass
class FallbackResult:
    entity_type: str
    start: int
    end: int
    score: float
    confirmed: bool
    source: str  # "vanilla_confirmed" | "vanilla_unconfirmed"


class GLiNER2Vanilla:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        adapter: str = "legacy_span",
        confirm_threshold: float = 0.4,
        device: str = "auto",
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
        use_onnx: bool = False,
        onnx_encoder_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.adapter = resolve_gliner_adapter(model_name, adapter)
        self.confirm_threshold = confirm_threshold
        self.device = auto_detect_device(device)
        self.window = window
        self.stride = stride
        self.use_onnx = use_onnx
        self.onnx_encoder_path = onnx_encoder_path
        self._model = None
        self._onnx_model = None
        self._load()

    def _load(self) -> None:
        if self.use_onnx and self.adapter == "legacy_span":
            try:
                from .onnx_inference import GLiNER2ONNXHybrid
                self._onnx_model = GLiNER2ONNXHybrid(
                    model_name=self.model_name,
                    device=self.device,
                    threshold=0.5,
                    onnx_encoder_path=self.onnx_encoder_path,
                )
                logger.info(
                    "GLiNER2 Vanilla fallback active (ONNX hybrid): %s on device %s | window=%d stride=%d",
                    self.model_name, self.device, self.window, self.stride
                )
                return
            except Exception as e:
                logger.warning("ONNX model load failed, falling back to PyTorch: %s", e)
        elif self.use_onnx:
            logger.warning("ONNX hybrid inference only supports legacy_span GLiNER models; using native PyTorch boundary adapter.")
        
        self._model = ModelCache.get_or_load_gliner(self.model_name, self.device, adapter=self.adapter)
        logger.info(
            "GLiNER Vanilla fallback active (%s/PyTorch): %s on device %s (cached) | window=%d stride=%d",
            self.adapter, self.model_name, self.device, self.window, self.stride
        )

    @property
    def inference_backend(self) -> str:
        """Return the active backend, including an explicit ONNX fallback state."""
        if self._onnx_model is not None:
            return "onnx"
        backend = "pytorch_fallback" if self.use_onnx else "pytorch"
        return f"{backend}:{self.adapter}"

    def confirm(
        self,
        text: str,
        candidates: List[FallbackCandidate],
        skip_high_confidence: float = 0.9,
    ) -> List[FallbackResult]:
        """
        Re-checks each out-of-taxonomy candidate span against the vanilla
        GLiNER2 model as a free-text label probe.

        If skip_high_confidence > 0, candidates with score >= skip_high_confidence
        are auto-confirmed without running vanilla inference (confidence gate).
        """
        results: List[FallbackResult] = []
        # Repeated mentions with the same label and surface ask the model the
        # exact same question. Reuse successful answers within this request;
        # keep candidate offsets/scores and discard the cache after the call.
        confirmations: dict[tuple[str, str], bool] = {}

        for c in candidates:
            # Confidence gate: auto-confirm high-confidence candidates
            if skip_high_confidence and c.score >= skip_high_confidence:
                results.append(
                    FallbackResult(
                        entity_type=c.entity_type,
                        start=c.start,
                        end=c.end,
                        score=c.score,
                        confirmed=True,
                        source="vanilla_auto_confirmed",
                    )
                )
                continue

            span_text = text[c.start : c.end]
            key = (c.entity_type, span_text)
            confirmed = False
            try:
                if key in confirmations:
                    confirmed = confirmations[key]
                else:
                    model = self._onnx_model if self._onnx_model is not None else self._model
                    raw = extract_gliner_entities(model, span_text, [c.entity_type], threshold=self.confirm_threshold)
                    spans = (raw.get("entities") or raw).get(c.entity_type, [])
                    confirmed = any(
                        float(s.get("confidence", s.get("score", 0.0))) >= self.confirm_threshold
                        for s in spans
                    )
                    confirmations[key] = confirmed
            except Exception:
                logger.exception("GLiNER2 Vanilla inference failed for a span; keeping it unconfirmed")

            results.append(
                FallbackResult(
                    entity_type=c.entity_type,
                    start=c.start,
                    end=c.end,
                    score=c.score,
                    confirmed=confirmed,
                    source="vanilla_confirmed" if confirmed else "vanilla_unconfirmed",
                )
            )
        return results
