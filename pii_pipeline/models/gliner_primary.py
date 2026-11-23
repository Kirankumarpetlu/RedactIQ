"""
GLiNER2 PII Primary (green box) -- 42-type PII taxonomy, explicitly no PHI.

Wraps `fastino/gliner2-privacy-filter-PII-multi` as a Presidio
EntityRecognizer so it runs inside the same AnalyzerEngine pass as the
built-in regex/checksum recognizers and the Presidio Medical recognizers.

Integrates with ModelCache for single startup load and memory residency.
Auto-detects any available GPU (CUDA 0..N, MPS, CPU).

Supports sliding window inference for texts longer than model max length (512).
Supports ONNX Runtime hybrid inference (encoder in ONNX, heads in PyTorch).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from ..model_cache import ModelCache, auto_detect_device, extract_gliner_entities, resolve_gliner_adapter
from ..taxonomy import get_entity_schema

logger = logging.getLogger(__name__)

MODEL_NAME = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_WINDOW = 512
DEFAULT_STRIDE = 384


class GLiNER2PIIPrimary(EntityRecognizer):
    """Presidio EntityRecognizer wrapping GLiNER2's 42-type PII taxonomy model."""

    def __init__(
        self,
        threshold: float = 0.5,
        model_name: str = MODEL_NAME,
        adapter: str = "legacy_span",
        entity_schema: str = "pii",
        device: str = "auto",
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
        use_onnx: bool = False,
        onnx_encoder_path: Optional[str] = None,
        window_stride_multiplier: int = 1,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self.adapter = resolve_gliner_adapter(model_name, adapter)
        self.entity_schema = entity_schema
        self.label_schema: Dict[str, str] = get_entity_schema(entity_schema)
        self.device = auto_detect_device(device)
        self.window = window
        self.stride = stride
        if (isinstance(window_stride_multiplier, bool) or not isinstance(window_stride_multiplier, int)
                or window_stride_multiplier < 1):
            raise ValueError("window_stride_multiplier must be a positive integer")
        if window_stride_multiplier > 1 and (stride <= 0 or stride * window_stride_multiplier > window * 4):
            raise ValueError("Window subsampling must keep complete text coverage")
        self.window_stride_multiplier = window_stride_multiplier
        self.use_onnx = use_onnx
        self.onnx_encoder_path = onnx_encoder_path
        self._model = None
        self._onnx_model = None

        super().__init__(
            supported_entities=list(self.label_schema),
            name="GLiNER2PIIPrimary",
            supported_language="en",
        )
        self.load()

    def load(self) -> None:
        """Loads or reuses the model from the singleton in-memory cache."""
        if self.use_onnx and self.adapter == "legacy_span":
            try:
                from .onnx_inference import GLiNER2ONNXHybrid
                self._onnx_model = GLiNER2ONNXHybrid(
                    model_name=self.model_name,
                    device=self.device,
                    threshold=self.threshold,
                    onnx_encoder_path=self.onnx_encoder_path,
                )
                logger.info(
                    "GLiNER2 PII Primary active (ONNX hybrid): %s on device %s | window=%d stride=%d",
                    self.model_name, self.device, self.window, self.stride
                )
                return
            except Exception as e:
                logger.warning("ONNX model load failed, falling back to PyTorch: %s", e)
        elif self.use_onnx:
            logger.warning("ONNX hybrid inference only supports legacy_span GLiNER models; using native PyTorch boundary adapter.")
        
        self._model = ModelCache.get_or_load_gliner(self.model_name, self.device, adapter=self.adapter)
        logger.info(
            "GLiNER PII Primary active (%s/PyTorch): %s on device %s (cached) | window=%d stride=%d",
            self.adapter, self.model_name, self.device, self.window, self.stride
        )

    @property
    def inference_backend(self) -> str:
        """Return the active backend, including an explicit ONNX fallback state."""
        if self._onnx_model is not None:
            return "onnx"
        backend = "pytorch_fallback" if self.use_onnx else "pytorch"
        return f"{backend}:{self.adapter}"

    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> List[RecognizerResult]:
        requested = set(entities or self.supported_entities)
        wanted = [e for e in self.label_schema if e in requested]
        if not wanted:
            return []

        # GLiNER 2.5 has a 4096-token boundary window.  The legacy adapter's
        # character approximation caused unnecessary overlapping calls and
        # duplicate candidates on ordinary multi-lingual records.
        if self.adapter == "boundary" or len(text) <= self.window * 4:
            return self._analyze_single(text, wanted)

        return self._analyze_sliding_window(text, wanted)

    def _analyze_single(self, text: str, wanted: List[str]) -> List[RecognizerResult]:
        try:
            if self._onnx_model is not None:
                raw = extract_gliner_entities(self._onnx_model, text, wanted, threshold=self.threshold)
            else:
                query = {label: self.label_schema[label] for label in wanted} if self.adapter == "boundary" else wanted
                raw = extract_gliner_entities(self._model, text, query, threshold=self.threshold)
        except Exception:
            logger.exception("GLiNER2 PII Primary inference failed")
            return []

        results: List[RecognizerResult] = []
        for label, spans in (raw.get("entities") or raw).items():
            if label not in wanted:
                continue
            for span in spans:
                score = float(span.get("confidence", span.get("score", 0.0)))
                if score < self.threshold:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=label,
                        start=span["start"],
                        end=span["end"],
                        score=score,
                        recognition_metadata={"recognizer_name": self.name, "recognizer_identifier": self.id},
                    )
                )
        self._release_idle_cuda_workspace()
        return results

    def _release_idle_cuda_workspace(self) -> None:
        """Keep variable-length windows from retaining most of a small GPU.

        Large attention buffers with slightly different shapes can accumulate
        in PyTorch's cache. Windows may back those idle allocations with system
        RAM once VRAM fills. Reclaim only unused blocks under memory pressure;
        model weights, inference windows and predictions stay resident/intact.
        """
        if self._onnx_model is not None or not self.device.startswith("cuda"):
            return
        import torch
        capacity = torch.cuda.get_device_properties(self.device).total_memory
        # Leave headroom for the next window's attention buffers. Waiting until
        # the cache nearly fills VRAM lets a larger subsequent window spill.
        if torch.cuda.memory_reserved(self.device) > capacity * 0.5:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def _analyze_sliding_window(self, text: str, wanted: List[str]) -> List[RecognizerResult]:
        """Process long text via overlapping windows, merge results."""
        all_results: List[RecognizerResult] = []

        starts = range(0, len(text), self.stride)
        if self.window_stride_multiplier > 1:
            # Select existing window positions and always retain the original
            # final window. Changing its alignment would change model context.
            remaining = max(0, len(text) - self.window * 4)
            last_start = ((remaining + self.stride - 1) // self.stride) * self.stride
            starts = list(range(0, last_start + 1, self.stride * self.window_stride_multiplier))
            if starts[-1] != last_start:
                starts.append(last_start)
        for start_char in starts:
            end_char = min(start_char + self.window * 4, len(text))
            window_text = text[start_char:end_char]

            window_results = self._analyze_single(window_text, wanted)

            for r in window_results:
                r.start += start_char
                r.end += start_char
                all_results.append(r)

            if end_char == len(text):
                break

        return self._merge_overlapping(all_results)

    def _merge_overlapping(self, results: List[RecognizerResult], iou_threshold: float = 0.5) -> List[RecognizerResult]:
        """Deduplicate entities spanning window boundaries."""
        if not results:
            return []

        results = sorted(results, key=lambda r: (r.start, -r.score))
        merged = [results[0]]

        for current in results[1:]:
            last = merged[-1]
            overlap_start = max(last.start, current.start)
            overlap_end = min(last.end, current.end)
            overlap = max(0, overlap_end - overlap_start)
            union = (last.end - last.start) + (current.end - current.start) - overlap
            iou = overlap / union if union > 0 else 0

            if iou >= iou_threshold and last.entity_type == current.entity_type:
                if current.score > last.score:
                    merged[-1] = current
            else:
                merged.append(current)

        return merged
