"""
In-Memory Model Cache & Universal Hardware Device Detection (Stage 9).

Ensures all ML models (GLiNER2 Primary, GLiNER2 Vanilla Fallback, spaCy NLP Engine,
and Presidio Analyzers) are loaded ONCE at application startup, stored in memory cache,
and reused across all requests and pipeline instances.

Supports ANY GPU (NVIDIA GeForce RTX, Tesla, A100, H100, L4, T4, etc.),
Apple Silicon MPS, or CPU fallback automatically.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Dict, Iterable, Optional, Tuple

# Configure PyTorch threading for CPU inference BEFORE importing torch
CPU_THREADS = int(os.environ.get("PII_PIPELINE_CPU_THREADS", "0")) or min(8, os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

import torch

if CPU_THREADS > 1:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)

logger = logging.getLogger(__name__)
logger.info(f"PyTorch configured with {torch.get_num_threads()} intra-op threads, {torch.get_num_interop_threads()} inter-op threads")


class ModelLoadError(RuntimeError):
    """Raised when a required model cannot be loaded; partial redaction is unsafe."""


_GLINER_ADAPTERS = {"legacy_span", "boundary", "auto"}


def resolve_gliner_adapter(model_name: str, adapter: str = "legacy_span") -> str:
    """Resolve a declared GLiNER checkpoint contract without changing output shape."""
    selected = (adapter or "legacy_span").strip().lower()
    if selected not in _GLINER_ADAPTERS:
        raise ValueError("GLiNER adapter must be one of: legacy_span, boundary, auto.")
    if selected != "auto":
        return selected
    # GLiNER 2.5 checkpoints have a boundary architecture. Keep unknown names
    # on the established span loader rather than silently changing behavior.
    return "boundary" if "gliner2.5" in model_name.lower() else "legacy_span"


def extract_gliner_entities(
    model: Any,
    text: str,
    labels: Iterable[str] | Mapping[str, str],
    threshold: float = 0.5,
) -> Dict[str, list[Dict[str, Any]]]:
    """Normalize every supported GLiNER adapter to the pipeline entity contract."""
    # Boundary schema models accept ``{label: description}``; preserving that
    # mapping is essential for domain-specific PII/PHI extraction.  Legacy
    # GLiNER checkpoints continue to receive their established label list.
    query: Any = dict(labels) if isinstance(labels, Mapping) else list(labels)
    raw = model.extract_entities(text, query, threshold=threshold, include_confidence=True, include_spans=True)
    source = raw.get("entities", raw) if isinstance(raw, dict) else None
    if not isinstance(source, dict):
        raise ModelLoadError("GLiNER adapter returned an invalid entity payload; expected a label-to-spans mapping.")

    normalized: Dict[str, list[Dict[str, Any]]] = {}
    for label, spans in source.items():
        if not isinstance(label, str) or not isinstance(spans, list):
            continue
        accepted = []
        for span in spans:
            if not isinstance(span, dict) or "start" not in span or "end" not in span:
                continue
            if any(isinstance(span[key], bool) or not isinstance(span[key], Integral) for key in ("start", "end")):
                continue
            try:
                start, end = int(span["start"]), int(span["end"])
                confidence = float(span.get("confidence", span.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
            if 0 <= start < end <= len(text):
                accepted.append({"start": start, "end": end, "confidence": confidence})
        if accepted:
            normalized[label] = accepted
    return {"entities": normalized}


def _available_cpu_count() -> int:
    """Return the CPUs assigned to this process, respecting Linux CPU affinity."""
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return len(affinity)
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def auto_detect_device(requested_device: str = "auto") -> str:
    """
    Auto-detect and return the optimal available compute device.
    
    Compatible with ANY GPU:
    - Any NVIDIA / CUDA GPU (cuda:0, cuda:1, etc.)
    - Apple Silicon MPS (mps) if available
    - CPU fallback if no GPU exists
    """
    requested = (requested_device or "auto").strip().lower()
    if requested == "cuda":
        requested = "cuda:0"

    if requested not in ("auto", ""):
        if requested == "cpu":
            return "cpu"
        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS was requested but is unavailable; using CPU.")
            return "cpu"
        if requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                logger.warning("%s was requested but CUDA is unavailable; using CPU.", requested)
                return "cpu"
            try:
                index = int(requested.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError("CUDA devices must use the form 'cuda:<index>'.") from exc
            if 0 <= index < torch.cuda.device_count():
                return requested
            logger.warning("%s is outside the available CUDA device range; using CPU.", requested)
            return "cpu"
        raise ValueError("device must be one of: auto, cpu, mps, cuda, or cuda:<index>.")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_index = 0
        free_memory: list[tuple[int, int]] = []
        for index in range(gpu_count):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
                free_memory.append((free_bytes, index))
            except (AttributeError, RuntimeError):
                # Older PyTorch builds may not expose per-device free memory.
                pass
        if free_memory:
            _, gpu_index = max(free_memory)
        gpu_name = torch.cuda.get_device_name(gpu_index)
        logger.info(
            "Auto-detected CUDA GPU: %s (Total GPUs: %s) -> using 'cuda:%s'",
            gpu_name,
            gpu_count,
            gpu_index,
        )
        return f"cuda:{gpu_index}"
    
    # Check for Apple Silicon MPS (Metal Performance Shaders). The configured
    # native model can trigger a non-recoverable Metal assertion on some Macs,
    # so retain CPU as the portable automatic choice unless MPS was explicitly
    # enabled by runtime configuration.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if os.environ.get("PII_PIPELINE_ENABLE_MPS", "0").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("Auto-detected Apple Silicon MPS -> using 'mps'")
            return "mps"
        logger.info("Apple Silicon MPS detected; using stable CPU fallback (set runtime.enable_mps=true to opt in).")

    logger.info("No GPU detected -> using 'cpu'")
    return "cpu"


def configure_cpu_threads(cpu_threads: Optional[int]) -> int:
    """Apply automatic or fixed CPU parallelism before loading models."""
    if cpu_threads is None:
        cpu_threads = _available_cpu_count()
        logger.info("Auto-selected %s CPU threads from the process CPU allocation.", cpu_threads)
    if cpu_threads < 1:
        raise ValueError("runtime.cpu_threads must be 'auto' or a positive integer.")

    os.environ["PII_PIPELINE_CPU_THREADS"] = str(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before parallel work starts.
        pass
    return torch.get_num_threads()


def configure_runtime(cpu_threads: Optional[int], enable_mps: bool = False) -> int:
    """Apply portable runtime settings before selecting an automatic device."""
    os.environ["PII_PIPELINE_ENABLE_MPS"] = "1" if enable_mps else "0"
    return configure_cpu_threads(cpu_threads)


def get_hardware_info(device_str: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve detailed hardware information for telemetry and logging."""
    device = auto_detect_device(device_str or "auto")
    info: Dict[str, Any] = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "available_cpu_count": _available_cpu_count(),
        "configured_cpu_threads": int(os.environ.get("PII_PIPELINE_CPU_THREADS", torch.get_num_threads())),
    }

    if torch.cuda.is_available():
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
        curr_idx = 0
        if device.startswith("cuda:"):
            try:
                curr_idx = int(device.split(":")[-1])
            except ValueError:
                curr_idx = 0
        info["current_gpu_name"] = torch.cuda.get_device_name(curr_idx)
        props = torch.cuda.get_device_properties(curr_idx)
        info["total_memory_mb"] = round(props.total_memory / (1024 ** 2), 2)
        try:
            free_bytes, _ = torch.cuda.mem_get_info(curr_idx)
            info["free_memory_mb"] = round(free_bytes / (1024 ** 2), 2)
        except (AttributeError, RuntimeError):
            pass
        info["allocated_memory_mb"] = round(torch.cuda.memory_allocated(curr_idx) / (1024 ** 2), 2)
        info["reserved_memory_mb"] = round(torch.cuda.memory_reserved(curr_idx) / (1024 ** 2), 2)
    elif device == "mps":
        info["current_gpu_name"] = "Apple Silicon (MPS)"
    else:
        info["current_gpu_name"] = "CPU"

    return info


class ModelCache:
    """Thread-safe singleton in-memory cache for all NLP and ML models."""

    _lock = threading.Lock()
    _gliner_models: Dict[Tuple[str, str, str], Any] = {}
    _spacy_engines: Dict[Tuple[str, str, str], Any] = {}

    @classmethod
    def get_or_load_gliner(
        cls,
        model_name: str,
        device: str = "auto",
        adapter: str = "legacy_span",
    ) -> Any:
        """Load a declared GLiNER checkpoint adapter once or fail closed."""
        resolved_device = auto_detect_device(device)
        resolved_adapter = resolve_gliner_adapter(model_name, adapter)
        cache_key = (model_name, resolved_device, resolved_adapter)

        with cls._lock:
            if cache_key in cls._gliner_models:
                logger.debug("Reusing cached GLiNER %s model: %s on %s", resolved_adapter, model_name, resolved_device)
                return cls._gliner_models[cache_key]

            # Production configuration uses a local, reviewed model directory.
            # Do not allow an unavailable local artifact to turn into an
            # unbounded network retry during request/startup handling.
            local_path = Path(model_name).expanduser()
            if not local_path.exists() and (model_name.startswith(".") or model_name.startswith("/") or "/" in model_name and not model_name.startswith("hf://")):
                raise ModelLoadError(
                    f"Required local GLiNER model is missing: '{local_path}'. "
                    "Run `python scripts/provision_models.py` or set an approved model path in config/pipeline_config.yaml."
                )

            logger.info("Loading GLiNER %s model into memory cache: %s on %s...", resolved_adapter, model_name, resolved_device)
            try:
                if resolved_adapter == "boundary":
                    if sys.version_info < (3, 10):
                        raise ModelLoadError(
                            "GLiNER boundary checkpoints require Python 3.10+ and gliner2 with AutoExtractor. "
                            "Use a Python 3.10+ runtime or keep adapter: legacy_span."
                        )
                    try:
                        from gliner2 import AutoExtractor
                    except ImportError as exc:
                        raise ModelLoadError(
                            "GLiNER boundary checkpoints require gliner2 with AutoExtractor. "
                            "Install the supported GLiNER 2.5 runtime before selecting adapter: boundary."
                        ) from exc
                    model = AutoExtractor.from_pretrained(model_name, map_location=resolved_device)
                else:
                    from gliner2 import GLiNER2
                    model = GLiNER2.from_pretrained(model_name, map_location=resolved_device)
                cls._gliner_models[cache_key] = model
                logger.info("Successfully cached GLiNER %s model: %s on %s", resolved_adapter, model_name, resolved_device)
                return model
            except ModelLoadError:
                raise
            except Exception as exc:
                raise ModelLoadError(
                    f"Required GLiNER {resolved_adapter} model '{model_name}' could not load on '{resolved_device}'. "
                    "Install the model/dependencies and make weights available before starting Stage 9."
                ) from exc

    @classmethod
    def get_or_load_spacy_engine(
        cls,
        language: str = "en",
        device: str = "auto",
        model_name: str = "en_core_web_sm"
    ) -> Any:
        """Load spaCy NLP engine once and keep resident in memory."""
        resolved_device = auto_detect_device(device)
        spacy_device = "cuda" if resolved_device.startswith("cuda") else "cpu"
        cache_key = (language, spacy_device, model_name)

        with cls._lock:
            if cache_key in cls._spacy_engines:
                logger.debug("Reusing cached spaCy engine for lang: %s, model: %s, device: %s", language, model_name, spacy_device)
                return cls._spacy_engines[cache_key]

            logger.info("Loading spaCy NLP engine into memory cache (lang=%s, model=%s, device=%s)...", language, model_name, spacy_device)
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            engine = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": language, "model_name": model_name}],
                    "device": spacy_device,
                }
            ).create_engine()
            cls._spacy_engines[cache_key] = engine
            logger.info("Successfully cached spaCy NLP engine.")
            return engine

    @classmethod
    def preload_all(
        cls,
        device: str = "auto",
        language: str = "en",
        gliner_model_name: str = "models/gliner2-primary",
        gliner_adapter: str = "legacy_span",
        fallback_gliner_model_name: Optional[str] = None,
        fallback_gliner_adapter: str = "legacy_span",
        spacy_model_name: str = "en_core_web_sm",
        fallback_to_cpu: bool = True,
    ) -> str:
        """
        Preload all pipeline models into memory cache at startup.
        Eliminates per-request cold-start latency.
        """
        resolved_device = auto_detect_device(device)

        def _preload(active_device: str) -> None:
            hw_info = get_hardware_info(active_device)
            logger.info("Preloading models on hardware: %s (Device: %s)", hw_info.get("current_gpu_name"), active_device)
            cls.get_or_load_spacy_engine(language=language, device=active_device, model_name=spacy_model_name)
            cls.get_or_load_gliner(gliner_model_name, device=active_device, adapter=gliner_adapter)
            if fallback_gliner_model_name and (
                fallback_gliner_model_name != gliner_model_name
                or resolve_gliner_adapter(fallback_gliner_model_name, fallback_gliner_adapter)
                != resolve_gliner_adapter(gliner_model_name, gliner_adapter)
            ):
                cls.get_or_load_gliner(
                    fallback_gliner_model_name,
                    device=active_device,
                    adapter=fallback_gliner_adapter,
                )

        try:
            _preload(resolved_device)
        except ModelLoadError:
            if not fallback_to_cpu or resolved_device == "cpu":
                raise
            logger.warning("Model preload failed on %s; retrying once on CPU.", resolved_device)
            cls.clear_cache()
            resolved_device = "cpu"
            _preload(resolved_device)

        logger.info("All Stage 9 models successfully preloaded and resident in memory.")
        return resolved_device

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models and free device memory."""
        with cls._lock:
            cls._gliner_models.clear()
            cls._spacy_engines.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model cache cleared.")
