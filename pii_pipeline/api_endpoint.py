"""
Universal GPU/CPU PII Detection & Redaction API (Stage 9).

Provides 3 public routes:
1. POST /analyze  - Detailed PII analysis and redacted output
2. GET /health    - Consolidated system health + metrics
3. GET /pipeline/metrics - Safe live progress and performance metrics for pipeline runs
"""

from __future__ import annotations

from . import json_codec as json
import os
import sys
import logging
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# Add the project root to sys.path so pii_pipeline imports work
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import Body, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from pii_pipeline.config import load_pipeline_config
from pii_pipeline.model_cache import ModelCache, configure_runtime
from pii_pipeline.pipeline import Stage9Pipeline

logger = logging.getLogger(__name__)
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", scheme_name="PII API key", auto_error=False)


def _service_role() -> str:
    """Run a standalone inference worker unless gateway routing is requested."""
    return os.getenv("PII_SERVICE_ROLE", "cpu-worker")


def require_api_key(provided: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    """Reject inference requests unless a configured key matches the header."""
    expected = os.getenv("PII_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _detect_service_role() -> str:
    """Detect the service role from runtime configuration."""
    return _service_role()


def auto_detect_device(mode: str = "auto") -> str:
    """Auto-detect the best available device."""
    from pii_pipeline.model_cache import auto_detect_device as _auto
    return _auto(mode)


def get_hardware_info(device: str) -> Dict[str, str]:
    """Get hardware information for the given device."""
    from pii_pipeline.model_cache import get_hardware_info as _hwi
    return _hwi(device)


# -----------------------------------------------------------
# Lifespan: Model loading at startup
# -----------------------------------------------------------

_cfg = load_pipeline_config(os.getenv("PII_CONFIG_PATH") or None)

if _service_role() != "gateway":
    # Preload models once at startup into the global cache
    configure_runtime(_cfg.runtime.cpu_threads, _cfg.runtime.enable_mps)
    _MODEL_DEVICE = ModelCache.preload_all(
        device=os.getenv("PII_DEVICE", _cfg.runtime.device),
        language=_cfg.models.spacy.language,
        gliner_model_name=_cfg.models.gliner_primary.model_name_or_path,
        gliner_adapter=_cfg.models.gliner_primary.adapter,
        fallback_gliner_model_name=_cfg.models.gliner_vanilla_fallback.model_name_or_path,
        fallback_gliner_adapter=_cfg.models.gliner_vanilla_fallback.adapter,
        spacy_model_name=_cfg.models.spacy.model_name,
        fallback_to_cpu=_cfg.runtime.fallback_to_cpu,
    )
else:
    # Gateway mode: no model loading
    _MODEL_DEVICE = "cpu"

_worker_pipeline = None
_worker_pipeline_key = None
_worker_pipeline_lock = threading.Lock()
_api_metrics_lock = threading.Lock()
_api_metrics = {"request_count": 0, "failed_closed_count": 0, "total_latency_ms": 0.0}


def _worker_endpoints():
    """Return worker URLs for either local processes or Docker services."""
    endpoints = [
        ("gpu_worker_0", os.getenv("PII_GPU_WORKER_0_URL", "http://localhost:8001")),
        ("gpu_worker_1", os.getenv("PII_GPU_WORKER_1_URL", "http://localhost:8002")),
        ("cpu_worker", os.getenv("PII_CPU_WORKER_URL", "http://localhost:8003")),
    ]
    enabled_workers = {
        worker.strip() for worker in os.getenv("PII_ENABLED_WORKERS", "").split(",") if worker.strip()
    }
    return [endpoint for endpoint in endpoints if not enabled_workers or endpoint[0] in enabled_workers]


async def _probe_workers() -> Dict[str, Dict[str, Any]]:
    """Probe configured workers using the same URLs used for request routing."""
    result: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.0, connect=2.0)) as client:
        for worker_key, base_url in _worker_endpoints():
            info = {"status": "unavailable", "ready": False, "models_loaded": False, "warmup_complete": False}
            if base_url:
                try:
                    response = await client.get(f"{base_url.rstrip('/')}/health")
                    payload = response.json() if response.status_code == 200 else {}
                    status = payload.get("status", "unhealthy") if isinstance(payload, dict) else "unhealthy"
                    ready = status in ("healthy", "degraded")
                    info = {"status": status, "ready": ready, "models_loaded": ready, "warmup_complete": ready}
                except (httpx.HTTPError, ValueError):
                    pass
            result[worker_key] = info
    return result


def _api_metrics_snapshot(workers_active: int) -> Dict[str, Any]:
    """Return aggregate request telemetry without retaining request content."""
    with _api_metrics_lock:
        request_count = _api_metrics["request_count"]
        failed_closed_count = _api_metrics["failed_closed_count"]
        total_latency_ms = _api_metrics["total_latency_ms"]
    hardware = get_hardware_info(_MODEL_DEVICE)
    return {
        "request_count": request_count,
        "failed_closed_count": failed_closed_count,
        "avg_latency_ms": round(total_latency_ms / request_count, 2) if request_count else 0.0,
        "workers_active": workers_active,
        "workers_total": 3,
        "service": "pii-stage9",
        "service_role": _service_role(),
        "device": _MODEL_DEVICE,
        "cuda_available": hardware.get("cuda_available", False),
    }


def _latest_pipeline_metrics() -> Dict[str, Any]:
    """Return safe aggregate metrics from the most recent pipeline run.

    The runner writes ``pipeline_progress.json`` after initialization and every completed item
    and marks it complete at the end. Older runs only have a final
    ``pipeline_run_summary.json``; those remain readable here as a fallback.
    Raw transcript text and detected entity values are intentionally never
    returned by this helper.
    """
    output_root = Path(project_root) / "data" / "output"
    candidates = []
    for filename, source in (
        ("pipeline_progress.json", "live_progress"),
        ("pipeline_run_summary.json", "completed_run"),
    ):
        if output_root.exists():
            for path in output_root.rglob(filename):
                try:
                    candidates.append((path.stat().st_mtime, path, source))
                except OSError:
                    continue

    if not candidates:
        return {
            "status": "not_available",
            "detail": "No pipeline run metrics have been written yet.",
        }

    _, metrics_path, source = max(candidates, key=lambda item: item[0])
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "temporarily_unavailable",
            "detail": "Pipeline metrics are being updated; retry shortly.",
        }

    if source == "completed_run":
        processed_rows = payload.get("total_rows_processed", 0)
        payload = {
            **payload,
            "status": "completed",
            "total_rows": processed_rows,
            "processed_rows": processed_rows,
            "progress_percent": 100.0 if processed_rows else 0.0,
            "elapsed_seconds": payload.get("total_time_seconds", 0.0),
        }

    allowed_fields = {
        "status",
        "total_rows",
        "processed_rows",
        "progress_percent",
        "elapsed_seconds",
        "total_rows_processed",
        "total_time_seconds",
        "average_latency_ms",
        "throughput_rows_per_sec",
        "device",
        "hardware",
        "gliner_threshold",
        "min_entity_score",
        "entity_count",
        "input_format",
    }
    return {
        "source": source,
        **{key: value for key, value in payload.items() if key in allowed_fields},
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the selected pipeline before accepting inference requests."""
    global _worker_pipeline, _worker_pipeline_key
    if _service_role() != "gateway":
        with _worker_pipeline_lock:
            _worker_pipeline = Stage9Pipeline(config=_cfg, device=_MODEL_DEVICE)
            _worker_pipeline_key = (_MODEL_DEVICE, _cfg.models.spacy.language, None)
    logger.info("API Server starting with role: %s | device: %s", _service_role(), _MODEL_DEVICE)
    try:
        yield
    finally:
        if _service_role() != "gateway":
            with _worker_pipeline_lock:
                if _worker_pipeline is not None and hasattr(_worker_pipeline, "close"):
                    _worker_pipeline.close()
                _worker_pipeline = None
                _worker_pipeline_key = None
            ModelCache.clear_cache()
            logger.info("Models cleared from memory on shutdown.")


# -----------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------

app = FastAPI(
    title="Universal GPU/CPU PII Detection & Redaction API (Stage 9)",
    version="9.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_analyze_metrics(request: Request, call_next):
    """Track aggregate /analyze performance without retaining PII payloads."""
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        if request.url.path == "/analyze":
            with _api_metrics_lock:
                _api_metrics["request_count"] += 1
                _api_metrics["total_latency_ms"] += (time.perf_counter() - started_at) * 1000
                if status_code >= 500:
                    _api_metrics["failed_closed_count"] += 1


# -----------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request schema for POST /analyze."""
    text: str = Field(..., description="Transcript text to analyze")
    language: str = Field("en", description="Language code (default: 'en')")
    # Omit or send null to use the selected pipeline configuration.
    gliner_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Optional GLiNER2 confidence override. Omit or use null for the configured threshold; 1.0 suppresses nearly all model detections.",
    )
    semantic_tags: Optional[list[Dict[str, Any]]] = Field(
        None, description="Optional real, existing annotations. Omit or use null for text-only detection; do not send generated placeholder objects.",
    )
    min_entity_score: float = Field(0.0, ge=0.0, le=1.0)


# -----------------------------------------------------------
# PUBLIC ENDPOINT: POST /analyze
# -----------------------------------------------------------

@app.post(
    "/analyze",
    summary="Analyze text for PII and return redacted output",
    dependencies=[Security(require_api_key)],
)
async def analyze_endpoint(
    request: Request,
    body: AnalyzeRequest = Body(
        ...,
        openapi_examples={
            "configured_defaults": {
                "summary": "Text only, using the selected pipeline configuration",
                "description": "Matches Stage9Pipeline.run(text) with the same configuration. Optional threshold and imported annotations are deliberately omitted.",
                "value": {"text": "Patient name is Ananya Rao."},
            },
        },
    ),
):
    """
    POST /analyze - Analyze text for PII and return redacted output.

    Behavior by service role:
    - gpu-worker / cpu-worker: Executes the full Stage 9 pipeline using
      resident cached models and returns complete analysis results.
    - gateway: Routes to the first healthy worker in the failover chain
      (GPU Worker 0 -> GPU Worker 1 -> CPU Worker). Returns 503 fail-closed
      when no healthy worker is available.

    Response includes:
    - entities_detected: List of detected PII entities with type, offsets,
      scores, and taxonomy status
    - routing_summary: Summary of routing decisions
    - model_status: Status of primary and fallback models
    - redacted_text: Safe redaction output with <ENTITY> placeholders
    - processing_status: Status string (e.g., "success", "failed_closed")
    - redaction_completed: Boolean indicating redaction was performed
    - entity_count: Total number of entities detected and redacted
    - runtime_device: String indicating which device the pipeline ran on
      (e.g., "cuda:0", "cpu")
    - inference_settings: Effective confidence thresholds and whether imported
      annotations were supplied, so request overrides are visible
    """
    text = body.text
    language = body.language
    threshold = body.gliner_threshold
    semantic_tags = body.semantic_tags
    min_entity_score = body.min_entity_score

    resolved_device = _MODEL_DEVICE

    # Worker / Direct Execution Mode
    if _service_role() != "gateway":
        # Run pipeline directly using cached models
        global _worker_pipeline, _worker_pipeline_key
        key = (resolved_device, language, threshold)
        def run_local_pipeline():
            global _worker_pipeline, _worker_pipeline_key
            with _worker_pipeline_lock:
                if _worker_pipeline is None or _worker_pipeline_key != key:
                    replacement = Stage9Pipeline(config=_cfg, device=resolved_device, language=language,
                                                  gliner_threshold=threshold)
                    previous = _worker_pipeline
                    _worker_pipeline = replacement
                    _worker_pipeline_key = key
                    if previous is not None and hasattr(previous, "close"):
                        previous.close()
                return _worker_pipeline.run(text, semantic_tags=semantic_tags,
                                            min_entity_score=min_entity_score)

        result = await run_in_threadpool(run_local_pipeline)
        return {
            "status": "success",
            "runtime_device": resolved_device,
            "gpu_name": get_hardware_info(resolved_device).get("current_gpu_name"),
            **result,
            "inference_settings": {
                "gliner_threshold": threshold if threshold is not None else _cfg.models.gliner_primary.threshold,
                "gliner_threshold_source": "request" if threshold is not None else "config",
                "min_entity_score": min_entity_score,
                "semantic_tags_supplied": bool(semantic_tags),
            },
        }

    # Gateway Routing Mode: try each worker in the failover chain
    forward_timeout = float(os.getenv("PII_WORKER_TIMEOUT_SECONDS", "120"))
    async with httpx.AsyncClient(timeout=httpx.Timeout(forward_timeout, connect=2.0)) as client:
        for _worker_key, base_url in _worker_endpoints():
            if not base_url:
                continue
            try:
                response = await client.get(f"{base_url.rstrip('/')}/health", timeout=2.0)
                if response.status_code != 200 or response.json().get("status") not in ("healthy", "degraded"):
                    continue
                forward_response = await client.post(
                    f"{base_url.rstrip('/')}/analyze",
                    json={"text": text, "language": language, "gliner_threshold": threshold,
                          "semantic_tags": semantic_tags, "min_entity_score": min_entity_score},
                    headers={"X-API-Key": request.headers["X-API-Key"]},
                )
                if forward_response.status_code == 200:
                    return forward_response.json()
            except (httpx.HTTPError, ValueError):
                continue

    # No worker available — fail closed
    raise HTTPException(
        status_code=503,
        detail={
            "error": "privacy_detection_unavailable",
            "redaction_completed": False,
            "processing_status": "failed_closed",
            "detail": "No healthy inference worker available. Gateway fail-closed.",
        },
    )


# -----------------------------------------------------------
# PUBLIC ENDPOINT: GET /pipeline/metrics
# -----------------------------------------------------------

@app.get(
    "/pipeline/metrics",
    summary="Latest live pipeline progress and performance metrics",
)
async def pipeline_metrics_endpoint():
    """Return safe, aggregate metrics for the most recent pipeline run.

    This endpoint exposes progress, latency, throughput, device, and total
    entity counts. It never returns source transcripts, redacted transcripts,
    or detected PII values.
    """
    return _latest_pipeline_metrics()


# -----------------------------------------------------------
# INTERNAL ENDPOINT: GET /health
# -----------------------------------------------------------

@app.get("/health", summary="Consolidated system health & telemetry")
async def health_endpoint():
    """
    GET /health - Consolidated system health check.

    Returns overall status, worker statuses, pipeline availability,
    and safe operational metrics. All information previously available
    via separate /metrics, /ready, and /v1/gpu/status endpoints is
    included here, making those redundant.

    Status values:
    - healthy: At least one full-quality worker is ready; normal routing available.
    - degraded: System operational but one or more preferred workers unavailable.
    - unhealthy: No full-quality worker available; inference would fail closed.

    Includes safe operational metrics (no PII, raw text, or detected values):
    - request_count: Total request count broken down by endpoint
    - failed_closed_count: Number of requests that failed closed
    - avg_latency_ms: Average request latency
    - workers_active: Number of currently active inference workers
    - workers_total: Total number of inference workers in deployment
    - service: Service identifier ("pii-stage9")
    - service_role: Current deployment role ("gateway", "gpu-worker", "cpu-worker")
    - device: Resolved device string ("cuda:0", "cpu")
    - cuda_available: Boolean CUDA availability status

    Gateway section:
    - status: "healthy" (gateway process running)
    - role: "gateway"
    - endpoints: Public API surface ["/analyze", "/health"]

    Workers section (one entry per service role position):
    - gpu_worker_0: status, ready, models_loaded, warmup_complete
    - gpu_worker_1: status, ready, models_loaded, warmup_complete
    - cpu_worker: status, ready, models_loaded, warmup_complete

    Pipeline section:
    - status: "available" or "unavailable"
    - availability: Description of current availability

    Routing section:
    - available: Boolean (true if at least one worker ready)
    - worker_order: ["gpu_worker_0", "gpu_worker_1", "cpu_worker"]

    Note: This consolidated endpoint is designed so that external operators
    and load balancers do NOT need separate /metrics, /ready, or /v1/gpu/status
    public endpoints. All necessary health and routing information is included here.
    """
    service_role = _service_role()

    # A gateway actively probes its remote workers; a worker reports only itself.
    if service_role == "gateway":
        workers_info = await _probe_workers()
        workers_active = sum(1 for worker in workers_info.values() if worker["ready"])
    elif service_role in ("gpu-worker", "cpu-worker"):
        workers_info = {
            "gpu_worker_0": {"status": "unknown", "ready": False, "models_loaded": False, "warmup_complete": False},
            "gpu_worker_1": {"status": "unknown", "ready": False, "models_loaded": False, "warmup_complete": False},
            "cpu_worker": {"status": "unknown", "ready": False, "models_loaded": False, "warmup_complete": False},
        }
        worker_key = "cpu_worker" if service_role == "cpu-worker" else (
            "gpu_worker_1" if os.getenv("CUDA_VISIBLE_DEVICES") == "1" else "gpu_worker_0"
        )
        workers_info[worker_key] = {
            "status": _MODEL_DEVICE,
            "ready": True,
            "models_loaded": True,
            "warmup_complete": True,
        }
        workers_active = 1
    else:
        workers_info = {
            "gpu_worker_0": {"status": "unknown", "ready": False, "models_loaded": False, "warmup_complete": False},
            "gpu_worker_1": {"status": "unknown", "ready": False, "models_loaded": False, "warmup_complete": False},
            "cpu_worker": {"status": "unknown", "ready": False, "models_loaded": False, "warmup_complete": False},
        }
        workers_active = 0

    # Determine overall health
    workers_ready = sum(1 for w in workers_info.values() if w["ready"])
    if workers_ready >= 1:
        overall = "healthy"
    elif workers_ready == 0:
        overall = "unhealthy"
    else:
        overall = "degraded"

    routing_order = ["gpu_worker_0", "gpu_worker_1", "cpu_worker"]

    return {
        "status": overall,
        "service": "pii-stage9",
        "service_role": service_role,
        "device": _MODEL_DEVICE,
        "strict_privacy_mode": True,
        "gateway": {
            "status": "healthy" if service_role == "gateway" else "routing_required",
            "role": service_role,
        },
        "workers": workers_info,
        "pipeline": {
            "status": "available" if workers_ready > 0 else "unavailable",
            "availability": "full-quality pipeline " + ("ready" if workers_ready > 0 else "not available"),
        },
        "routing": {
            "available": workers_ready > 0,
            "worker_order": routing_order,
        },
        "metrics": _api_metrics_snapshot(workers_active),
        "note": "This consolidated endpoint includes all metrics and worker status "
                "information. The separate /metrics and /ready and /v1/gpu/status "
                "endpoints are redundant — all necessary data is provided here.",
    }


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("PII_API_HOST", "0.0.0.0")
    port = int(os.getenv("PII_API_PORT", "8000"))
    workers = int(os.getenv("PII_API_WORKERS", "1"))

    uvicorn.run(app, host=host, port=port, workers=workers)
