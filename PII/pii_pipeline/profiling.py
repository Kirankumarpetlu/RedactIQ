"""Content-free, process-scoped stage accounting; no sampling threads required."""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional

try:
    import psutil
except ImportError:  # Keep profiling usable in minimal/offline deployments.
    psutil = None


_MIB = 1024 * 1024


@dataclass
class StageMetrics:
    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    cpu_percent: float = 0.0
    cpu_cores_used: float = 0.0
    cpu_times_user: float = 0.0
    cpu_times_system: float = 0.0
    system_cpu_percent: float = 0.0
    process_rss_start_mb: float = 0.0
    process_rss_end_mb: float = 0.0
    process_rss_delta_mb: float = 0.0
    accelerator: str = "none"
    accelerator_allocated_delta_mb: float = 0.0
    accelerator_driver_delta_mb: float = 0.0
    depth: int = 0
    status: str = "ok"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("start_time")
        result.pop("end_time")
        result["cpu_time_user_sec"] = result.pop("cpu_times_user")
        result["cpu_time_system_sec"] = result.pop("cpu_times_system")
        result["cpu_time_sec"] = self.cpu_times_user + self.cpu_times_system
        return result


class PipelineProfiler:
    """Inclusive stage metrics. CPU%: 100 equals one fully occupied logical core."""
    def __init__(self):
        self.stages = []
        self._stack = contextvars.ContextVar("stage_stack", default=())

    @staticmethod
    def _resource_snapshot() -> dict[str, float | str]:
        """Return process/system counters only; never inspect input content."""
        snapshot: dict[str, float | str] = {
            "rss_mb": 0.0,
            "system_cpu_time": 0.0,
            "accelerator": "none",
            "allocated_mb": 0.0,
            "driver_mb": 0.0,
        }
        if psutil is not None:
            try:
                snapshot["rss_mb"] = psutil.Process().memory_info().rss / _MIB
                cpu_times = psutil.cpu_times()
                snapshot["system_cpu_time"] = sum(cpu_times)
            except (OSError, AttributeError):
                pass
        try:
            import torch
            if torch.cuda.is_available():
                snapshot.update({
                    "accelerator": "cuda",
                    "allocated_mb": torch.cuda.memory_allocated() / _MIB,
                    "driver_mb": torch.cuda.memory_reserved() / _MIB,
                })
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                snapshot["accelerator"] = "mps"
                snapshot["allocated_mb"] = torch.mps.current_allocated_memory() / _MIB
                snapshot["driver_mb"] = torch.mps.driver_allocated_memory() / _MIB
        except (ImportError, AttributeError, RuntimeError):
            pass
        return snapshot

    def start_stage(self, stage_name, metadata=None):
        stack = self._stack.get()
        stage = StageMetrics(stage_name, metadata=dict(metadata or {}), depth=len(stack))
        cpu = os.times()
        resources = self._resource_snapshot()
        stage.start_time = time.perf_counter()
        self._stack.set(stack + ((stage, cpu, resources),))
        return stage

    def end_stage(self, stage_name=None):
        stack = self._stack.get()
        if not stack:
            return None
        stage, initial, resources = stack[-1]
        if stage_name is not None and stage.stage_name != stage_name:
            raise RuntimeError("Stages must finish in nesting order")
        stage.end_time = time.perf_counter()
        final = os.times()
        final_resources = self._resource_snapshot()
        elapsed = stage.end_time - stage.start_time
        stage.duration_ms = elapsed * 1000
        stage.cpu_times_user = max(0.0, final.user - initial.user)
        stage.cpu_times_system = max(0.0, final.system - initial.system)
        stage.cpu_cores_used = (stage.cpu_times_user + stage.cpu_times_system) / elapsed if elapsed else 0.0
        stage.cpu_percent = stage.cpu_cores_used * 100
        stage.process_rss_start_mb = float(resources["rss_mb"])
        stage.process_rss_end_mb = float(final_resources["rss_mb"])
        stage.process_rss_delta_mb = stage.process_rss_end_mb - stage.process_rss_start_mb
        stage.accelerator = str(final_resources["accelerator"])
        stage.accelerator_allocated_delta_mb = float(final_resources["allocated_mb"]) - float(resources["allocated_mb"])
        stage.accelerator_driver_delta_mb = float(final_resources["driver_mb"]) - float(resources["driver_mb"])
        if elapsed and psutil is not None:
            system_cpu_delta = float(final_resources["system_cpu_time"]) - float(resources["system_cpu_time"])
            stage.system_cpu_percent = max(0.0, system_cpu_delta / elapsed / (os.cpu_count() or 1) * 100)
        self._stack.set(stack[:-1])
        self.stages.append(stage)
        return stage

    def get_summary(self):
        return {
            "total_duration_ms": sum(s.duration_ms for s in self.stages if s.depth == 0),
            "stage_count": len(self.stages),
            "logical_cpu_count": os.cpu_count(),
            "cpu_scope": "current_process_all_threads; excludes child processes and remote servers",
            "timing_scope": "inclusive wall time; accelerator operations are not synchronized",
            "cores_definition": "average logical core equivalents = CPU seconds / wall seconds",
            "system_cpu_definition": "host CPU utilization during the stage, normalized across logical CPUs; 0 when psutil is unavailable",
            "memory_scope": "RSS is current-process memory; accelerator deltas are CUDA allocated/reserved or MPS allocated/driver memory",
            "stages": [s.to_dict() for s in sorted(self.stages, key=lambda s: s.start_time)],
        }

    def print_report(self):
        print_report(self.get_summary())


def print_report(summary):
    summary = summary.get("aggregate", summary)
    print("\nStage                                      Time ms  Proc CPU%  Sys CPU%  Avg cores   RSS ΔMB")
    for stage in summary["stages"]:
        name = "  " * stage["depth"] + stage["stage_name"]
        print(f"{name:<42} {stage['duration_ms']:9.2f} {stage['cpu_percent']:10.1f} {stage['system_cpu_percent']:9.1f} {stage['cpu_cores_used']:10.3f} {stage['process_rss_delta_mb']:9.2f}")
    print(f"Total: {summary['total_duration_ms']:.2f} ms (nested stages are inclusive)")


_active = contextvars.ContextVar("active_profiler", default=None)
_last = contextvars.ContextVar("last_profiler", default=None)
_UNSET = object()


def get_profiler():
    p = _active.get() or _last.get()
    if p is None:
        p = PipelineProfiler()
        _last.set(p)
    return p


@contextlib.contextmanager
def profile_context(stage_name, metadata=None, profiler=_UNSET):
    # Explicit None disables measurement; omitted uses the active session only.
    p = _active.get() if profiler is _UNSET else profiler
    if p is None:
        yield None
        return
    stage = p.start_stage(stage_name, metadata)
    try:
        yield stage
    except BaseException:
        stage.status = "error"  # Never record exception messages or input values.
        raise
    finally:
        p.end_stage(stage_name)


def profile_stage(stage_name=None, metadata=None, profiler=_UNSET):
    def decorator(func):
        name = stage_name or func.__name__
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with profile_context(name, metadata, profiler):
                    return await func(*args, **kwargs)
            return async_wrapper
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profile_context(name, metadata, profiler):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_run(func):
    """Isolate each request, including nested JSON field runs, and attach metrics."""
    signature = inspect.signature(func)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        enabled = bound.arguments.get("enable_profiling", False)
        p = PipelineProfiler() if enabled else None
        parent = _active.get()
        token = _active.set(p)
        try:
            with profile_context(func.__name__, profiler=p):
                result = func(*args, **kwargs)
            if p is not None:
                summary = p.get_summary()
                if func.__name__ == "run_json":
                    fields = result.pop("profiling", {}).get("per_field", [])
                    result["profiling"] = {"per_field": fields, "aggregate": summary}
                else:
                    result["profiling"] = summary
            return result
        finally:
            _active.reset(token)
            if p is not None:
                if parent is not None:
                    depth = len(parent._stack.get())
                    parent.stages.extend(replace(stage, depth=stage.depth + depth) for stage in p.stages)
                _last.set(p)
    return wrapper


def reset_profiler(profiler=None):
    p = profiler or get_profiler()
    if p._stack.get():
        raise RuntimeError("Cannot reset a profiler with active stages")
    p.stages.clear()


def time_stage(stage_name: str, func: Callable, *args, **kwargs):
    p = PipelineProfiler()
    with profile_context(stage_name, profiler=p) as stage:
        result = func(*args, **kwargs)
    return result, stage


if __name__ == "__main__":
    import time
    
    print("Testing PipelineProfiler...")
    profiler = PipelineProfiler()
    
    # Test 1: Basic timing
    profiler.start_stage('indic_ner_enrichment')
    time.sleep(0.05)
    profiler.end_stage('indic_ner_enrichment')
    
    # Test 2: Context manager  
    with profile_context('presidio_orchestrator', profiler=profiler):
        time.sleep(0.03)
    
    # Test 3: Decorator
    @profile_stage('gliner_vanilla_fallback', profiler=profiler)
    def fallback_fn():
        time.sleep(0.02)
        return 'done'
    
    fallback_fn()
    
    # Test 4: Another context
    with profile_context('anonymizer', profiler=profiler):
        time.sleep(0.01)
    
    profiler.print_report()
    print("\nAll tests passed!")
