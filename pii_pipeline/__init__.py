"""Public pipeline API, loaded on demand so asset utilities stay lightweight."""
from importlib import import_module

_EXPORTS = {
    "Stage9Pipeline": ".pipeline",
    "Stage4Pipeline": ".pipeline",
    "PipelineConfig": ".config",
    "load_pipeline_config": ".config",
    "IndicNERPipeline": ".models.indic_ner",
    "ModelCache": ".model_cache",
    "auto_detect_device": ".model_cache",
    "get_hardware_info": ".model_cache",
    "PipelineProfiler": ".profiling",
    "get_profiler": ".profiling",
    "profile_stage": ".profiling",
    "profile_context": ".profiling",
    "StageMetrics": ".profiling",
    "reset_profiler": ".profiling",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
