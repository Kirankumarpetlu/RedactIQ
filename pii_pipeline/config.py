"""
Stage 9 Pipeline Configuration Loader & Schema.

Provides type-safe configuration management for the Multilingual Indic PII/PHI pipeline.
Supports loading from YAML/JSON files, auto-discovery of default config, and programmatic instantiation.
"""
from __future__ import annotations

from . import json_codec as json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Default config paths in order of preference
DEFAULT_CONFIG_LOCATIONS = [
    Path("config/pipeline_config.yaml"),
    Path("config/pipeline_config.json"),
    Path(__file__).resolve().parents[1] / "config" / "pipeline_config.yaml",
    Path(__file__).resolve().parents[1] / "config" / "pipeline_config.json",
]


@dataclass
class GLiNERConfig:
    model_name_or_path: str = "models/gliner2-primary"
    # legacy_span = GLiNER2; boundary = GLiNER 2.5 AutoExtractor; auto infers
    # the adapter only for known GLiNER 2.5 checkpoint names.
    adapter: str = "legacy_span"
    # ``pii`` is the established 42-label contract. ``pii_phi`` additionally
    # queries the six HIPAA identifier labels when using a schema model.
    entity_schema: str = "pii"
    threshold: float = 0.95
    window_size: int = 512
    stride: int = 384
    window_stride_multiplier: int = 1
    use_onnx: bool = False
    onnx_encoder_path: Optional[str] = "onnx_models/gliner_primary/encoder.onnx"


@dataclass
class GLiNERVanillaConfig:
    model_name_or_path: str = "models/gliner2-primary"
    adapter: str = "legacy_span"
    confirm_threshold: float = 0.40


@dataclass
class TeluguEnrichmentConfig:
    """Post-process IndicNER proposals containing Telugu script.

    Empty threshold overrides retain the existing IndicNER confidence gate.
    The enrichment layer must return its input unchanged without Telugu text.
    """

    enabled: bool = True
    normalization: bool = True
    boundary_repair: bool = True
    fragment_merge: bool = True
    suffix_stem: bool = True
    cue_evidence: bool = True
    type_reconciliation: bool = True
    confidence_recalibration: bool = True
    cue_boost: float = 0.0
    # Optional keys: person:cue, person:no_cue, location:cue,
    # location:no_cue. No-cue thresholds should preserve the strict gate.
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class HindiEnrichmentConfig:
    """Post-process existing Devanagari IndicNER proposals only."""

    enabled: bool = True
    normalization: bool = True
    boundary_repair: bool = True
    type_reconciliation: bool = True
    organization_suffix: bool = True
    confidence_recalibration: bool = True
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "person:cue": 0.45, "location:cue": 0.65, "organization:cue": 0.55,
    })


@dataclass
class IndicNERConfig:
    use_onnx: bool = False
    cpu_int8: bool = False
    # None follows runtime.device. CPU placement can leave GPU memory for
    # GLiNER when both checkpoints do not fit comfortably on a small GPU.
    device: Optional[str] = None
    model_path: str = "models/indicner"
    model_weights_path: Optional[str] = None
    onnx_model_path: Optional[str] = "onnx_models/indicner.onnx"
    # IndicNER is a sentence-level model: handed a whole multi-turn transcript it
    # predicts all-O and returns nothing, well below the 512-token limit. Text is
    # therefore split into sentence-sized chunks before inference.
    max_chunk_chars: int = 120
    chunk_overlap_chars: int = 40
    # IndicNER only contributes on Indic script; below this share of non-ASCII
    # letters a chunk is left to GLiNER2 and spaCy, which are stronger there.
    min_indic_fraction: float = 0.5
    # Minimum real softmax probability for an IndicNER span.
    min_confidence: float = 0.80
    # Empty overrides preserve min_confidence. Keys: person or person:tamil.
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)
    decoding_strategy: str = "argmax"
    temperature: float = 1.0
    min_word_margin: float = 0.0
    overlap_strategy: str = "best_context"
    telugu_enrichment: TeluguEnrichmentConfig = field(default_factory=TeluguEnrichmentConfig)
    hindi_enrichment: HindiEnrichmentConfig = field(default_factory=HindiEnrichmentConfig)


@dataclass
class SpacyConfig:
    model_name: str = "en_core_web_sm"
    language: str = "en"


@dataclass
class ModelsConfig:
    gliner_primary: GLiNERConfig = field(default_factory=GLiNERConfig)
    gliner_vanilla_fallback: GLiNERVanillaConfig = field(default_factory=GLiNERVanillaConfig)
    indic_ner: IndicNERConfig = field(default_factory=IndicNERConfig)
    spacy: SpacyConfig = field(default_factory=SpacyConfig)


@dataclass
class RuntimeConfig:
    device: str = "auto"
    # ``None`` represents automatic sizing from the CPU allocation visible to
    # this process. A positive integer pins execution to that thread count.
    cpu_threads: Optional[int] = None
    enable_profiling: bool = False
    fallback_to_cpu: bool = True
    enable_mps: bool = False
    # Overlap the CUDA primary detector with CPU IndicNER. The stages are
    # independent and are merged only after both complete.
    parallel_cpu_gpu: bool = False


@dataclass
class AnonymizerConfig:
    prefix: str = "<"
    suffix: str = ">"
    indexed: bool = True


@dataclass
class AcceptanceConfig:
    # Flat source:type:script keys, most specific first. Populate only after
    # calibration on independently reviewed development data.
    score_thresholds: Dict[str,float] = field(default_factory=dict)


@dataclass
class HindiConfig:
    organization_enabled: bool = True


@dataclass
class PipelineConfig:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    anonymizer: AnonymizerConfig = field(default_factory=AnonymizerConfig)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    hindi: HindiConfig = field(default_factory=HindiConfig)


def _load_raw_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            logger.warning("PyYAML not installed, attempting JSON parse for yaml file")
            return json.loads(text)
    return json.loads(text)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse config booleans without treating the string ``"false"`` as true."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_cpu_threads(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Parse ``auto``/empty as automatic sizing, or a positive fixed count."""
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "auto"}:
            return None
        value = normalized
    try:
        threads = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime.cpu_threads must be 'auto' or a positive integer.") from exc
    if threads < 1:
        raise ValueError("runtime.cpu_threads must be 'auto' or a positive integer.")
    return threads


def _as_gliner_adapter(value: Any, default: str = "legacy_span") -> str:
    """Validate the model loader selected by a configuration file."""
    adapter = str(value if value is not None else default).strip().lower()
    if adapter not in {"legacy_span", "boundary", "auto"}:
        raise ValueError("GLiNER adapter must be one of: legacy_span, boundary, auto.")
    return adapter


def _as_entity_schema(value: Any, default: str = "pii") -> str:
    schema = str(value if value is not None else default).strip().lower()
    if schema not in {"pii", "pii_phi"}:
        raise ValueError("GLiNER entity_schema must be one of: pii, pii_phi.")
    return schema


def _as_window_stride_multiplier(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("GLiNER window_stride_multiplier must be a positive integer.")
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError("GLiNER window_stride_multiplier must be a positive integer.") from exc
    if result < 1:
        raise ValueError("GLiNER window_stride_multiplier must be a positive integer.")
    return result


def _as_indic_device(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("models.indic_ner.device must be auto, cpu, mps or cuda:<index>.")
    device = value.strip().lower()
    if not device:
        return None
    if device in {"auto", "cpu", "mps", "cuda"} or (
            device.startswith("cuda:") and device[5:].isdigit()):
        return device
    raise ValueError("models.indic_ner.device must be auto, cpu, mps or cuda:<index>.")


def _telugu_enrichment_config(value: Any) -> TeluguEnrichmentConfig:
    """Parse independent Telugu enrichment controls with strict validation."""
    import math

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("models.indic_ner.telugu_enrichment must be a mapping.")
    defaults = TeluguEnrichmentConfig()
    switches = (
        "enabled", "normalization", "boundary_repair", "fragment_merge",
        "suffix_stem", "cue_evidence", "type_reconciliation",
        "confidence_recalibration",
    )
    allowed = {*switches, "cue_boost", "confidence_thresholds"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown Telugu enrichment setting: {sorted(unknown)[0]}.")
    parsed = {}
    for key in switches:
        setting = value.get(key, getattr(defaults, key))
        if not isinstance(setting, bool):
            raise ValueError(f"models.indic_ner.telugu_enrichment.{key} must be boolean.")
        parsed[key] = setting
    boost = value.get("cue_boost", defaults.cue_boost)
    if isinstance(boost, bool) or not isinstance(boost, (int, float)) or not math.isfinite(boost) or not 0 <= boost <= 1:
        raise ValueError("models.indic_ner.telugu_enrichment.cue_boost must be a finite score between 0 and 1.")
    thresholds = value.get("confidence_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("models.indic_ner.telugu_enrichment.confidence_thresholds must be a mapping.")
    valid_keys = {"person:cue", "person:no_cue", "location:cue", "location:no_cue"}
    for key, score in thresholds.items():
        if key not in valid_keys or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Telugu enrichment thresholds require person/location:cue/no_cue keys and finite scores between 0 and 1.")
    return TeluguEnrichmentConfig(**parsed, cue_boost=float(boost), confidence_thresholds=dict(thresholds))


def _hindi_enrichment_config(value: Any) -> HindiEnrichmentConfig:
    import math
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("models.indic_ner.hindi_enrichment must be a mapping")
    defaults = HindiEnrichmentConfig()
    switches = ("enabled", "normalization", "boundary_repair", "type_reconciliation",
                "organization_suffix", "confidence_recalibration")
    if set(value) - {*switches, "confidence_thresholds"}:
        raise ValueError("Unknown Hindi enrichment setting")
    parsed = {}
    for key in switches:
        setting = value.get(key, getattr(defaults, key))
        if not isinstance(setting, bool):
            raise ValueError(f"models.indic_ner.hindi_enrichment.{key} must be boolean")
        parsed[key] = setting
    thresholds = value.get("confidence_thresholds", defaults.confidence_thresholds)
    if not isinstance(thresholds, dict):
        raise ValueError("Hindi enrichment confidence_thresholds must be a mapping")
    allowed = {f"{label}:{state}" for label in ("person", "location", "organization")
               for state in ("cue", "no_cue")}
    for key, score in thresholds.items():
        if key not in allowed or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Hindi enrichment thresholds require entity:cue/no_cue and finite scores between 0 and 1")
    return HindiEnrichmentConfig(**parsed, confidence_thresholds=dict(thresholds))


def load_pipeline_config(config_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """
    Loads pipeline configuration from file or returns default settings.
    
    If `config_path` is None, checks environment variable `PII_CONFIG_PATH`,
    followed by default locations (`config/pipeline_config.yaml`).
    """
    target_path: Optional[Path] = None

    if config_path:
        target_path = Path(config_path)
    elif os.environ.get("PII_CONFIG_PATH"):
        target_path = Path(os.environ["PII_CONFIG_PATH"])
    else:
        for loc in DEFAULT_CONFIG_LOCATIONS:
            if loc.is_file():
                target_path = loc
                break

    if not target_path or not target_path.is_file():
        logger.debug("No configuration file found; using default pipeline parameters.")
        return PipelineConfig()

    logger.info("Loading Stage 9 pipeline configuration from: %s", target_path)
    data = _load_raw_config(target_path)
    
    models_data = data.get("models", {})
    gp_data = models_data.get("gliner_primary", {})
    gv_data = models_data.get("gliner_vanilla_fallback", {})
    in_data = models_data.get("indic_ner", {})
    sp_data = models_data.get("spacy", {})
    rt_data = data.get("runtime", {})
    an_data = data.get("anonymizer", {})
    acceptance_data = data.get("acceptance", {})
    hindi_data = data.get("hindi", {})
    if not isinstance(hindi_data, dict):
        raise ValueError("hindi configuration must be a mapping")
    from .models.indic_scoring import validate_confidence_thresholds, validate_inference_controls
    indic_strategy, indic_temperature, indic_margin = validate_inference_controls(
        in_data.get("decoding_strategy", "argmax"), in_data.get("temperature", 1.0),
        in_data.get("min_word_margin", 0.0))
    indic_thresholds = validate_confidence_thresholds(in_data.get("confidence_thresholds", {}))
    overlap_strategy = in_data.get("overlap_strategy", "best_context")
    if overlap_strategy not in {"first", "best_context"}:
        raise ValueError("IndicNER overlap_strategy must be first or best_context")
    thresholds = acceptance_data.get("score_thresholds", {})
    if not isinstance(thresholds,dict):
        raise ValueError("acceptance.score_thresholds must be a mapping")
    import math
    for key,value in thresholds.items():
        if not isinstance(key,str) or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("Acceptance thresholds require string keys and finite scores between 0 and 1")

    return PipelineConfig(
        acceptance=AcceptanceConfig(score_thresholds=dict(thresholds)),
        hindi=HindiConfig(organization_enabled=_as_bool(hindi_data.get("organization_enabled"), True)),
        models=ModelsConfig(
            gliner_primary=GLiNERConfig(
                model_name_or_path=gp_data.get("model_name_or_path", "models/gliner2-primary"),
                adapter=_as_gliner_adapter(gp_data.get("adapter")),
                entity_schema=_as_entity_schema(gp_data.get("entity_schema")),
                threshold=float(gp_data.get("threshold", 0.95)),
                window_size=int(gp_data.get("window_size", 512)),
                stride=int(gp_data.get("stride", 384)),
                window_stride_multiplier=_as_window_stride_multiplier(gp_data.get("window_stride_multiplier", 1)),
                use_onnx=_as_bool(gp_data.get("use_onnx"), False),
                onnx_encoder_path=gp_data.get("onnx_encoder_path", "onnx_models/gliner_primary/encoder.onnx"),
            ),
            gliner_vanilla_fallback=GLiNERVanillaConfig(
                model_name_or_path=gv_data.get("model_name_or_path", "models/gliner2-primary"),
                adapter=_as_gliner_adapter(gv_data.get("adapter")),
                confirm_threshold=float(gv_data.get("confirm_threshold", 0.40)),
            ),
            indic_ner=IndicNERConfig(
                device=_as_indic_device(in_data.get("device")),
                use_onnx=_as_bool(in_data.get("use_onnx"), False),
                cpu_int8=_as_bool(in_data.get("cpu_int8"), False),
                model_path=in_data.get("model_path", "models/indicner"),
                model_weights_path=in_data.get("model_weights_path"),
                onnx_model_path=in_data.get("onnx_model_path", "onnx_models/indicner.onnx"),
                max_chunk_chars=int(in_data.get("max_chunk_chars", 120)),
                chunk_overlap_chars=int(in_data.get("chunk_overlap_chars", 40)),
                min_indic_fraction=float(in_data.get("min_indic_fraction", 0.5)),
                min_confidence=float(in_data.get("min_confidence", 0.80)),
                confidence_thresholds=indic_thresholds,
                decoding_strategy=indic_strategy,
                temperature=indic_temperature,
                min_word_margin=indic_margin,
                overlap_strategy=overlap_strategy,
                telugu_enrichment=_telugu_enrichment_config(in_data.get("telugu_enrichment")),
                hindi_enrichment=_hindi_enrichment_config(in_data.get("hindi_enrichment")),
            ),
            spacy=SpacyConfig(
                model_name=sp_data.get("model_name", "en_core_web_sm"),
                language=sp_data.get("language", "en"),
            ),
        ),
        runtime=RuntimeConfig(
            device=rt_data.get("device", "auto"),
            cpu_threads=_as_cpu_threads(rt_data.get("cpu_threads"), None),
            enable_profiling=_as_bool(rt_data.get("enable_profiling"), False),
            fallback_to_cpu=_as_bool(rt_data.get("fallback_to_cpu"), True),
            enable_mps=_as_bool(rt_data.get("enable_mps"), False),
            parallel_cpu_gpu=_as_bool(rt_data.get("parallel_cpu_gpu"), False),
        ),
        anonymizer=AnonymizerConfig(
            prefix=an_data.get("prefix", "<"),
            suffix=an_data.get("suffix", ">"),
            indexed=_as_bool(an_data.get("indexed"), True),
        ),
    )
