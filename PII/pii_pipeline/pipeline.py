"""
Stage 9: PII & PHI Detection & Redaction Pipeline:

  Valid Transcript Input (Plain Text or JSON)
    -> Microsoft Presidio Orchestrator + Regex/Checksum
         -> GLiNER2 PII Primary (42-type, no PHI, in-memory cached)
         -> Presidio Medical (PHI)
    -> Label Outside Taxonomy?
         No  -> Resolved -----------------------------         Yes -> GLiNER2 Vanilla (Conditional Fallback) -/
    -> Configured Placeholder Anonymizer Redacts All Detected
    -> Redacted Transcript with Placeholders (Text or exact JSON schema)

In-Memory Residency: Models load once on startup and are cached in memory.
Universal Device Support: Auto-detects ANY available GPU (CUDA 0..N, MPS, CPU).
JSON Support: Preserves exact JSON schema and recursively redacts all string values.
Centralized Config Support: Configurable via config/pipeline_config.yaml or programmatic PipelineConfig.
"""
from __future__ import annotations

import logging
import contextvars
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from presidio_analyzer import RecognizerResult

from .anonymizer import PlaceholderAnonymizer
from .config import PipelineConfig, load_pipeline_config
from .policy.entity_policy import entity_from_result, parse_semantic_tags, resolve_entities
from .models.gliner_vanilla import FallbackCandidate, GLiNER2Vanilla
from .models.indic_ner import IndicNERPipeline
from .model_cache import ModelCache, configure_runtime, auto_detect_device, get_hardware_info
from .orchestrator import PresidioOrchestrator
from .profiling import get_profiler, profile_context, profile_run

# Default entity types to ignore (Presidio built-ins not in 42-type taxonomy)
DEFAULT_LABELS_TO_IGNORE = {
    "CARDINAL", "QUANTITY", "PRODUCT", "MONEY", "EVENT",
    "ORDINAL", "PERCENT", "DATE_TIME", "TIME", "DATE",
    "DURATION", "SET", "MISC", "NRP", "LOCATION", "ORGANIZATION",
    "URL", "CRYPTO",
}

logger = logging.getLogger(__name__)


class Stage9Pipeline:
    def __init__(
        self,
        config: Optional[Union[str, Path, PipelineConfig]] = None,
        language: Optional[str] = None,
        gliner_threshold: Optional[float] = None,
        gliner_model_name: Optional[str] = None,
        gliner_vanilla_model_name: Optional[str] = None,
        gliner_vanilla_confirm_threshold: Optional[float] = None,
        indic_ner_model_path: Optional[str] = None,
        indic_ner_onnx_path: Optional[str] = None,
        spacy_model_name: Optional[str] = None,
        device: Optional[str] = None,
        labels_to_ignore: Optional[set[str]] = None,
        gliner_window: Optional[int] = None,
        gliner_stride: Optional[int] = None,
        use_onnx: Optional[bool] = None,
        onnx_encoder_path: Optional[str] = None,
        onnx_indicner_path: Optional[str] = None,
        *, decider: Any = None,
    ):
        # Resolve configuration
        if isinstance(config, PipelineConfig):
            cfg = config
        elif isinstance(config, (str, Path)):
            cfg = load_pipeline_config(config)
        else:
            cfg = load_pipeline_config()

        # Merge explicit constructor args with loaded config (explicit args take precedence)
        resolved_device_str = device or cfg.runtime.device
        self.config = cfg
        self._analysis_executor: Optional[ThreadPoolExecutor] = None
        configure_runtime(cfg.runtime.cpu_threads, cfg.runtime.enable_mps)
        self.device = auto_detect_device(resolved_device_str)
        self._decider = decider
        if decider is not None:
            self.orchestrator = PresidioOrchestrator(
                language=language or cfg.models.spacy.language,
                spacy_model_name=spacy_model_name or cfg.models.spacy.model_name,
                device=self.device,
                labels_to_ignore=labels_to_ignore,
                hindi_organization_enabled=cfg.hindi.organization_enabled,
                enable_gliner=False,
            )
            self.anonymizer = PlaceholderAnonymizer(
                cfg.anonymizer.prefix, cfg.anonymizer.suffix, cfg.anonymizer.indexed,
            )
            decider.warm()
            return

        res_lang = language or cfg.models.spacy.language
        res_gliner_threshold = gliner_threshold if gliner_threshold is not None else cfg.models.gliner_primary.threshold
        res_gliner_model = gliner_model_name or cfg.models.gliner_primary.model_name_or_path
        res_gliner_adapter = cfg.models.gliner_primary.adapter
        res_gliner_entity_schema = cfg.models.gliner_primary.entity_schema
        res_gliner_window = gliner_window if gliner_window is not None else cfg.models.gliner_primary.window_size
        res_gliner_stride = gliner_stride if gliner_stride is not None else cfg.models.gliner_primary.stride
        res_use_onnx = use_onnx if use_onnx is not None else cfg.models.gliner_primary.use_onnx
        res_onnx_encoder = onnx_encoder_path or cfg.models.gliner_primary.onnx_encoder_path

        res_vanilla_model = gliner_vanilla_model_name or cfg.models.gliner_vanilla_fallback.model_name_or_path
        res_vanilla_adapter = cfg.models.gliner_vanilla_fallback.adapter
        res_vanilla_threshold = gliner_vanilla_confirm_threshold if gliner_vanilla_confirm_threshold is not None else cfg.models.gliner_vanilla_fallback.confirm_threshold

        res_indic_onnx = use_onnx if use_onnx is not None else cfg.models.indic_ner.use_onnx
        res_indic_path = indic_ner_model_path or cfg.models.indic_ner.model_path
        res_indic_weights_path = cfg.models.indic_ner.model_weights_path
        res_indic_onnx_path = onnx_indicner_path or cfg.models.indic_ner.onnx_model_path
        res_spacy_model = spacy_model_name or cfg.models.spacy.model_name

        self.use_onnx = res_use_onnx
        component_settings = {
            "language": res_lang,
            "gliner_threshold": res_gliner_threshold,
            "gliner_model": res_gliner_model,
            "gliner_adapter": res_gliner_adapter,
            "gliner_entity_schema": res_gliner_entity_schema,
            "spacy_model": res_spacy_model,
            "labels_to_ignore": labels_to_ignore or DEFAULT_LABELS_TO_IGNORE,
            "hindi_organization_enabled": cfg.hindi.organization_enabled,
            "gliner_window": res_gliner_window,
            "gliner_stride": res_gliner_stride,
            "gliner_window_stride_multiplier": cfg.models.gliner_primary.window_stride_multiplier,
            "use_onnx": res_use_onnx,
            "onnx_encoder": res_onnx_encoder,
            "vanilla_model": res_vanilla_model,
            "vanilla_adapter": res_vanilla_adapter,
            "vanilla_threshold": res_vanilla_threshold,
            "indic_use_onnx": res_indic_onnx,
            "indic_model_path": res_indic_path,
            "indic_model_weights_path": res_indic_weights_path,
            "indic_onnx_path": res_indic_onnx_path,
            "indic_device": cfg.models.indic_ner.device,
            "indic_cpu_int8": cfg.models.indic_ner.cpu_int8,
        }
        try:
            self._initialize_components(component_settings)
        except Exception:
            if not cfg.runtime.fallback_to_cpu or self.device == "cpu":
                raise
            logger.warning(
                "Pipeline initialization failed on %s; retrying configured models once on CPU.",
                self.device,
                exc_info=True,
            )
            ModelCache.clear_cache()
            self.device = "cpu"
            self._initialize_components(component_settings)

        if (
            cfg.runtime.parallel_cpu_gpu
            and self.device.startswith("cuda")
            and getattr(self.indic_ner, "device", "") == "cpu"
        ):
            # Keep one worker alive for the lifetime of the loaded pipeline.
            # Recreating it for every row adds scheduling and CUDA context cost.
            self._analysis_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pii-primary-gpu"
            )

    def _initialize_components(self, settings: Dict[str, Any]) -> None:
        """Create all stages on ``self.device`` from the already-resolved config."""
        self.orchestrator = PresidioOrchestrator(
            language=settings["language"],
            gliner_threshold=settings["gliner_threshold"],
            gliner_model_name=settings["gliner_model"],
            gliner_adapter=settings["gliner_adapter"],
            entity_schema=settings["gliner_entity_schema"],
            spacy_model_name=settings["spacy_model"],
            device=self.device,
            labels_to_ignore=settings["labels_to_ignore"],
            hindi_organization_enabled=settings.get("hindi_organization_enabled", self.config.hindi.organization_enabled),
            gliner_window=settings["gliner_window"],
            gliner_stride=settings["gliner_stride"],
            gliner_window_stride_multiplier=settings.get("gliner_window_stride_multiplier", 1),
            use_onnx=settings["use_onnx"],
            onnx_encoder_path=settings["onnx_encoder"],
        )
        self.vanilla_fallback = GLiNER2Vanilla(
            model_name=settings["vanilla_model"],
            adapter=settings["vanilla_adapter"],
            confirm_threshold=settings["vanilla_threshold"],
            device=self.device,
            window=settings["gliner_window"],
            stride=settings["gliner_stride"],
            use_onnx=settings["use_onnx"],
            onnx_encoder_path=settings["onnx_encoder"],
        )
        self.anonymizer = PlaceholderAnonymizer(
            prefix=self.config.anonymizer.prefix,
            suffix=self.config.anonymizer.suffix,
            indexed=self.config.anonymizer.indexed,
        )
        self.indic_ner = IndicNERPipeline(
            model_path=settings["indic_model_path"],
            model_weights_path=settings["indic_model_weights_path"],
            use_onnx=settings["indic_use_onnx"],
            onnx_model_path=settings["indic_onnx_path"],
            device=settings.get("indic_device") or self.device,
            max_chunk_chars=self.config.models.indic_ner.max_chunk_chars,
            chunk_overlap_chars=self.config.models.indic_ner.chunk_overlap_chars,
            min_indic_fraction=self.config.models.indic_ner.min_indic_fraction,
            min_confidence=self.config.models.indic_ner.min_confidence,
            confidence_thresholds=self.config.models.indic_ner.confidence_thresholds,
            decoding_strategy=self.config.models.indic_ner.decoding_strategy,
            temperature=self.config.models.indic_ner.temperature,
            min_word_margin=self.config.models.indic_ner.min_word_margin,
            overlap_strategy=self.config.models.indic_ner.overlap_strategy,
            telugu_enrichment=self.config.models.indic_ner.telugu_enrichment,
            hindi_enrichment=self.config.models.indic_ner.hindi_enrichment,
            cpu_int8=settings.get("indic_cpu_int8", False),
        )

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> Stage9Pipeline:
        """Create a Stage9Pipeline instance loaded directly from a YAML/JSON config file."""
        return cls(config=config_path)

    @property
    def inference_backends(self) -> Dict[str, str]:
        """Expose the actual backend selected for every model stage."""
        decider = getattr(self, "_decider", None)
        if decider is not None:
            return {"pplx_decider": decider.inference_backend}
        return {
            "gliner2_pii_primary": self.orchestrator.gliner_primary.inference_backend,
            "gliner2_vanilla_fallback": self.vanilla_fallback.inference_backend,
            "indic_ner": self.indic_ner.inference_backend,
        }

    @profile_run
    def run(self, text: str, enable_profiling: bool = False, *,
            semantic_tags: Any = None, min_entity_score: float = 0.0,
            include_diagnostics: bool = False) -> Dict[str, Any]:
        """Run PII/PHI detection & redaction on a single text string."""
        profiler = get_profiler() if enable_profiling else None
        score_thresholds = getattr(getattr(self.config,"acceptance",None),"score_thresholds",{})
        decider = getattr(self, "_decider", None)
        if decider is not None:
            decider.verify_input(text)

        if not text or not isinstance(text, str):
            imported, issues = parse_semantic_tags(semantic_tags)
            _, rejected = resolve_entities("", imported, min_entity_score)
            result = {
                "input_char_count": len(text) if isinstance(text, str) else 0,
                "entities_detected": [],
                "entity_rejections": issues + rejected,
                "routing_summary": {
                    "resolved_no_fallback_needed": 0,
                    "routed_to_gliner2_vanilla_fallback": 0,
                    "fallback_confirmed": 0,
                    "fallback_unconfirmed_but_kept": 0,
                },
                "model_status": self._model_status(),
                "inference_backends": self.inference_backends,
                "device": self.device,
                "redacted_text": text or "",
            }
            if profiler:
                result["profiling"] = profiler.get_summary()
            if decider is not None:
                result["routing_summary"].update(decider_calls=0, decider_cache_hits=0, decider_failed=0)
            if include_diagnostics:
                result["decision_trace"]={"policy_candidates":imported,"policy_rejections":issues+rejected,"selected_spans":[]}
            return result

        imported, parse_issues = parse_semantic_tags(semantic_tags)
        if decider is None:
            routed, candidates_for_policy, routing_summary, indic_diagnostics = self._legacy_signals(
                text, enable_profiling, include_diagnostics, profiler,
            )
            decider_trace = []
        else:
            routed = self._analyze_primary(text, enable_profiling, profiler)
            with profile_context("decider_stage", profiler=profiler):
                decider_kwargs = {}
                if imported and getattr(decider, "uses_semantic_tags", False):
                    validated_tags, _ = resolve_entities(
                        text, imported, min_entity_score, score_thresholds=score_thresholds,
                    )
                    decider_kwargs["validated_tags"] = validated_tags
                candidates_for_policy, routing_summary, decider_trace = decider.analyze(text, routed, **decider_kwargs)
            indic_diagnostics = {}

        # Candidate preparation and policy resolution have different costs.
        # Keep them separate so the timing report exposes validation/conflict work.
        with profile_context("entity_resolution", profiler=profiler):
            entities_detected, rejected = resolve_entities(text, candidates_for_policy + imported, min_entity_score,
                                                          score_thresholds=score_thresholds)

        # 7. Anonymization/Redaction
        with profile_context("anonymizer", profiler=profiler):
            redacted_text = self.anonymizer.redact_selected(text, entities_detected)

        # 8. Result Assembly
        with profile_context("result_assembly", profiler=profiler):
            entity_rejections = parse_issues + rejected

        result = {
            "input_char_count": len(text),
            "entities_detected": entities_detected,
            "entity_rejections": entity_rejections,
            "routing_summary": routing_summary,
            "model_status": self._model_status(),
            "inference_backends": self.inference_backends,
            "device": self.device,
            "redacted_text": redacted_text,
        }

        if include_diagnostics:
            result["decision_trace"] = {
                "indic_ner":indic_diagnostics,
                "analyzer_candidates":[entity_from_result(r,"analyzer") for r in getattr(routed,"raw_results",[])],
                "routing_rejections":getattr(routed,"rejections",[]),
                "policy_candidates":candidates_for_policy+imported,
                "policy_rejections":entity_rejections,
                "acceptance_thresholds":score_thresholds,
                "selected_spans":[{"start":e["start"],"end":e["end"],"entity_type":e["entity_type"]} for e in entities_detected],
            }
            if decider is not None:
                result["decision_trace"]["decider"] = decider_trace
            else:
                result["decision_trace"]["native_emission_note"] = (
                    "GLiNER candidates below its native emission threshold are not returned; "
                    "use a lower proposal threshold for calibration."
                )

        if profiler:
            result["profiling"] = profiler.get_summary()

        return result

    def _model_status(self):
        if getattr(self, "_decider", None) is not None:
            return {"pplx_decider": "live"}
        return {"gliner2_pii_primary": "live", "gliner2_vanilla_fallback": "live"}

    def _legacy_signals(self, text, enable_profiling, include_diagnostics, profiler):
        indic_diagnostics = {}
        # The primary CUDA detector and CPU IndicNER consume only the input
        # text, so the latency profile can run them concurrently. Copying the
        # context keeps their profiling stages as siblings under this request.
        analysis_future: Optional[Future] = None
        executor = getattr(self, "_analysis_executor", None)
        if executor is not None:
            context = contextvars.copy_context()
            analysis_future = executor.submit(
                context.run, self._analyze_primary, text, enable_profiling, profiler
            )

        # 1. Enrichment Stage: Get IndicNER signals
        try:
            with profile_context("indic_ner_enrichment", profiler=profiler):
                if include_diagnostics and hasattr(self.indic_ner,"enrich_text_with_diagnostics"):
                    indic_entities,indic_diagnostics = self.indic_ner.enrich_text_with_diagnostics(text)
                else:
                    indic_entities = self.indic_ner.enrich_text(text)
        except BaseException:
            # Never leave GPU work running after a failed request.
            if analysis_future is not None:
                try:
                    analysis_future.result()
                except BaseException:
                    pass
            raise

        # Convert IndicNER signals to Presidio RecognizerResult format
        with profile_context("indic_ner_conversion", profiler=profiler):
            indic_results = [
                RecognizerResult(entity_type=e["entity_type"], start=e["start"], end=e["end"], score=e["score"],
                                 recognition_metadata={**e.get("recognition_metadata", {}), "recognizer_name": "IndicNER"})
                for e in indic_entities
            ]

        # 2. Primary Analysis (Presidio Orchestrator + GLiNER2 Primary + Medical)
        if analysis_future is None:
            routed = self._analyze_primary(text, enable_profiling, profiler)
        else:
            routed = analysis_future.result()

        # 3. GLiNER2 Vanilla Fallback (with confidence gate)
        with profile_context("gliner_vanilla_fallback", profiler=profiler):
            candidates = [
                FallbackCandidate(r.entity_type, r.start, r.end, r.score)
                for r in routed.needs_fallback
            ]
            fallback_results = self.vanilla_fallback.confirm(
                text, candidates, skip_high_confidence=0.9
            )

        # 4. Merge all signals (IndicNER + Primary + Fallback)
        with profile_context("signal_merge", profiler=profiler):
            resolved_with_source = [(r, "resolved") for r in routed.resolved]
            original_fallback_results = {(r.entity_type, r.start, r.end): r for r in routed.needs_fallback}
            fallback_with_source = [
                (RecognizerResult(entity_type=f.entity_type, start=f.start, end=f.end, score=f.score,
                                  recognition_metadata={**(original_fallback_results[(f.entity_type, f.start, f.end)].recognition_metadata or {}),
                                                        "confirmation_recognizer": "GLiNER2Vanilla"}), "fallback_routed")
                for f in fallback_results
            ]
            indic_with_source = [(r, "indic_ner") for r in indic_results]
            candidate_results = resolved_with_source + fallback_with_source + indic_with_source

        # 6. Deduplication & Sorting
        with profile_context("deduplication_sort", profiler=profiler):
            candidates_for_policy = [entity_from_result(r, source) for r, source in candidate_results]
            fallback_by_key = {(f.entity_type, f.start, f.end): f for f in fallback_results}
            for entity in candidates_for_policy:
                if entity["source"] == "fallback_routed":
                    f = fallback_by_key[(entity["entity_type"], entity["start"], entity["end"])]
                    entity.update(fallback_confirmed=f.confirmed, fallback_source=f.source)
        routing = {
            "resolved_no_fallback_needed": len(routed.resolved),
            "routed_to_gliner2_vanilla_fallback": len(routed.needs_fallback),
            "fallback_confirmed": sum(1 for f in fallback_results if f.confirmed),
            "fallback_unconfirmed_but_kept": sum(1 for f in fallback_results if not f.confirmed),
        }
        return routed, candidates_for_policy, routing, indic_diagnostics

    def _analyze_primary(self, text: str, enable_profiling: bool, profiler: Any):
        """Run the primary analyzer with a stable profiling boundary."""
        with profile_context("presidio_orchestrator", profiler=profiler):
            return self.orchestrator.analyze(text, enable_profiling=enable_profiling)

    def close(self) -> None:
        """Release the optional persistent analysis worker."""
        executor = getattr(self, "_analysis_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._analysis_executor = None

    @profile_run
    def run_json(
        self,
        json_data: Union[Dict[str, Any], List[Any]],
        enable_profiling: bool = False,
    ) -> Dict[str, Any]:
        """
        Run PII/PHI detection & redaction on an arbitrary JSON data structure.
        """
        profiler = get_profiler() if enable_profiling else None

        all_entities: List[Dict[str, Any]] = []
        routing_totals = {
            "resolved_no_fallback_needed": 0,
            "routed_to_gliner2_vanilla_fallback": 0,
            "fallback_confirmed": 0,
            "fallback_unconfirmed_but_kept": 0,
        }
        total_input_chars = 0
        fields_processed = 0
        all_profiling: List[Dict[str, Any]] = []
        # A JSON request often repeats patient details in several fields.
        # Detection depends on the field's exact text, not its JSON path.
        # Keep reuse local to this call so text never survives the request.
        field_results: Dict[str, Dict[str, Any]] = {}

        def _traverse(node: Any, path: str = "$") -> Any:
            nonlocal total_input_chars, fields_processed
            if isinstance(node, dict):
                return {k: _traverse(v, f"{path}.{k}") for k, v in node.items()}
            elif isinstance(node, list):
                return [_traverse(item, f"{path}[{idx}]") for idx, item in enumerate(node)]
            elif isinstance(node, str):
                fields_processed += 1
                total_input_chars += len(node)
                cache_hit = node in field_results
                if cache_hit:
                    res = field_results[node]
                else:
                    res = self.run(node, enable_profiling=enable_profiling)
                    field_results[node] = res
                for e in res["entities_detected"]:
                    entity_copy = deepcopy(e) if cache_hit else dict(e)
                    entity_copy["json_field_path"] = path
                    all_entities.append(entity_copy)

                for k, v in res["routing_summary"].items():
                    routing_totals[k] = routing_totals.get(k, 0) + v

                if profiler and "profiling" in res:
                    field_profile = res["profiling"]
                    if cache_hit:
                        field_profile = {**field_profile, "total_duration_ms": 0.0, "stage_count": 0, "stages": []}
                    all_profiling.append({"field_index": fields_processed - 1, "cache_hit": cache_hit,
                                          "profiling": field_profile})

                return res["redacted_text"]
            else:
                return node

        try:
            redacted_json = _traverse(json_data)
        finally:
            field_results.clear()

        result = {
            "input_char_count": total_input_chars,
            "fields_processed": fields_processed,
            "entities_detected": all_entities,
            "routing_summary": routing_totals,
            "model_status": self._model_status(),
            "inference_backends": self.inference_backends,
            "device": self.device,
            "redacted_json": redacted_json,
        }

        if profiler and all_profiling:
            result["profiling"] = {
                "per_field": all_profiling,
                "aggregate": profiler.get_summary(),
            }

        return result


# Backward compatibility alias
Stage4Pipeline = Stage9Pipeline
