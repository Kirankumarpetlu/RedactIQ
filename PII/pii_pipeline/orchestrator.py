"""
Microsoft Presidio Orchestrator + Regex/Checksum (Stage 9).

Builds the Presidio AnalyzerEngine, registers:
  - Presidio's built-in regex/checksum recognizers (credit card, IBAN,
    US SSN/ITIN, email, phone, IP, crypto wallet, URL, etc.)
  - GLiNER2PIIPrimary (42-type taxonomy, no PHI, cached in memory)
  - The Presidio Medical PHI recognizers

Integrates with ModelCache to avoid recreating spaCy NLP engines and recognizers.
Auto-detects ANY available GPU device.
Supports ONNX Runtime hybrid inference for GLiNER2.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from .models.gliner_primary import GLiNER2PIIPrimary
from .policy.entity_policy import recognizer_name, entity_from_result, resolve_entities
from .recognizers.medical_recognizers import build_medical_recognizers
from .recognizers.indic_identifiers import build_indic_identifier_recognizers
from .recognizers.telugu_presidio import LabelledIndianPinRecognizer
from .recognizers.hindi_address import HindiAddressRecognizer
from .recognizers.hindi_identity import HindiNameRecognizer, HindiOrganizationRecognizer
from .recognizers.hindi_context import enhance_hindi_results
from .policy.indic_filters import is_clinical_measurement, is_english_model_hallucination, is_suppressed
from .model_cache import ModelCache, auto_detect_device
from .profiling import profile_context, get_profiler, profile_stage
from .taxonomy import is_outside_taxonomy

logger = logging.getLogger(__name__)

_INDIC_SCRIPT = re.compile(r"[\u0900-\u0d7f]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def _unsupported_spacy_indic_result(result: RecognizerResult, text: str) -> bool:
    """Discard English spaCy's Indic-script guesses before context boosts them.

    Context enhancement cannot turn the English model into an Indic NER model.
    Indic-aware recognizers and IndicNER remain free to propose the same span.
    Mixed Latin/Indic spans are left to the shared policy's context checks.
    """
    metadata = result.recognition_metadata or {}
    name = str(metadata.get("recognizer_name", ""))
    if name != "SpacyRecognizer":
        return False
    surface = text[result.start:result.end]
    return bool(_INDIC_SCRIPT.search(surface)) and not _LATIN_LETTER.search(surface)


def run_recognizers(engine, text, language, recognizers=None, artifacts=None, enhance_context=True, *, ignored_labels=()):
    selected = recognizers if recognizers is not None else engine.registry.get_recognizers(
        language=language, all_fields=True, entities=None)
    if ignored_labels:
        selected = [rec for rec in selected if not set(rec.supported_entities).issubset(ignored_labels)]
    if artifacts is None:
        artifacts = engine.nlp_engine.process_text(text, language)
    results = []
    for rec in selected:
        if not rec.is_loaded:
            rec.load()
            rec.is_loaded = True
        for result in rec.analyze(text, list(rec.supported_entities), artifacts):
            metadata = dict(result.recognition_metadata or {})
            metadata.setdefault("recognizer_name", rec.name)
            metadata.setdefault("recognizer_identifier", rec.id)
            result.recognition_metadata = metadata
            if not _unsupported_spacy_indic_result(result, text):
                results.append(result)
    if not results:
        return results
    if enhance_context:
        # Presidio's context enhancement is retained; its overlap removal is
        # deliberately deferred to our shared, source-aware span policy.
        results = engine._enhance_using_context(text, results, artifacts, selected, None)
        results = enhance_hindi_results(text, results)
    thresholds = {rec.id: getattr(rec, "score_thresholds", {}) for rec in selected}
    accepted = []
    for result in results:
        limits = thresholds.get(result.recognition_metadata.get("recognizer_identifier"), {})
        threshold = limits.get(result.entity_type, limits.get("default", engine.default_score_threshold))
        if result.score >= threshold:
            accepted.append(result)
    return accepted



@dataclass
class RoutedResults:
    resolved: List[RecognizerResult]          # "No: Resolved" path
    needs_fallback: List[RecognizerResult]    # "Yes: Custom label" path
    raw_results: List[RecognizerResult] = field(default_factory=list)
    rejections: List[dict] = field(default_factory=list)


# Entity types emitted by Presidio built-ins that are NOT in the 42-type taxonomy
# and should be filtered early to avoid expensive GLiNER2 Vanilla fallback
DEFAULT_LABELS_TO_IGNORE = {
    "CARDINAL", "QUANTITY", "PRODUCT", "MONEY", "EVENT",
    "ORDINAL", "PERCENT", "DATE_TIME", "TIME", "DATE",
    "DURATION", "SET", "MISC", "NRP", "LOCATION", "ORGANIZATION",
    "URL", "CRYPTO",
}


class PresidioOrchestrator:
    def __init__(
        self,
        language: str = "en",
        gliner_threshold: float = 0.5,
        gliner_model_name: Optional[str] = None,
        gliner_adapter: str = "legacy_span",
        entity_schema: str = "pii",
        spacy_model_name: Optional[str] = None,
        device: str = "auto",
        labels_to_ignore: Optional[set[str]] = None,
        gliner_window: int = 512,
        gliner_stride: int = 384,
        use_onnx: bool = False,
        onnx_encoder_path: Optional[str] = None,
        gliner_window_stride_multiplier: int = 1,
        hindi_organization_enabled: bool = True,
        enable_gliner: bool = True,
    ):
        self.language = language
        self.device = auto_detect_device(device)
        self.gliner_threshold = gliner_threshold
        self.labels_to_ignore = labels_to_ignore or DEFAULT_LABELS_TO_IGNORE

        # Retrieve cached spaCy engine (loaded once into memory)
        nlp_engine = ModelCache.get_or_load_spacy_engine(
            language=language,
            device=self.device,
            model_name=spacy_model_name or "en_core_web_sm"
        )

        self.analyzer_engine = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[language],
        )

        # GLiNER2 PII Primary — 42-type taxonomy, no PHI (resident in ModelCache)
        kwargs = {
            "threshold": gliner_threshold,
            "adapter": gliner_adapter,
            "entity_schema": entity_schema,
            "device": self.device,
            "window": gliner_window,
            "stride": gliner_stride,
            "window_stride_multiplier": gliner_window_stride_multiplier,
            "use_onnx": use_onnx,
            "onnx_encoder_path": onnx_encoder_path,
        }
        if gliner_model_name:
            kwargs["model_name"] = gliner_model_name

        self.gliner_primary = GLiNER2PIIPrimary(**kwargs) if enable_gliner else None
        if self.gliner_primary is not None:
            self.analyzer_engine.registry.add_recognizer(self.gliner_primary)

        # Presidio Medical — built-in + regex/checksum recognizers for PHI
        medical_recognizers = build_medical_recognizers()
        for recognizer in medical_recognizers:
            self.analyzer_engine.registry.add_recognizer(recognizer)

        # Indian/Indic structured identifiers. GLiNER2 only fires on Latin
        # script, so without these an Aadhaar, PAN, UPI handle, UPI PIN or
        # labelled MRN is invisible on Devanagari/Telugu/Tamil/Kannada/
        # Bengali/Malayalam rows, where IndicNER is the only other detector.
        indic_identifier_recognizers = build_indic_identifier_recognizers()
        indic_identifier_recognizers.append(LabelledIndianPinRecognizer())
        indic_identifier_recognizers.append(HindiAddressRecognizer())
        hindi_recognizers = [HindiNameRecognizer()]
        if hindi_organization_enabled:
            hindi_recognizers.append(HindiOrganizationRecognizer())
        for recognizer in indic_identifier_recognizers:
            self.analyzer_engine.registry.add_recognizer(recognizer)
        for recognizer in hindi_recognizers:
            self.analyzer_engine.registry.add_recognizer(recognizer)

        # Decorate instance methods once; no input or result values are collected.
        for recognizer in self.analyzer_engine.registry.recognizers:
            if recognizer is self.gliner_primary:
                name = "gliner_primary"
            elif recognizer in medical_recognizers:
                name = "medical." + type(recognizer).__name__
            elif recognizer in indic_identifier_recognizers or recognizer in hindi_recognizers:
                # Several keyed recognizers share one Python class. Use their
                # registered names so stage timings remain individually useful.
                name = "indic_id." + recognizer.name
            else:
                name = "presidio." + type(recognizer).__name__
            recognizer.analyze = profile_stage(name)(recognizer.analyze)

    def analyze(self, text: str, enable_profiling: bool = False) -> RoutedResults:
        """
        Runs the full analyzer (built-in regex/checksum + GLiNER2 primary +
        medical recognizers) and applies the 'Label Outside Taxonomy?'
        routing decision.
        """
        profiler = get_profiler() if enable_profiling else None
        
        with profile_context("analyzer_engine", profiler=profiler):
            with profile_context("nlp_tokenization", profiler=profiler):
                artifacts = self.analyzer_engine.nlp_engine.process_text(text, self.language)
            # return_decision_process carries the recognizer's name on each
            # result, which is what lets the English-only spaCy backbone be told
            # apart from the Indic-aware recognizers below.
            results = run_recognizers(self.analyzer_engine, text, self.language, artifacts=artifacts,
                                      ignored_labels=self.labels_to_ignore)

        with profile_context("taxonomy_routing", profiler=profiler):
            resolved, needs_fallback, rejections = [], [], []
            for r in results:
                if r.entity_type in self.labels_to_ignore:
                    rejections.append({**entity_from_result(r,"analyzer"),"reason":"ignored_label"})
                    continue  # Skip early - not in taxonomy, not useful, would trigger fallback
                span = text[r.start:r.end]
                recognizer = recognizer_name(r)
                # spaCy is English-only and returns free-text spans over Indic
                # script that carry no signal at all. IndicNER and the Indic
                # recognizers are left alone.
                # Use exactly the common validation rules before expensive
                # fallback too. In particular explicit credentials must not
                # be dropped by a separate, weaker measurement check.
                accepted, issues = resolve_entities(text,[entity_from_result(r,"analyzer")])
                needs_decider_evidence = (
                    getattr(self, "gliner_primary", False) is None and issues
                    and issues[0]["reason"] == "unsupported_spacy_mixed_context"
                )
                if not accepted and not needs_decider_evidence:
                    rejections.extend(issues)
                    continue
                (needs_fallback if is_outside_taxonomy(r.entity_type) else resolved).append(r)
        return RoutedResults(resolved=resolved, needs_fallback=needs_fallback,raw_results=results,rejections=rejections)
