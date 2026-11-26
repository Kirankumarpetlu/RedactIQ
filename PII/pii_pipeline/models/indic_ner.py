"""
IndicNER Pipeline — supports both PyTorch (local) and ONNX Runtime inference.
"""
from __future__ import annotations

import importlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from ..policy.indic_filters import filter_indic_entities, indic_script_fraction

from ..model_cache import auto_detect_device
from ..enrichment.telugu_enrichment import normalize_with_offsets, repair_boundaries, merge_fragments
from ..enrichment.telugu_cues import enrich_cues, has_telugu
from ..enrichment.hindi_enrichment import enrich_candidates as enrich_hindi_candidates, has_devanagari
from ..telugu_utils import is_akshara_boundary, split_telugu_suffix
from .indic_scoring import validate_confidence_thresholds, validate_inference_controls

_LOCAL_INDICNER_ROOT = Path(__file__).resolve().parents[2] / "models" / "indicner"

# Sentence terminators across the scripts in this corpus: Latin punctuation plus
# the Devanagari danda and double danda, and any newline.
_SENTENCE_END = re.compile(r"[.!?।॥\n]+")

# IndicNER labels measured at 0% precision against the real-transcript
# benchmark. Dropped at source rather than suppressed word by word.
DROPPED_LABELS = frozenset({"ORGANIZATION"})


def _akshara_boundary(text: str, index: int) -> bool:
    """Whether a cut at index leaves Indic combining and conjunct forms intact."""
    if index <= 0 or index >= len(text):
        return True
    before, after = text[index - 1], text[index]
    if unicodedata.category(after).startswith("M") or after in "\u200c\u200d":
        return False
    if before in "\u0c4d\u200c\u200d":  # Telugu virama and joiners
        return False
    return True


def iter_chunks(text: str, max_chars: int, overlap: int) -> Iterator[Tuple[int, str]]:
    """Yield ``(offset, chunk)`` pairs of sentence-sized text.

    IndicNER is fine-tuned on single sentences. Handed a full multi-turn
    transcript it returns no entities at all — not a truncation effect, since a
    400-character transcript is only ~157 tokens against a 512-token limit. It
    simply predicts all-O on input far longer than anything it was trained on,
    which silently switched the detector off for almost every production row.

    Splitting on sentence boundaries restores it. Over-long sentences are cut on
    whitespace with an overlap, so a name landing on a cut still appears whole in
    the following chunk.
    """
    if max_chars <= 0:
        yield 0, text
        return

    segments: List[Tuple[int, str]] = []
    pos = 0
    for match in _SENTENCE_END.finditer(text):
        segment = text[pos:match.end()]
        if segment.strip():
            segments.append((pos, segment))
        pos = match.end()
    if pos < len(text) and text[pos:].strip():
        segments.append((pos, text[pos:]))

    for offset, segment in segments:
        if len(segment) <= max_chars:
            yield offset, segment
            continue
        start = 0
        while start < len(segment):
            end = min(start + max_chars, len(segment))
            if end < len(segment):
                space = segment.rfind(" ", start, end)
                if space > start:
                    end = space
                else:
                    while end > start + 1 and not _akshara_boundary(segment, end):
                        end -= 1
                    if not _akshara_boundary(segment, end):
                        while end < len(segment) and not _akshara_boundary(segment, end):
                            end += 1
            chunk = segment[start:end]
            if chunk.strip():
                yield offset + start, chunk
            if end >= len(segment):
                break
            next_start = max(end - overlap, start + 1)
            # Character overlap can begin inside a Telugu syllable or even in
            # the middle of a word. Rewind to the start of that word when it
            # fits in the overlap; otherwise at least keep the akshara whole.
            word_start = segment.rfind(" ", start, next_start)
            if word_start >= start:
                next_start = word_start + 1
            else:
                while next_start > start + 1 and not _akshara_boundary(segment, next_start):
                    next_start -= 1
                if not _akshara_boundary(segment, next_start):
                    next_start = end
            start = next_start


def _load_local_service(package_path: Optional[str], model_weights_path: Optional[str], device: str,
                        *, decoding_strategy="argmax", temperature=1.0, min_word_margin=0.0, cpu_int8=False) -> Any:
    """Load the configured local IndicNER package and model without eager startup work."""
    package_root = Path(package_path or _LOCAL_INDICNER_ROOT).expanduser().resolve()
    if not package_root.is_dir():
        raise RuntimeError(f"Configured IndicNER package path does not exist: {package_root}")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    try:
        service_module = importlib.import_module("indicner.service")
        precision_options = {"cpu_int8": True} if cpu_int8 else {}
        service = service_module.IndicNERService(model_path=model_weights_path, device=device, **precision_options)
        if hasattr(service, "configure_inference"):
            service.configure_inference(decoding_strategy=decoding_strategy, temperature=temperature,
                                        min_word_margin=min_word_margin)
        elif (decoding_strategy, temperature, min_word_margin) != ("argmax", 1.0, 0.0):
            raise RuntimeError("Configured IndicNER package does not support inference tuning")
        return service
    except Exception as exc:
        raise RuntimeError(f"IndicNER could not load from configured path: {package_root}") from exc


class IndicNERPipeline:
    """
    Plug-and-play wrapper to integrate IndicNER into the PII pipeline.
    Supports both local PyTorch model and ONNX Runtime inference.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_weights_path: Optional[str] = None,
        use_onnx: bool = False,
        onnx_model_path: Optional[str] = None,
        device: str = "auto",
        max_chunk_chars: int = 120,
        chunk_overlap_chars: int = 40,
        min_indic_fraction: float = 0.5,
        min_confidence: float = 0.80,
        confidence_thresholds: Optional[Dict[str, float]] = None,
        decoding_strategy: str = "argmax",
        temperature: float = 1.0,
        min_word_margin: float = 0.0,
        overlap_strategy: str = "best_context",
        cpu_int8: bool = False,
        telugu_enrichment: Any = None,
        hindi_enrichment: Any = None,
    ):
        self.decoding_strategy, self.temperature, self.min_word_margin = validate_inference_controls(
            decoding_strategy, temperature, min_word_margin)
        self.confidence_thresholds = validate_confidence_thresholds(
            {} if confidence_thresholds is None else confidence_thresholds)
        if overlap_strategy not in {"first", "best_context"}:
            raise ValueError("IndicNER overlap_strategy must be first or best_context")
        self.overlap_strategy = overlap_strategy
        self.use_onnx = use_onnx
        if cpu_int8 and use_onnx:
            raise ValueError("cpu_int8 selects native PyTorch quantization; use_onnx must be false")
        self.cpu_int8 = cpu_int8
        self.onnx_requested = use_onnx
        self.model_path = model_path
        self.model_weights_path = model_weights_path
        self.onnx_model_path = onnx_model_path
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.min_indic_fraction = min_indic_fraction
        self.min_confidence = min_confidence
        self.telugu_enrichment = telugu_enrichment if telugu_enrichment is not None else {"enabled": True}
        self.hindi_enrichment = hindi_enrichment if hindi_enrichment is not None else {"enabled": True}
        self.device = auto_detect_device(device)
        self._onnx_model = None
        self._local_service = None
        
        if use_onnx:
            try:
                from .onnx_inference import IndicNERONNX
                if onnx_model_path is None:
                    onnx_model_path = str(
                        Path(__file__).resolve().parents[2] / "onnx_models" / "indicner.onnx"
                    )
                self._onnx_model = IndicNERONNX(
                    model_path=onnx_model_path,
                    device="cpu",  # IndicNER ONNX runs on CPU
                )
                self._onnx_model.decoding_strategy = self.decoding_strategy
                self._onnx_model.temperature = self.temperature
                self._onnx_model.min_word_margin = self.min_word_margin
                print(f"IndicNER ONNX loaded: {onnx_model_path}")
            except Exception as e:
                print(f"IndicNER ONNX load failed, falling back to local: {e}")
                self.use_onnx = False
        
        if not self.use_onnx:
            self._local_service = _load_local_service(
                self.model_path, self.model_weights_path, self.device,
                decoding_strategy=self.decoding_strategy, temperature=self.temperature,
                min_word_margin=self.min_word_margin, cpu_int8=cpu_int8)

    def confidence_threshold(self, entity: Dict[str, Any]) -> float:
        enriched = (entity.get("recognition_metadata") or {}).get("enrichment_threshold")
        if enriched is not None:
            return float(enriched)
        from ..policy.entity_policy import candidate_script
        label = entity["entity_type"].casefold()
        overrides = getattr(self, "confidence_thresholds", {})
        return overrides.get(f"{label}:{candidate_script(entity['text'])}",
                             overrides.get(label, self.min_confidence))

    def _telugu_option(self, key: str, default: Any) -> Any:
        settings = getattr(self, "telugu_enrichment", None)
        if isinstance(settings, Mapping):
            return settings.get(key, default)
        return getattr(settings, key, default)

    def _hindi_option(self, key: str, default: Any) -> Any:
        settings = getattr(self, "hindi_enrichment", None)
        if isinstance(settings, Mapping):
            return settings.get(key, default)
        return getattr(settings, key, default)

    def _enrich_telugu_candidates(self, text: str, entities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Post-process existing IndicNER proposals in original-text offsets.

        The layer never creates a proposal from a cue alone. Raw candidates are
        copied to diagnostics before this method runs.
        """
        if not self._telugu_option("enabled", True) or not has_telugu(text):
            return entities, []
        candidates = []
        for original in entities:
            item = dict(original)
            metadata = dict(item.get("recognition_metadata") or {})
            metadata.setdefault("raw_indic_score", item.get("score"))
            metadata.setdefault("raw_acceptance_threshold", self.confidence_threshold(item))
            item["recognition_metadata"] = metadata
            candidates.append(item)
        normalized = normalize_with_offsets(text) if self._telugu_option("normalization", True) else None
        if self._telugu_option("boundary_repair", True):
            candidates = [item if "suffix_stripped" in (item.get("enrichment_actions") or [])
                          else repair_boundaries(text, item) for item in candidates]
        if self._telugu_option("fragment_merge", True):
            candidates = merge_fragments(text, candidates)
        if self._telugu_option("suffix_stem", True):
            for item in candidates:
                kind = {"PERSON": "name", "LOCATION": "place"}.get(str(item.get("entity_type", "")).upper())
                if kind is None:
                    continue
                start, end = item.get("start"), item.get("end")
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
                    continue
                surface = text[start:end]
                stem, suffix = split_telugu_suffix(surface, kind=kind)
                if not suffix:
                    continue
                trimmed = stem.rstrip()
                new_end = start + len(trimmed)
                if new_end <= start or not is_akshara_boundary(text, new_end):
                    continue
                item["end"], item["text"] = new_end, text[start:new_end]
                metadata = dict(item.get("recognition_metadata") or {})
                metadata.update(telugu_stem=trimmed, telugu_suffix=suffix)
                item["recognition_metadata"] = metadata
                item["enrichment_actions"] = list(dict.fromkeys(
                    [*(item.get("enrichment_actions") or []), "suffix_stripped"]))
        configured = self._telugu_option("confidence_thresholds", {}) or {}
        cue_limits = {label: configured[f"{label}:cue"] for label in ("person", "location")
                      if f"{label}:cue" in configured}
        options = {
            "baseline_threshold": self.min_confidence,
            "cue_evidence": self._telugu_option("cue_evidence", True),
            "type_reconciliation": self._telugu_option("type_reconciliation", True),
            "cue_thresholds_enabled": self._telugu_option("confidence_recalibration", True),
            "cue_thresholds": cue_limits,
            "cue_score_lift": self._telugu_option("cue_boost", 0.0),
        }
        candidates = enrich_cues(text, candidates, options=options, normalized=normalized)
        rejected = []
        deduplicated: dict[tuple[int, int, str], dict[str, Any]] = {}
        for item in candidates:
            metadata = dict(item.get("recognition_metadata") or {})
            label = str(item.get("entity_type", "")).lower()
            kind = label if label in {"person", "location"} else None
            cue = metadata.get("telugu_cue_evidence") or {}
            has_matching_cue = bool(cue.get("name" if kind == "person" else "place")) if kind else False
            baseline = float(metadata.get("raw_acceptance_threshold", self.min_confidence))
            if self._telugu_option("confidence_recalibration", True) and kind:
                state = "cue" if has_matching_cue else "no_cue"
                threshold = float(configured.get(f"{kind}:{state}", baseline))
                metadata["enrichment_threshold"] = threshold if has_matching_cue else max(baseline, threshold)
            metadata["enrichment_actions"] = list(item.get("enrichment_actions") or [])
            item["recognition_metadata"] = metadata
            if item.get("enrichment_rejection_reason"):
                rejected.append({**item, "reason": item["enrichment_rejection_reason"]})
                continue
            key = (item["start"], item["end"], item["entity_type"])
            previous = deduplicated.get(key)
            if previous is None or float(item.get("score") or 0) > float(previous.get("score") or 0):
                deduplicated[key] = item
        return list(deduplicated.values()), rejected

    def enrich_text(self, text: str, *, diagnostics: dict | None = None) -> List[Dict[str, Any]]:
        """
        Enriches raw text with IndicNER signals.
        Returns entity signals in pipeline format.
        """
        if not text:
            return []

        # Run per sentence-sized chunk and map every span back onto the full
        # text, deduplicating the spans that the chunk overlap reports twice.
        # IndicNER only contributes on Indic script. On Latin-script passages
        # GLiNER2 and spaCy are both stronger, and IndicNER actively harms the
        # output there — it tagged "bowel", "giddiness", "CBC" and "BP" as
        # PERSON. Skipping those chunks removes that noise and the inference
        # cost with it.
        all_chunks = list(iter_chunks(text, self.max_chunk_chars, self.chunk_overlap_chars))
        chunks = [(offset, chunk) for offset, chunk in all_chunks
                  if indic_script_fraction(chunk) >= self.min_indic_fraction]
        chunk_diagnostics = {} if diagnostics is not None else None
        per_chunk = self._enrich_chunks([chunk for _, chunk in chunks], diagnostics=chunk_diagnostics)

        candidates: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
        views: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
        for (offset, chunk), found in zip(chunks, per_chunk):
            for entity in found:
                entity = dict(entity)
                edge_distance = min(entity["start"], len(chunk)-entity["end"])
                entity["start"] += offset
                entity["end"] += offset
                entity["text"] = text[entity["start"]:entity["end"]]
                key = (entity["start"], entity["end"], entity["entity_type"])
                views.setdefault(key, []).append({"chunk_start": offset, "chunk_end": offset+len(chunk),
                                                  "score": entity["score"], "edge_distance": edge_distance})
                rank = (edge_distance, -offset)
                previous = candidates.get(key)
                if previous is None or (getattr(self, "overlap_strategy", "best_context") == "best_context"
                                        and rank > previous["_context_rank"]):
                    entity["_context_rank"] = rank
                    candidates[key] = entity
        entities = list(candidates.values())
        for entity in entities:
            entity.pop("_context_rank")
            key = (entity["start"], entity["end"], entity["entity_type"])
            entity["recognition_metadata"] = {"recognizer_name": "IndicNER",
                "overlap_views": views[key], "overlap_strategy": getattr(self, "overlap_strategy", "best_context"),
                "probability_margin": entity.get("probability_margin"),
                "acceptance_threshold": self.confidence_threshold(entity)}
        raw_entities = [dict(e) for e in entities]
        enrichment_active = bool(self._telugu_option("enabled", True) and has_telugu(text))
        enrichment_rejections = []
        if enrichment_active:
            entities, enrichment_rejections = self._enrich_telugu_candidates(text, entities)
        hindi_active = bool(self._hindi_option("enabled", True) and has_devanagari(text))
        if hindi_active:
            entities, hindi_rejections = enrich_hindi_candidates(text, entities, options={
                key: self._hindi_option(key, default) for key, default in (
                    ("enabled", True), ("normalization", True), ("boundary_repair", True),
                    ("type_reconciliation", True), ("organization_suffix", True),
                    ("confidence_recalibration", True), ("confidence_thresholds", {}))
            })
            enrichment_rejections.extend(
                {**item, "reason": item.get("enrichment_rejection_reason", "hindi_enrichment_rejected")}
                for item in hindi_rejections
            )
        if diagnostics is not None:
            diagnostics.update(raw_candidates=raw_entities,
                chunk_ranges=[{"start":offset,"end":offset+len(chunk)} for offset,chunk in chunks],
                skipped_chunks=[{"start": offset, "end": offset + len(chunk),
                                 "indic_fraction": indic_script_fraction(chunk),
                                 "reason": "below_min_indic_fraction"}
                                for offset, chunk in all_chunks if indic_script_fraction(chunk) < self.min_indic_fraction],
                min_confidence=self.min_confidence, min_indic_fraction=self.min_indic_fraction,
                confidence_thresholds=getattr(self, "confidence_thresholds", {}),
                decoding_strategy=getattr(self, "decoding_strategy", "argmax"),
                temperature=getattr(self, "temperature", 1.0),
                min_word_margin=getattr(self, "min_word_margin", 0.0),
                token_scores=(chunk_diagnostics or {}).get("token_scores", []),
                rejections=enrichment_rejections + [{**e,"reason":"unsupported_indic_label" if e.get("entity_type") in DROPPED_LABELS and not (hindi_active and has_devanagari(e.get("text", ""))) else
                             "below_indic_confidence" if float(e.get("score") or 0)<self.confidence_threshold(e) else "known_non_name"}
                            for e in entities if (e.get("entity_type") in DROPPED_LABELS and not (hindi_active and has_devanagari(e.get("text", "")))) or
                            float(e.get("score") or 0)<self.confidence_threshold(e) or
                            not filter_indic_entities([e],text)])
            if enrichment_active or hindi_active:
                diagnostics["enriched_candidates"] = [dict(e) for e in entities]

        # IndicNER's ORGANIZATION label carries no signal on these transcripts:
        # 95 spans over the 49-row benchmark, none of them real PII, and the
        # surface forms are ordinary words — "సెట్" (set), "గ్రామ్స్" (grams),
        # "ओपीडी" (OPD). Its PERSON and LOCATION labels do find real names, so
        # only ORGANIZATION is dropped.
        # Now that the model reports a real softmax probability rather than a
        # hardcoded 0.99, its spans can finally be gated. Measured against the
        # real-transcript benchmark the probability separates well: at 0.80 the
        # pipeline keeps 86% of true PII while cutting detections by more than
        # half. Applied here rather than as a global floor, because the date and
        # spoken-place recognizers legitimately score below it.
        entities = [
            e for e in entities
            if (e.get("entity_type") not in DROPPED_LABELS or (hindi_active and has_devanagari(e.get("text", ""))))
            and float(e.get("score") or 0.0) >= self.confidence_threshold(e)
        ]

        entities.sort(key=lambda e: (e["start"], e["end"]))

        # IndicNER scores a discourse particle and a real name almost
        # identically (~0.993), so a confidence threshold cannot separate them.
        # Drop only spans whose whole surface form is a known non-name.
        return filter_indic_entities(entities, text)

    def enrich_text_with_diagnostics(self,text):
        diagnostics={}
        entities=self.enrich_text(text,diagnostics=diagnostics)
        return entities,diagnostics

    @property
    def inference_backend(self) -> str:
        """Return the active backend, including an explicit ONNX fallback state."""
        if self._onnx_model is not None:
            return "onnx"
        if getattr(self, "cpu_int8", False):
            return "pytorch:cpu_int8"
        return "pytorch_fallback" if self.onnx_requested else "pytorch"

    def _enrich_chunks(self, chunks: List[str], *, diagnostics=None) -> List[List[Dict[str, Any]]]:
        """Score every chunk, in one batched pass where the backend allows it.

        Chunking turns a 3,265-row corpus into ~205,000 forward passes of ~50
        tokens each. Issued one at a time that is kernel-launch bound on a GPU,
        so the batched path matters far more on CUDA than on CPU. A backend
        without ``analyze_batch`` (an older vendored package, or ONNX) falls back
        to the per-chunk loop, which is slower but produces the same spans.
        """
        if not chunks:
            return []

        if not (self.use_onnx and self._onnx_model is not None):
            batch_analyze = getattr(self._local_service, "analyze_batch", None)
            if batch_analyze is not None:
                results = batch_analyze(chunks)
                if diagnostics is not None:
                    diagnostics["token_scores"] = [{"chunk_index": i,
                        "subword_predictions": result.get("subword_predictions", []),
                        "word_predictions": result.get("word_predictions", [])}
                        for i, result in enumerate(results)]
                return [self._to_entities(result["entities"]) for result in results]

        if self.use_onnx and self._onnx_model is not None:
            found = []
            for index, chunk in enumerate(chunks):
                found.append(self._enrich_onnx(chunk))
                if diagnostics is not None:
                    diagnostics.setdefault("token_scores", []).append(
                        {"chunk_index": index, **getattr(self._onnx_model, "last_diagnostics", {})})
            return found
        results = [self._local_service.analyze(chunk) for chunk in chunks]
        if diagnostics is not None:
            diagnostics["token_scores"] = [{"chunk_index": i,
                "subword_predictions": result.get("subword_predictions", []),
                "word_predictions": result.get("word_predictions", [])}
                for i, result in enumerate(results)]
        return [self._to_entities(result["entities"]) for result in results]

    def _enrich_onnx(self, text: str) -> List[Dict[str, Any]]:
        """Run IndicNER ONNX inference."""
        results = self._onnx_model.predict(text)
        entities = []
        for ent in results:
            entities.append({
                "start": ent["start"],
                "end": ent["end"],
                "entity_type": ent["label"],
                "score": ent["confidence"],
                "text": ent["text"],
                "source": "indic_ner_onnx",
                "probability_margin": ent.get("probability_margin"),
            })
        return entities

    def _enrich_local(self, text: str) -> List[Dict[str, Any]]:
        """Run local IndicNER (PyTorch)."""
        return self._to_entities(self._local_service.analyze(text)["entities"])

    @staticmethod
    def _to_entities(raw_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert reconstructed IndicNER entities into pipeline signal dicts."""
        return [
            {
                "start": ent["start"],
                "end": ent["end"],
                "entity_type": ent["type"],
                "score": ent["confidence"],
                "text": ent["text"],
                "source": "indic_ner",
                "probability_margin": ent.get("probability_margin"),
            }
            for ent in raw_entities
        ]
