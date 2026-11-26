"""Regression checks for gliner."""
from __future__ import annotations

from pii_pipeline.model_cache import extract_gliner_entities, resolve_gliner_adapter
from pii_pipeline.models.gliner_vanilla import FallbackCandidate, GLiNER2Vanilla
from unittest.mock import Mock
import pytest
from contextlib import nullcontext
from types import SimpleNamespace
from pii_pipeline.models.gliner_primary import GLiNER2PIIPrimary


class _FakeGLiNER:
    def extract_entities(self, _text, _labels, **_kwargs):
        return {
            "entities": {
                "PERSON": [
                    {"start": 0, "end": 4, "confidence": 0.91},
                    {"start": "bad", "end": 8, "confidence": 0.2},
                ],
                "EMAIL": [{"start": 5, "end": 9, "score": 0.88}],
            }
        }


def test_gliner_entity_contract_normalizes_supported_adapters():
    result = extract_gliner_entities(_FakeGLiNER(), "John mail", ["PERSON", "EMAIL"])

    assert result == {
        "entities": {
            "PERSON": [{"start": 0, "end": 4, "confidence": 0.91}],
            "EMAIL": [{"start": 5, "end": 9, "confidence": 0.88}],
        }
    }


def test_gliner_adapter_resolution_requires_an_explicit_contract_for_unknown_models():
    assert resolve_gliner_adapter("fastino/gliner2.5-multi-v1", "auto") == "boundary"
    assert resolve_gliner_adapter("company/custom-checkpoint", "auto") == "legacy_span"
    assert resolve_gliner_adapter("company/custom-checkpoint", "boundary") == "boundary"


@pytest.mark.parametrize("confidence", [0.1, 0.8])
@pytest.mark.parametrize("onnx", [False, True])
def test_fallback_reuses_repeated_probes_only_within_one_request(confidence, onnx):
    model = Mock()
    model.extract_entities.return_value = {"entities": {"MRN": [
        {"start": 0, "end": 3, "confidence": confidence}]}}
    fallback = GLiNER2Vanilla.__new__(GLiNER2Vanilla)
    fallback._model = None if onnx else model
    fallback._onnx_model = model if onnx else None
    fallback.confirm_threshold = 0.4
    candidates = [FallbackCandidate("MRN", 0, 3, 0.5), FallbackCandidate("MRN", 4, 7, 0.6)]
    results = fallback.confirm("abc abc", candidates)
    assert model.extract_entities.call_count == 1
    assert [r.confirmed for r in results] == [confidence >= 0.4] * 2
    assert [(r.start, r.end, r.score) for r in results] == [(0, 3, 0.5), (4, 7, 0.6)]
    assert fallback.confirm("abc abc", candidates) == results
    assert model.extract_entities.call_count == 2  # No text retained across requests.


def test_fallback_does_not_reuse_errors_or_mix_labels_and_confidence_gates():
    model = Mock()
    model.extract_entities.side_effect = [RuntimeError("temporary failure"), {"entities": {}}, {"entities": {}}]
    fallback = GLiNER2Vanilla.__new__(GLiNER2Vanilla)
    fallback._model, fallback._onnx_model = model, None
    fallback.confirm_threshold = 0.4
    candidates = [FallbackCandidate("MRN", 0, 3, 0.95),  # Skips inference, no cache entry.
                  FallbackCandidate("MRN", 4, 7, 0.5),  # Failure must be retried.
                  FallbackCandidate("MRN", 8, 11, 0.5),
                  FallbackCandidate("MRN", 12, 15, 0.5),  # Reuses negative answer.
                  FallbackCandidate("NPI", 16, 19, 0.5)]
    results = fallback.confirm("abc abc abc abc abc", candidates)
    assert [r.confirmed for r in results] == [True, False, False, False, False]
    assert results[0].source == "vanilla_auto_confirmed"
    assert model.extract_entities.call_count == 3
    assert model.extract_entities.call_args.args[1] == ["NPI"]


@pytest.mark.parametrize("device,onnx,reserved,expected", [
    ("cpu", False, 900, False), ("cuda:0", True, 900, False),
    ("cuda:0", False, 400, False), ("cuda:0", False, 500, False),
    ("cuda:1", False, 900, True),
])
def test_cuda_workspace_cleanup_respects_device_backend_and_pressure(monkeypatch, device, onnx, reserved, expected):
    import torch
    primary = GLiNER2PIIPrimary.__new__(GLiNER2PIIPrimary)
    primary.device, primary._onnx_model = device, object() if onnx else None
    properties = Mock(return_value=SimpleNamespace(total_memory=1000))
    reserved_bytes = Mock(return_value=reserved)
    context = Mock(side_effect=lambda selected: nullcontext())
    reclaim = Mock()
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "memory_reserved", reserved_bytes)
    monkeypatch.setattr(torch.cuda, "device", context)
    monkeypatch.setattr(torch.cuda, "empty_cache", reclaim)
    primary._release_idle_cuda_workspace()
    assert reclaim.call_count == int(expected)
    if device == "cpu" or onnx:
        properties.assert_not_called()
        reserved_bytes.assert_not_called()
    if expected:
        context.assert_called_once_with(device)


def test_workspace_cleanup_keeps_every_window_and_source_offset(monkeypatch):
    from pii_pipeline import model_cache
    calls = []
    model = Mock()
    def extract(*args, **kwargs):
        calls.append("infer")
        return {"entities": {"full_name": [{"start": 0, "end": 1, "confidence": 0.99}]}}
    model.extract_entities.side_effect = extract
    monkeypatch.setattr(model_cache.ModelCache, "get_or_load_gliner", lambda *args, **kwargs: model)
    primary = GLiNER2PIIPrimary(device="cpu", window=4, stride=8)
    monkeypatch.setattr(primary, "_release_idle_cuda_workspace", lambda: calls.append("release"))
    results = primary._analyze_sliding_window("x" * 20, ["full_name"])
    assert calls == ["infer", "release", "infer", "release"]
    assert [(r.start, r.end, r.entity_type, r.score) for r in results] == [
        (0, 1, "full_name", 0.99), (8, 9, "full_name", 0.99)]


def test_window_subsampling_retains_original_contexts_tail_offsets_and_complete_coverage(monkeypatch):
    from pii_pipeline import model_cache
    text = ''.join(chr(0x400 + i) for i in range(35))
    calls = []
    model = Mock()
    def extract(window, *args, **kwargs):
        calls.append((text.index(window), window))
        return {"entities": {"full_name": [{"start": 0, "end": 1, "confidence": 0.99}]}}
    model.extract_entities.side_effect = extract
    monkeypatch.setattr(model_cache.ModelCache, "get_or_load_gliner", lambda *a, **k: model)
    standard = GLiNER2PIIPrimary(device="cpu", window=4, stride=3)
    standard.analyze(text, ["full_name"])
    original = dict(calls)
    calls.clear()
    fast = GLiNER2PIIPrimary(device="cpu", window=4, stride=3, window_stride_multiplier=4)
    result = fast.analyze(text, ["full_name"])
    assert [start for start, _ in calls] == [0, 12, 21]
    assert all(original[start] == window for start, window in calls)
    assert {i for start, window in calls for i in range(start, start+len(window))} == set(range(len(text)))
    assert [(r.start, r.end) for r in result] == [(0, 1), (12, 13), (21, 22)]


@pytest.mark.parametrize("multiplier", [0, -1, True, 1.5])
def test_invalid_window_stride_multiplier_is_rejected(multiplier):
    with pytest.raises(ValueError, match="positive integer"):
        GLiNER2PIIPrimary(window_stride_multiplier=multiplier)


def test_window_subsampling_cannot_leave_unprocessed_gaps():
    with pytest.raises(ValueError, match="complete text coverage"):
        GLiNER2PIIPrimary(window=4, stride=5, window_stride_multiplier=4)


def test_regular_overlap_keeps_name_crossing_previous_window_end(monkeypatch):
    from pii_pipeline import model_cache
    text = "x" * 29 + "Rao Sami" + "x" * 48
    windows = []
    model = Mock()

    def extract(window, *args, **kwargs):
        windows.append(window)
        start = window.find("Rao Sami")
        return {"entities": {"full_name": [] if start < 0 else [
            {"start": start, "end": start + 8, "confidence": .99}]}}

    model.extract_entities.side_effect = extract
    monkeypatch.setattr(model_cache.ModelCache, "get_or_load_gliner", lambda *a, **k: model)
    primary = GLiNER2PIIPrimary(device="cpu", window=8, stride=24)
    result = primary.analyze(text, ["full_name"])
    assert [(r.start, r.end) for r in result] == [(29, 37)]
    assert windows[0] == text[:32]
    assert windows[-1] == text[72:]
