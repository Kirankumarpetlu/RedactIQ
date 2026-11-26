"""Pipeline initialization, configuration and device placement."""
from __future__ import annotations

from unittest.mock import patch
from pii_pipeline import json_codec as json
import unittest

import pytest

from pii_pipeline.config import load_pipeline_config
from pii_pipeline.pipeline import Stage9Pipeline


def test_model_and_runtime_selection_load_from_config(tmp_path):
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "gliner_primary": {"model_name_or_path": "local/primary", "adapter": "boundary", "use_onnx": "false"},
                    "gliner_vanilla_fallback": {"model_name_or_path": "local/fallback", "adapter": "legacy_span"},
                    "indic_ner": {
                        "model_path": "local/indicner-package",
                        "model_weights_path": "local/indicner-weights",
                    },
                    "spacy": {"model_name": "custom_spacy", "language": "en"},
                },
                "runtime": {"device": "auto", "cpu_threads": 2, "fallback_to_cpu": "true",
                            "enable_mps": "false", "parallel_cpu_gpu": "true"},
            }
        ),
        encoding="utf-8",
    )

    config = load_pipeline_config(config_path)

    assert config.models.gliner_primary.model_name_or_path == "local/primary"
    assert config.models.gliner_primary.use_onnx is False
    assert config.models.gliner_primary.adapter == "boundary"
    assert config.models.gliner_vanilla_fallback.model_name_or_path == "local/fallback"
    assert config.models.gliner_vanilla_fallback.adapter == "legacy_span"
    assert config.models.indic_ner.model_weights_path == "local/indicner-weights"
    assert config.models.spacy.model_name == "custom_spacy"
    assert config.runtime.device == "auto"
    assert config.runtime.cpu_threads == 2
    assert config.runtime.fallback_to_cpu is True
    assert config.runtime.enable_mps is False
    assert config.runtime.parallel_cpu_gpu is True


def test_cpu_threads_supports_automatic_or_fixed_configuration(tmp_path):
    auto_path = tmp_path / "automatic.json"
    auto_path.write_text(json.dumps({"runtime": {"cpu_threads": "auto"}}), encoding="utf-8")
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps({"runtime": {"cpu_threads": 16}}), encoding="utf-8")

    assert load_pipeline_config(auto_path).runtime.cpu_threads is None
    assert load_pipeline_config(fixed_path).runtime.cpu_threads == 16


def test_default_configuration_requires_a_local_reviewed_gliner_artifact(tmp_path):
    config_path = tmp_path / "minimal.json"
    config_path.write_text("{}", encoding="utf-8")

    config = load_pipeline_config(config_path)

    assert config.models.gliner_primary.model_name_or_path == "models/gliner2-primary"
    assert config.models.gliner_vanilla_fallback.model_name_or_path == "models/gliner2-primary"


class PipelineInitializationTests(unittest.TestCase):
    @patch("pii_pipeline.pipeline.IndicNERPipeline")
    @patch("pii_pipeline.pipeline.GLiNER2Vanilla")
    @patch("pii_pipeline.pipeline.PresidioOrchestrator")
    def test_anonymizer_uses_resolved_pipeline_config(self, _orchestrator, _fallback, _indic) -> None:
        pipeline = Stage9Pipeline.__new__(Stage9Pipeline)
        pipeline.config = load_pipeline_config()
        pipeline.device = "cpu"
        pipeline._initialize_components({
            "language": "en",
            "gliner_threshold": 0.95,
            "gliner_model": "local/model",
            "gliner_adapter": "legacy_span",
            "gliner_entity_schema": "pii",
            "spacy_model": "en_core_web_sm",
            "labels_to_ignore": set(),
            "gliner_window": 512,
            "gliner_stride": 384,
            "use_onnx": False,
            "onnx_encoder": None,
            "vanilla_model": "local/model",
            "vanilla_adapter": "legacy_span",
            "vanilla_threshold": 0.4,
            "indic_use_onnx": False,
            "indic_model_path": "local/indic",
            "indic_model_weights_path": "local/indic/weights.bin",
            "indic_onnx_path": None,
        })

        self.assertEqual(pipeline.anonymizer.prefix, pipeline.config.anonymizer.prefix)
        self.assertEqual(pipeline.anonymizer.suffix, pipeline.config.anonymizer.suffix)
        self.assertEqual(pipeline.anonymizer.indexed, pipeline.config.anonymizer.indexed)


@pytest.mark.parametrize("device", [None, "cpu", "auto", "cuda:0", "mps"])
def test_indic_device_override_loads_and_keeps_default_model_settings(tmp_path, device):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {"indic_ner": {"device": device}}}), encoding="utf-8")
    config = load_pipeline_config(path)
    assert config.models.indic_ner.device == device
    assert config.models.indic_ner.max_chunk_chars == 120
    assert config.runtime.device == "auto"


@pytest.mark.parametrize("device", ["cuda:-1", "gpu", 1, True])
def test_invalid_indic_device_override_is_rejected(tmp_path, device):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {"indic_ner": {"device": device}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="indic_ner.device"):
        load_pipeline_config(path)


def test_telugu_enrichment_defaults_do_not_change_indic_confidence_gate(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    indic = load_pipeline_config(path).models.indic_ner
    settings = indic.telugu_enrichment
    assert settings.enabled is True
    assert settings.cue_boost == 0.0
    assert settings.confidence_thresholds == {}
    assert settings.normalization and settings.boundary_repair and settings.fragment_merge
    assert settings.suffix_stem and settings.cue_evidence
    assert settings.type_reconciliation and settings.confidence_recalibration


def test_telugu_enrichment_switches_and_calibration_load(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {"indic_ner": {"telugu_enrichment": {
        "fragment_merge": False,
        "cue_boost": 0.02,
        "confidence_thresholds": {"person:cue": 0.91, "location:no_cue": 0.96},
    }}}}), encoding="utf-8")
    settings = load_pipeline_config(path).models.indic_ner.telugu_enrichment
    assert settings.fragment_merge is False
    assert settings.boundary_repair is True
    assert settings.cue_boost == 0.02
    assert settings.confidence_thresholds == {"person:cue": 0.91, "location:no_cue": 0.96}


@pytest.mark.parametrize("settings", [
    {"enabled": "false"},
    {"cue_boost": float("nan")},
    {"cue_boost": 1.1},
    {"confidence_thresholds": {"person:cue": -0.1}},
    {"confidence_thresholds": {"age:cue": 0.9}},
    {"confidence_thresholds": []},
    {"unknown": True},
])
def test_invalid_telugu_enrichment_settings_are_rejected(tmp_path, settings):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"models": {"indic_ner": {"telugu_enrichment": settings}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Telugu enrichment|telugu_enrichment"):
        load_pipeline_config(path)
