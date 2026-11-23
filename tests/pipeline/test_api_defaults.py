"""The Swagger example must exercise configured inference, without overrides."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from presidio_analyzer import RecognizerResult

from pii_pipeline.config import load_pipeline_config
from tests.helpers import fake_pipeline


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("PII_SERVICE_ROLE", "gateway")
    from pii_pipeline import api_endpoint as api

    monkeypatch.setenv("PII_SERVICE_ROLE", "cpu-worker")
    monkeypatch.setenv("PII_API_KEY", "test-only-secret")
    monkeypatch.setattr(api, "_MODEL_DEVICE", "cpu")
    monkeypatch.setattr(api, "_worker_pipeline", None)
    monkeypatch.setattr(api, "_worker_pipeline_key", None)
    monkeypatch.setattr(api, "_cfg", load_pipeline_config("config/pipeline_config.balanced_latency.yaml"))
    monkeypatch.setattr(api, "get_hardware_info", lambda device: {})
    monkeypatch.setattr(api.ModelCache, "clear_cache", lambda: None)

    def factory(**kwargs):
        pipe = fake_pipeline()
        threshold = kwargs.get("gliner_threshold")
        if threshold is None:
            threshold = kwargs["config"].models.gliner_primary.threshold
        pipe.orchestrator.analyze = lambda text, **kw: SimpleNamespace(
            resolved=[RecognizerResult("full_name", 0, 4, .75)] if .75 >= threshold else [],
            needs_fallback=[],
        )
        return pipe

    monkeypatch.setattr(api, "Stage9Pipeline", factory)
    with TestClient(api.app, headers={"X-API-Key": "test-only-secret"}) as client:
        yield api, client


def test_swagger_example_uses_text_only_and_matches_direct_pipeline(api_client):
    api, client = api_client
    operation = client.get("/openapi.json").json()["paths"]["/analyze"]["post"]
    content = operation["requestBody"]["content"]["application/json"]
    example = content["examples"]["configured_defaults"]["value"]
    assert set(example) == {"text"}
    direct = api.Stage9Pipeline(config=api._cfg, device="cpu").run(example["text"])
    response = client.post("/analyze", json=example)
    assert response.status_code == 200
    result = response.json()
    assert result["redacted_text"] == direct["redacted_text"]
    assert result["entities_detected"] == direct["entities_detected"]
    assert result["inference_settings"] == {
        "gliner_threshold": .5,
        "gliner_threshold_source": "config",
        "min_entity_score": 0.0,
        "semantic_tags_supplied": False,
    }


def test_explicit_override_is_visible_and_cannot_leak_into_default_requests(api_client):
    _, client = api_client
    for override, expected, source in [
        ({}, "<NAME_1>", "config"),
        ({"gliner_threshold": 1.0}, "Name", "request"),
        ({"gliner_threshold": None}, "<NAME_1>", "config"),
        ({"gliner_threshold": .5}, "<NAME_1>", "request"),
        ({}, "<NAME_1>", "config"),
    ]:
        response = client.post("/analyze", json={"text": "Name", **override})
        assert response.status_code == 200
        result = response.json()
        assert result["redacted_text"] == expected
        assert result["inference_settings"]["gliner_threshold_source"] == source
        assert result["inference_settings"]["gliner_threshold"] == (override.get("gliner_threshold") or .5)


@pytest.mark.parametrize("payload", [None, {}, {"text": "Name", "gliner_threshold": 1.1}])
def test_missing_or_invalid_request_is_rejected(api_client, payload):
    _, client = api_client
    assert client.post("/analyze", json=payload).status_code == 422


def test_empty_text_uses_the_same_pipeline_contract(api_client):
    _, client = api_client
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["redacted_text"] == ""
    assert response.json()["entities_detected"] == []


def test_standalone_worker_is_default_and_gateway_is_explicit(api_client, monkeypatch):
    api, _ = api_client
    monkeypatch.delenv("PII_SERVICE_ROLE", raising=False)
    assert api._service_role() == "cpu-worker"
    monkeypatch.setenv("PII_SERVICE_ROLE", "gateway")
    assert api._service_role() == "gateway"
