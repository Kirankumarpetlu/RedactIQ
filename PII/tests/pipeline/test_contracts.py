"""End-to-end span contracts, input formats and API consistency."""
from __future__ import annotations

from types import SimpleNamespace
import csv
from pii_pipeline import json_codec as json

from models.indicner.entity_reconstruction import EntityReconstructor
from presidio_analyzer import RecognizerResult
import pytest

from pii_pipeline.anonymizer import PlaceholderAnonymizer
from pii_pipeline.orchestrator import run_recognizers
from pii_pipeline.policy.entity_policy import parse_semantic_tags, recognizer_name, resolve_entities
from pii_pipeline.models.gliner_primary import GLiNER2PIIPrimary
from pii_pipeline.policy.indic_filters import is_suppressed
from pii_pipeline.recognizers.indic_identifiers import IndicNameContextRecognizer, SpokenPlaceRecognizer


@pytest.mark.parametrize("profile", [False, True])
def test_json_reuses_exact_fields_with_independent_entities_and_request_caches(profile):
    from unittest.mock import Mock
    from tests.helpers import fake_pipeline
    from pii_pipeline.profiling import profile_run

    pipe = fake_pipeline()

    @profile_run
    def detect(text, enable_profiling=False):
        return {"entities_detected": [{"entity_type": "PERSON", "start": 0, "end": len(text),
                                       "recognition_metadata": {"views": [1]}}],
                "routing_summary": {"resolved_no_fallback_needed": 1,
                                    "routed_to_gliner2_vanilla_fallback": 0,
                                    "fallback_confirmed": 0, "fallback_unconfirmed_but_kept": 0},
                "redacted_text": "<NAME_1>"}

    pipe.run = Mock(side_effect=detect)
    source = {"name": "Anya", "nested": ["Anya", "Bea", {"name": "Anya"}, 42, None]}
    result = pipe.run_json(source, enable_profiling=profile)
    assert pipe.run.call_count == 2
    assert result["fields_processed"] == 4 and result["input_char_count"] == 15
    assert result["redacted_json"] == {"name": "<NAME_1>", "nested": ["<NAME_1>", "<NAME_1>", {"name": "<NAME_1>"}, 42, None]}
    assert source["name"] == "Anya"
    entities = result["entities_detected"]
    assert [e["json_field_path"] for e in entities] == ["$.name", "$.nested[0]", "$.nested[1]", "$.nested[2].name"]
    assert result["routing_summary"]["resolved_no_fallback_needed"] == 4
    entities[0]["recognition_metadata"]["views"].append(2)
    assert entities[1]["recognition_metadata"]["views"] == [1]
    if profile:
        fields = result["profiling"]["per_field"]
        assert [f["cache_hit"] for f in fields] == [False, True, False, True]
        assert fields[1]["profiling"]["total_duration_ms"] == 0
        assert fields[3]["profiling"]["stages"] == []
        assert result["profiling"]["aggregate"]["stage_count"] == 3
    pipe.run_json(source, enable_profiling=profile)
    assert pipe.run.call_count == 4
from pii_pipeline.model_cache import extract_gliner_entities
from pii_pipeline.pipeline import Stage9Pipeline

from tests.helpers import fake_pipeline
from pii_pipeline.run_csv_through_stage9 import _load_input_records
from scripts.evaluate_pipeline import PipelineEvaluator
from scripts.prepare_precision_dataset import benchmark_draft, select_heldout, validate_reviewed


def test_cuda_primary_and_cpu_indic_stages_can_overlap_without_changing_output():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from types import SimpleNamespace

    pipe = fake_pipeline()
    pipe.device = "cuda:0"
    primary_started, indic_started = Event(), Event()

    def analyze(text, **kwargs):
        primary_started.set()
        assert indic_started.wait(1), "IndicNER did not overlap primary analysis"
        return SimpleNamespace(resolved=[RecognizerResult("full_name", 0, 4, .99)], needs_fallback=[])

    def enrich(text):
        indic_started.set()
        assert primary_started.wait(1), "primary analysis did not overlap IndicNER"
        return []

    pipe.orchestrator.analyze = analyze
    pipe.indic_ner.enrich_text = enrich
    pipe._analysis_executor = ThreadPoolExecutor(max_workers=1)
    try:
        result = pipe.run("Name")
    finally:
        pipe.close()

    assert result["redacted_text"] == "<NAME_1>"
    assert result["entities_detected"][0]["start"] == 0


def test_parallel_pipeline_drains_primary_work_when_indic_stage_fails():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    pipe = fake_pipeline()
    primary_finished = Event()

    def analyze(text, **kwargs):
        primary_finished.set()
        return pipe.orchestrator_result

    pipe.orchestrator_result = pipe.orchestrator.analyze("Name")
    pipe.orchestrator.analyze = analyze
    pipe.indic_ner.enrich_text = lambda text: (_ for _ in ()).throw(RuntimeError("indic failed"))
    pipe._analysis_executor = ThreadPoolExecutor(max_workers=1)
    try:
        with pytest.raises(RuntimeError, match="indic failed"):
            pipe.run("Name")
        assert primary_finished.is_set()
    finally:
        pipe.close()


def entity(label, start, end, score=0.9, **metadata):
    return dict(entity_type=label, start=start, end=end, score=score, **metadata)


@pytest.mark.parametrize("label", ["PERSON", "person", "full_name", "first_name", "middle_name", "last_name"])
def test_name_aliases_cannot_bypass_suppression(label):
    assert is_suppressed(label, "సరే")
    assert not is_suppressed(label, "Ananya")
    assert not is_suppressed("date_of_birth", "so")


@pytest.mark.parametrize("change,reason", [
    ({"start": -1}, "span_out_of_bounds"), ({"end": 100}, "span_out_of_bounds"),
    ({"start": "0"}, "invalid_offsets"), ({"start": True}, "invalid_offsets"),
    ({"score": float("nan")}, "invalid_score"), ({"score": 1.1}, "invalid_score"),
    ({"text": "different"}, "text_offset_mismatch"),
])
def test_invalid_imports_are_rejected_without_offset_repair(change, reason):
    tags, _ = parse_semantic_tags([dict(tag="person", start=0, end=6, **change)] if "start" not in change and "end" not in change
                                  else [{**dict(tag="person", start=0, end=6), **change}])
    accepted, rejected = resolve_entities("Ananya", tags)
    assert not accepted
    assert rejected[0]["reason"] == reason


def test_invalid_json_and_non_lists_are_reported():
    assert parse_semantic_tags("{")[1][0]["reason"] == "invalid_tag_json"
    assert parse_semantic_tags({"tag": "PERSON"})[1][0]["reason"] == "tags_not_a_list"


def test_precise_provenance_wins_and_identical_sources_survive():
    candidates = [entity("full_name", 0, 17, .99, recognizer="GLiNER2PIIPrimary"),
                  entity("PERSON", 0, 6, .9, recognizer="IndicPersonNameGazetteer", source="resolved"),
                  entity("first_name", 0, 6, .8, source="gemma_tag")]
    selected, rejected = resolve_entities("Ananya feels well", candidates)
    assert len(selected) == 1 and selected[0]["end"] == 6
    assert selected[0]["original_entity_type"] == "PERSON"
    assert selected[0]["canonical_entity_type"] == "PERSON"
    assert selected[0]["supporting_evidence"][0]["source"] == "gemma_tag"
    assert rejected[0]["reason"] == "overlapping_candidate"
    assert PlaceholderAnonymizer().redact_selected("Ananya feels well", selected) == "<PERSON_1> feels well"
    assert resolve_entities("Ananya feels well", list(reversed(candidates)))[0] == selected


@pytest.mark.parametrize("text", ["patient conscious and oriented.", "doctor advised rest.", "డాక్టర్: సరే మేడం."])
def test_role_labels_do_not_capture_ordinary_clinical_speech(text):
    assert IndicNameContextRecognizer().analyze(text, ["PERSON"]) == []


def test_explicit_name_stops_at_sentence_and_speaker_boundaries():
    rec = IndicNameContextRecognizer()
    for text in ["My name is Ananya. Patient feels better.", "Name: Ananya\nDoctor: Continue tablets.", "नाम है अनन्या। मरीज ठीक है।"]:
        results = rec.analyze(text, ["PERSON"])
        assert len(results) == 1
        assert text[results[0].start:results[0].end] in {"Ananya", "अनन्या"}


def test_initials_are_kept_without_crossing_sentences_or_speakers():
    rec = IndicNameContextRecognizer()
    text = "Name: R. S. Kumar. Patient feels better."
    assert [text[r.start:r.end] for r in rec.analyze(text, ["PERSON"])] == ["R. S. Kumar"]
    assert not rec.analyze("Doctor:\nAnanya", ["PERSON"])


def test_from_medication_is_not_an_address_but_self_identification_is():
    rec = SpokenPlaceRecognizer()
    assert not rec.analyze("Symptoms improved from medicine", ["address"])
    text = "I'm from Kukatpalli."
    assert [text[r.start:r.end] for r in rec.analyze(text, ["address"])] == ["Kukatpalli"]


def test_supported_clinical_identity_and_full_address_boundaries():
    rec = IndicNameContextRecognizer()
    text = "Patient Ananya Rao ka mobile +91 90000 00002 hai."
    results = rec.analyze(text, ["PERSON"])
    assert [text[r.start:r.end] for r in results] == ["Ananya Rao"]
    assert results[0].recognition_metadata["boundary_evidence"] == "identity_details"
    assert not rec.analyze("patient conscious and mobile contact available.", ["PERSON"])
    place = SpokenPlaceRecognizer()
    text = "My address is Flat 7, Example Enclave, Pune 411000; DOB 01/01/1990."
    results = place.analyze(text, ["address"])
    accepted, _ = resolve_entities(text, [dict(entity_type=r.entity_type, start=r.start, end=r.end,
            score=r.score, recognizer=r.recognition_metadata["recognizer_name"], recognition_metadata=r.recognition_metadata) for r in results])
    assert [e["text"] for e in accepted] == ["Flat 7, Example Enclave, Pune 411000"]


def test_onnx_decoder_uses_measured_probabilities_and_native_reconstruction():
    import numpy as np
    from pii_pipeline.models.onnx_inference import IndicNERONNX
    class Tokenizer:
        all_special_tokens = []
        def __call__(self, text, **kwargs):
            return {"input_ids": np.array([[1,2]]), "attention_mask": np.array([[1,1]]),
                    "offset_mapping": np.array([[[0,6],[7,10]]])}
        def convert_ids_to_tokens(self, ids):
            return ["Ananya", "Rao"]
    logits = np.log(np.array([[[.1,.8,.1],[.3,.1,.6]]]))
    model = IndicNERONNX.__new__(IndicNERONNX)
    model.tokenizer = Tokenizer()
    model.output_names = ["logits"]
    model.session = SimpleNamespace(run=lambda *a: [logits])
    model.id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
    result = model.predict("Ananya Rao")
    assert len(result) == 1
    assert result[0]["confidence"] == .7  # B/I probabilities summed; weakest word sets span score.
    assert result[0]["label"] == "PERSON"


def raw(start, end, label, confidence=.99, **extra):
    return dict(start=start, end=end, label=label, confidence=confidence, **extra)


def test_outside_evidence_and_missing_coverage_prevent_word_expansion():
    rec = EntityReconstructor()
    assert rec.reconstruct("clinical", [raw(0, 1, "B-PER"), raw(1, 8, "O")])["entities"] == []
    assert rec.reconstruct("clinical", [raw(0, 1, "B-PER")])["entities"] == []
    assert rec.reconstruct("clinical", [raw(0, 1, "B-PER"), raw(0, 1, "B-PER"), raw(1, 8, "O")])["entities"] == []


def test_word_confidence_uses_probabilities_without_consensus_inflation():
    ent = EntityReconstructor().reconstruct("Ananya", [raw(0, 6, "B-PER", .7,
        probabilities={"B-PER": .7, "I-PER": .05, "O": .25})])["entities"][0]
    assert ent["confidence"] == .75


def test_bio_boundaries_and_weak_words_survive_reconstruction():
    rec = EntityReconstructor()
    text = "Ananya Rao"
    separate = rec.reconstruct(text, [raw(0, 6, "B-PER"), raw(7, 10, "B-PER")])["entities"]
    assert len(separate) == 2
    merged = rec.reconstruct(text, [raw(0, 6, "B-PER"), raw(7, 10, "I-PER", .6)])["entities"]
    assert len(merged) == 1 and merged[0]["confidence"] == .6
    assert len(rec.reconstruct("Ananya।Rao", [raw(0, 6, "B-PER"), raw(7, 10, "I-PER")])["entities"]) == 2


def test_threshold_below_native_default_is_forwarded_and_schema_order_is_fixed(monkeypatch):
    calls = []
    class FakeModel:
        def extract_entities(self, text, labels, **kwargs):
            calls.append((labels, kwargs["threshold"]))
            return {"entities": {"first_name": [{"start": 0, "end": 6, "confidence": .1}]}}
    monkeypatch.setattr(GLiNER2PIIPrimary, "load", lambda self: setattr(self, "_model", FakeModel()))
    rec = GLiNER2PIIPrimary(threshold=.05, device="cpu")
    results = rec.analyze("Ananya", ["last_name", "first_name"])
    assert calls[0] == ([e for e in rec.label_schema if e in {"last_name", "first_name"}], .05)
    assert results and recognizer_name(results[0]) == "GLiNER2PIIPrimary"
    assert results[0].analysis_explanation is None


def test_standalone_runs_only_selected_recognizer_and_keeps_competing_spans():
    called = []
    class Rec:
        name, id, supported_entities, is_loaded, score_thresholds = "chosen", "id", ["PERSON"], True, {}
        def analyze(self, text, entities, artifacts):
            called.append(self.name)
            return [RecognizerResult("PERSON", 0, 6, .9), RecognizerResult("PERSON", 0, 10, .95)]
    engine = SimpleNamespace(default_score_threshold=0, _enhance_using_context=lambda t,r,a,s,c: r)
    results = run_recognizers(engine, "Ananya Rao", "en", [Rec()], artifacts=object())
    assert len(results) == 2 and called == ["chosen"]
    assert all(recognizer_name(r) == "chosen" for r in results)




def test_core_and_alternate_runner_report_exactly_applied_entities():
    from scripts.run_batched_csv import BatchedPipelineRunner
    pipe = fake_pipeline()
    text = "సరే Ananya"
    tags = [dict(tag="person", text="Ananya", start=4, end=10), dict(tag="person", start=0, end=3)]
    result = pipe.run(text, semantic_tags=tags)
    runner = BatchedPipelineRunner.__new__(BatchedPipelineRunner)
    runner.pipeline = pipe
    batched = runner.process_batch([text], [tags])[0]
    assert result["redacted_text"] == batched["redacted_text"] == "సరే <NAME_1>"
    assert batched["entities"] == result["entities_detected"]
    assert len(result["entities_detected"]) == 1
    assert result["entities_detected"][0]["source"] == "gemma_tag"


def test_standard_csv_uses_core_output_without_second_redaction(tmp_path, monkeypatch):
    from pii_pipeline import run_csv_through_stage9 as runner
    pipe = fake_pipeline()
    monkeypatch.setattr(runner, "Stage9Pipeline", lambda **kw: pipe)
    monkeypatch.setattr(runner, "configure_runtime", lambda *args: None)
    monkeypatch.setattr(runner, "get_hardware_info", lambda device: {})
    text = "సరే Ananya"
    tags = [dict(tag="person", text="Ananya", start=4, end=10)]
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_index", "context", "semantic_tags"])
        writer.writeheader()
        writer.writerow(dict(row_index="heldout-007", context=text, semantic_tags=json.dumps(tags)))
    output = tmp_path / "output.csv"
    runner.run_pipeline_on_file(str(path), str(output), device="cpu")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    result = pipe.run(text, semantic_tags=tags)
    assert row["redacted_text"] == result["redacted_text"]
    assert json.loads(row["pipeline_entities"]) == result["entities_detected"]
    assert row["row_index"] == "heldout-007"
    runner.run_pipeline_on_file(str(path), str(output), device="cpu")
    with output.open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_api_applies_each_request_configuration_and_same_tag_policy(monkeypatch):
    import asyncio
    monkeypatch.setenv("PII_SERVICE_ROLE", "gateway")
    from pii_pipeline import api_endpoint as api
    monkeypatch.setenv("PII_SERVICE_ROLE", "cpu-worker")
    monkeypatch.setattr(api, "_worker_pipeline", None)
    monkeypatch.setattr(api, "_worker_pipeline_key", None)
    monkeypatch.setattr(api, "get_hardware_info", lambda device: {})
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return fake_pipeline()
    monkeypatch.setattr(api, "Stage9Pipeline", factory)
    for threshold in [.05, .8, .8]:
        request = api.AnalyzeRequest(text="సరే Ananya", gliner_threshold=threshold,
            semantic_tags=[dict(tag="person", text="Ananya", start=4, end=10)])
        response = asyncio.run(api.analyze_endpoint(None, request))
        assert response["redacted_text"] == "సరే <NAME_1>"
        assert len(response["entities_detected"]) == 1
    assert [c["gliner_threshold"] for c in calls] == [.05, .8]


def test_analyze_api_warms_one_pipeline_and_redact_route_is_absent(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PII_SERVICE_ROLE", "gateway")
    from pii_pipeline import api_endpoint as api

    monkeypatch.setenv("PII_SERVICE_ROLE", "cpu-worker")
    monkeypatch.setenv("PII_API_KEY", "test-only-secret")
    monkeypatch.setattr(api, "_MODEL_DEVICE", "cpu")
    monkeypatch.setattr(api, "get_hardware_info", lambda device: {})
    monkeypatch.setattr(api.ModelCache, "clear_cache", lambda: None)
    initialized = []

    def factory(**kwargs):
        initialized.append(kwargs)
        return fake_pipeline()

    monkeypatch.setattr(api, "Stage9Pipeline", factory)
    with TestClient(api.app) as client:
        assert client.get("/docs").status_code == 200
        assert "/redact" not in client.get("/openapi.json").json()["paths"]
        assert client.post("/redact", json={"text": "Name"}).status_code == 404
        monkeypatch.delenv("PII_API_KEY")
        assert client.post("/analyze", json={"text": "Name"},
                           headers={"X-API-Key": "test-only-secret"}).status_code == 503
        monkeypatch.setenv("PII_API_KEY", "test-only-secret")
        assert client.post("/analyze", json={"text": "Name"}).status_code == 401
        assert client.post("/analyze", json={"text": "Name"},
                           headers={"X-API-Key": "wrong"}).status_code == 401
        headers = {"X-API-Key": "test-only-secret"}
        first = client.post("/analyze", json={"text": "Name"}, headers=headers)
        second = client.post("/analyze", json={"text": "Name"}, headers=headers)

    assert first.status_code == second.status_code == 200
    payload = first.json()
    assert payload["redacted_text"] == "<NAME_1>"
    assert [(entity["entity_type"], entity["start"], entity["end"])
            for entity in payload["entities_detected"]] == [("full_name", 0, 4)]
    assert payload["runtime_device"] == "cpu"
    assert len(initialized) == 1
    assert initialized[0]["config"] is api._cfg


def test_live_ui_displays_core_applied_spans():
    # Exercise the real UI callbacks without requiring Gradio/model startup.
    import ast
    import time
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple
    source = Path(__file__).resolve().parents[2] / "gradio_app.py"
    names = {"redact", "_entity_rows", "_coverage", "_summary"}
    nodes = [n for n in ast.parse(source.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name in names]
    pipe = fake_pipeline()
    original_run = pipe.run
    pipe.run = lambda text: original_run(text, semantic_tags=[dict(tag="person", text="Ananya", start=4, end=10)])
    namespace = dict(time=time, get_pipeline=lambda: pipe, Any=Any, Dict=Dict, List=List, Optional=Optional, Tuple=Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    redacted, rows, summary = namespace["redact"]("సరే Ananya")
    assert redacted == "సరే <NAME_1>" and len(rows) == 1
    assert rows[0][2] == "Ananya"
    assert "**1** entities" in summary


def test_heldout_excludes_groups_and_near_duplicates_and_is_seeded():
    rows = [dict(group="used", context="unrelated message"),
            dict(group="copy", context="The patient discusses symptoms and treatment with the clinician."),
            dict(group="a", context="Unique consultation alpha"),
            dict(group="a", context="Followup alpha consultation"),
            dict(group="b", context="Another separate appointment beta")]
    excluded = [dict(group="used", context="previous visit"),
                dict(context="The patient discusses symptoms and treatment with the clinician!")]
    selected = select_heldout(rows, excluded, "group", count=1, seed=42)
    assert {r["group"] for r in selected}.isdisjoint({"used", "copy"})
    assert selected == select_heldout(rows, excluded, "group", count=1, seed=42)
    assert all(r in selected for r in rows if r["group"] == selected[0]["group"])
    with pytest.raises(ValueError, match="Missing"):
        select_heldout([dict(context="text")], [], "group")


def test_freeze_and_evaluation_reject_unreviewed_annotations(tmp_path):
    row = dict(row_index="1", context="Ananya", review_status="pending", annotation_scope="", reviewer="",
               verified_semantic_tags='[]')
    with pytest.raises(ValueError, match="review"):
        validate_reviewed([row])
    path = tmp_path / "sheet.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="reviewed"):
        PipelineEvaluator().load_ground_truth(str(path))
    row.update(review_status="reviewed", annotation_scope="exhaustive", reviewer="human")
    validate_reviewed([row])  # Explicitly reviewed no-PII row is valid.
    row["verified_semantic_tags"] = '[{"entity_type":"PERSON","text":"Ananya","start":1,"end":6}]'
    with pytest.raises(ValueError, match="original text"):
        validate_reviewed([row])


def test_repeated_benchmark_values_are_marked_for_review():
    tags, issues = benchmark_draft(dict(context="age 35, dose 35", pii_dictionary='{"AGE":["35"]}'))
    assert len(tags) == 2 and "2 candidate occurrences" in issues[0]


def test_load_input_records_accepts_csv_json_and_txt(tmp_path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("context,semantic_tags\nJane Doe,[]\n", encoding="utf-8")
    csv_rows, csv_format = _load_input_records(csv_path, None)
    assert csv_format == "csv"
    assert csv_rows == [{"context": "Jane Doe", "semantic_tags": "[]"}]

    json_path = tmp_path / "records.json"
    json_path.write_text(json.dumps([{"text": "Jane Doe"}, "John Doe"]), encoding="utf-8")
    json_rows, json_format = _load_input_records(json_path, 1)
    assert json_format == "json"
    assert json_rows == [{"context": "Jane Doe", "semantic_tags": []}]

    txt_path = tmp_path / "records.txt"
    txt_path.write_text("Jane Doe\n\nJohn Doe\n", encoding="utf-8")
    txt_rows, txt_format = _load_input_records(txt_path, None)
    assert txt_format == "txt"
    assert txt_rows == [
        {"context": "Jane Doe", "semantic_tags": []},
        {"context": "John Doe", "semantic_tags": []},
    ]
