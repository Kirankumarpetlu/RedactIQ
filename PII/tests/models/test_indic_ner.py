"""IndicNER chunking, batching, offsets and bounded device execution."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
import torch

from pii_pipeline.config import load_pipeline_config
from pii_pipeline.models.indic_ner import iter_chunks, _akshara_boundary


def _chunks(text, max_chars=120, overlap=40):
    return list(iter_chunks(text, max_chars, overlap))


def test_offsets_map_back_onto_the_source_text():
    text = "ఆ, విజయలక్ష్మి.  ನಂಬರ್ 150661576.  ఎస్.  ఉందా నెంబర్ లోనే?  ఉమ్."
    for offset, chunk in _chunks(text):
        assert text[offset:offset + len(chunk)] == chunk


def test_every_non_delimiter_character_is_covered():
    text = "ఒకటి. రెండు! మూడు?  నాలుగు" * 20
    covered = set()
    for offset, chunk in _chunks(text):
        covered.update(range(offset, offset + len(chunk)))
    missing = [
        i for i, ch in enumerate(text)
        if i not in covered and not ch.isspace() and ch not in ".!?।॥"
    ]
    assert missing == []


def test_long_transcript_is_split_into_sentence_sized_pieces():
    text = "ఇది ఒక పెద్ద వాక్యం. " * 60
    chunks = _chunks(text)
    assert len(chunks) > 1
    assert max(len(c) for _, c in chunks) <= 120


def test_sentence_longer_than_the_budget_is_split_with_overlap():
    # No terminator anywhere, so this is one oversized segment.
    text = "word " * 100
    chunks = _chunks(text, max_chars=60, overlap=20)
    assert len(chunks) > 1
    assert max(len(c) for _, c in chunks) <= 60
    # Overlap means consecutive chunks revisit characters, so a name sitting on
    # a cut still appears whole in the next chunk.
    starts = [o for o, _ in chunks]
    assert any(starts[i + 1] < starts[i] + len(chunks[i][1]) for i in range(len(chunks) - 1))


def test_telugu_overlap_keeps_akshara_and_name_cue_whole():
    text = "అది " * 12 + "నా పేరు రవి" + " ఉంది " * 8
    chunks = _chunks(text, max_chars=20, overlap=8)
    assert any("నా పేరు రవి" in chunk for _, chunk in chunks)
    assert all(_akshara_boundary(text, offset) and
               _akshara_boundary(text, offset + len(chunk))
               for offset, chunk in chunks)


def test_devanagari_danda_terminates_a_sentence():
    text = "एक। दो॥ तीन"
    assert len(_chunks(text)) == 3


def test_empty_and_whitespace_only_text_yield_nothing():
    assert _chunks("") == []
    assert _chunks("   \n  ") == []


def test_text_without_any_terminator_is_still_returned():
    assert [c for _, c in _chunks("no terminator here")] == ["no terminator here"]


def test_non_positive_budget_disables_chunking():
    text = "ఒకటి. రెండు."
    assert _chunks(text, max_chars=0) == [(0, text)]


def test_chunk_settings_are_configurable_and_default_to_recall_first():
    cfg = load_pipeline_config("config/pipeline_config.yaml")
    assert cfg.models.indic_ner.max_chunk_chars == 120
    assert cfg.models.indic_ner.chunk_overlap_chars == 40


class _FakeService:
    """Stands in for an older vendored IndicNER service: per-text analyze only."""

    def __init__(self):
        self.analyze_calls = []
        self.batch_calls = []

    @staticmethod
    def _entities(text):
        # Flag the literal token "NAME" wherever it appears in a chunk.
        out, start = [], text.find("NAME")
        while start != -1:
            out.append({"start": start, "end": start + 4, "type": "PERSON",
                        "confidence": 0.99, "text": "NAME"})
            start = text.find("NAME", start + 1)
        return out

    def analyze(self, text):
        self.analyze_calls.append(text)
        return {"entities": self._entities(text)}


class _FakeBatchService(_FakeService):
    """The current vendored service, which also exposes analyze_batch."""

    def analyze_batch(self, texts):
        self.batch_calls.append(list(texts))
        return [{"entities": self._entities(t)} for t in texts]


def _pipeline_with(service):
    from pii_pipeline.models.indic_ner import IndicNERPipeline
    p = IndicNERPipeline.__new__(IndicNERPipeline)
    p.use_onnx = False
    p.onnx_requested = False
    p._onnx_model = None
    p._local_service = service
    p.max_chunk_chars = 120
    p.chunk_overlap_chars = 40
    p.min_indic_fraction = 0.5
    p.min_confidence = 0.80
    return p


TEXT = "ఒకటి NAME ఉంది. రెండు NAME ఉంది. మూడు ఏమీ లేదు."


def test_batched_and_per_chunk_paths_agree():
    batched = _pipeline_with(_FakeBatchService()).enrich_text(TEXT)
    looped = _pipeline_with(_FakeService()).enrich_text(TEXT)
    assert batched == looped
    assert [e["text"] for e in batched] == ["NAME", "NAME"]


def test_batched_path_issues_one_call_for_all_chunks():
    svc = _FakeBatchService()
    _pipeline_with(svc).enrich_text(TEXT)
    assert len(svc.batch_calls) == 1
    assert svc.analyze_calls == []


def test_diagnostics_explain_skipped_romanized_and_mixed_chunks():
    pipeline = _pipeline_with(_FakeService())
    _, romanized = pipeline.enrich_text_with_diagnostics("naa peru Ravi")
    _, mixed = pipeline.enrich_text_with_diagnostics("నా పేరు Ravi Kumar")
    assert romanized["chunk_ranges"] == []
    assert mixed["chunk_ranges"] == []
    assert all(item["reason"] == "below_min_indic_fraction"
               for item in romanized["skipped_chunks"] + mixed["skipped_chunks"])


def test_service_without_analyze_batch_falls_back_per_chunk():
    svc = _FakeService()
    _pipeline_with(svc).enrich_text(TEXT)
    assert len(svc.analyze_calls) > 1


def test_spans_are_reported_against_the_full_text_not_the_chunk():
    entities = _pipeline_with(_FakeBatchService()).enrich_text(TEXT)
    for e in entities:
        assert TEXT[e["start"]:e["end"]] == e["text"]


def test_latin_script_chunks_are_left_to_gliner_and_spacy():
    """IndicNER tagged "bowel", "giddiness" and "CBC" as PERSON; it should not
    see Latin-script sentences at all."""
    svc = _FakeBatchService()
    latin = "The NAME is here. Bowel and bladder normal. CBC done."
    assert _pipeline_with(svc).enrich_text(latin) == []
    assert svc.batch_calls in ([], [[]])


def test_mixed_script_keeps_only_the_indic_sentences():
    svc = _FakeBatchService()
    mixed = "Bowel normal NAME here. ఒకటి NAME ఉంది."
    entities = _pipeline_with(svc).enrich_text(mixed)
    # Only the Telugu sentence is scored, so only its NAME comes back.
    assert len(entities) == 1
    assert mixed[entities[0]["start"]:entities[0]["end"]] == "NAME"
    assert entities[0]["start"] > mixed.index("ఒకటి")


def model_class():
    path = Path(__file__).resolve().parents[2] / "models/indicner/indicner/model.py"
    spec = importlib.util.spec_from_file_location("bounded_indicner_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IndicNERModel


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    all_special_tokens = []

    def __init__(self):
        self.sizes = []

    def __call__(self, texts, **kwargs):
        self.sizes.append(len(texts))
        return Inputs(input_ids=torch.tensor([[int(t)] for t in texts]),
                      offset_mapping=torch.tensor([[[0, len(t)]] for t in texts]))

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


class Model:
    config = SimpleNamespace(id2label={0: "O", 1: "B-PER"})

    def __call__(self, input_ids):
        logits = torch.stack((torch.zeros_like(input_ids), input_ids), dim=2).float()
        return SimpleNamespace(logits=logits)


def fake_model(device):
    model = object.__new__(model_class())
    model.device = SimpleNamespace(type=device)
    model.tokenizer, model.model = Tokenizer(), Model()
    return model


def test_cpu_int8_and_float32_models_have_separate_cache_entries(monkeypatch):
    cls = model_class()
    calls = []
    monkeypatch.setattr(cls, '_load_model', lambda self, path, device, **options: calls.append(options))
    fp32 = cls('same-weights', device='cpu')
    int8 = cls('same-weights', device='cpu', cpu_int8=True)
    assert int8 is not fp32
    assert cls('same-weights', device='cpu') is fp32
    assert cls('same-weights', device='cpu', cpu_int8=True) is int8
    assert calls == [{'cpu_int8': False}, {'cpu_int8': True}]


def test_cpu_int8_cannot_be_combined_with_onnx():
    from pii_pipeline.models.indic_ner import IndicNERPipeline
    with pytest.raises(ValueError, match='use_onnx must be false'):
        IndicNERPipeline(cpu_int8=True, use_onnx=True)


def test_cpu_int8_rejects_cuda_before_moving_model(monkeypatch):
    from unittest.mock import Mock
    cls = model_class()
    namespace = cls._load_model.__globals__
    model = Mock()
    monkeypatch.setattr(namespace['AutoTokenizer'], 'from_pretrained', Mock())
    monkeypatch.setattr(namespace['AutoModelForTokenClassification'], 'from_pretrained', Mock(return_value=model))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    with pytest.raises(ValueError, match='requires device=cpu'):
        cls('unused-local-model', device='cuda:0', cpu_int8=True)
    model.to.assert_not_called()


def test_cuda_default_is_bounded_without_losing_results():
    chunks = [str(i) for i in range(1, 20)]
    cuda_model, single_model = fake_model("cuda"), fake_model("cpu")
    batched = cuda_model.predict_batch(chunks)
    assert cuda_model.tokenizer.sizes == [8, 8, 3]
    assert batched == single_model.predict_batch(chunks, batch_size=1)
    assert [r[0]["text"] for r in batched] == chunks


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size_is_rejected(size):
    with pytest.raises(ValueError, match="positive integer"):
        fake_model("cpu").predict_batch(["1"], batch_size=size)
