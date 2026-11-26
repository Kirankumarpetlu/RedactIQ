"""Shared synthetic pipeline factory; no model weights or inference are loaded."""
from types import SimpleNamespace

from presidio_analyzer import RecognizerResult

from pii_pipeline.anonymizer import PlaceholderAnonymizer
from pii_pipeline.pipeline import Stage9Pipeline


def fake_pipeline():
    pipe = Stage9Pipeline.__new__(Stage9Pipeline)
    pipe.device = "cpu"
    pipe.config = SimpleNamespace(anonymizer=SimpleNamespace(prefix="<", suffix=">", indexed=True))
    pipe.indic_ner = SimpleNamespace(enrich_text=lambda text: [], inference_backend="fake")
    results = [RecognizerResult("full_name", 0, 4, .99)]  # known filler
    pipe.orchestrator = SimpleNamespace(analyze=lambda text, **kw: SimpleNamespace(resolved=results, needs_fallback=[]),
                                       gliner_primary=SimpleNamespace(inference_backend="fake"))
    pipe.vanilla_fallback = SimpleNamespace(confirm=lambda *a, **k: [], inference_backend="fake")
    pipe.anonymizer = PlaceholderAnonymizer()
    return pipe
