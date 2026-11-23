import sys
from pathlib import Path

# Ensure parent directory is in path for EntityReconstructor
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from .model import IndicNERModel
from entity_reconstruction import EntityReconstructor
from pii_pipeline.models.indic_scoring import validate_inference_controls

class IndicNERService:
    def __init__(self, model_path: "str | None" = None, device: str = "auto", *, cpu_int8=False):
        resolved_model_path = model_path or str(parent_dir / "models" / "IndicNER")
        self.model = IndicNERModel(resolved_model_path, device=device, cpu_int8=cpu_int8)
        self.reconstructor = EntityReconstructor()
        self.configure_inference()

    def configure_inference(self, *, decoding_strategy="argmax", temperature=1.0, min_word_margin=0.0):
        self.decoding_strategy, self.temperature, margin = validate_inference_controls(
            decoding_strategy, temperature, min_word_margin)
        self.reconstructor = EntityReconstructor(min_word_margin=margin)

    def analyze_batch(self, texts):
        """Batched counterpart to :meth:`analyze`; reconstruction stays per text."""
        if not texts:
            return []
        raw_batch = self.model.predict_batch(texts, decoding_strategy=self.decoding_strategy,
                                             temperature=self.temperature)
        return [
            self.reconstructor.reconstruct(text, raw)
            for text, raw in zip(texts, raw_batch)
        ]

    def analyze(self, text: str):
        if not text:
            return {"subword_predictions": [], "word_predictions": [], "entities": []}
        
        raw_predictions = self.model.predict(text, decoding_strategy=self.decoding_strategy,
                                              temperature=self.temperature)
        return self.reconstructor.reconstruct(text, raw_predictions)

class _LazyService:
    """Backward-compatible default service that does not load a model on import."""
    _instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = IndicNERService()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


service = _LazyService()
