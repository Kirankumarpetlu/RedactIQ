import os

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pii_pipeline.models.indic_scoring import decode_probabilities, validate_inference_controls

class IndicNERModel:
    _instances = {}

    def __new__(cls, model_path, device="auto", *, cpu_int8=False):
        key = (str(model_path), device, cpu_int8)
        if key not in cls._instances:
            instance = super(IndicNERModel, cls).__new__(cls)
            instance._load_model(model_path, device, cpu_int8=cpu_int8)
            cls._instances[key] = instance
        return cls._instances[key]

    def _load_model(self, model_path, requested_device, *, cpu_int8=False):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        requested_device = requested_device.strip().lower()
        if requested_device == "cpu":
            device_name = "cpu"
        elif requested_device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("PII_DEVICE=mps was requested but MPS is unavailable.")
            device_name = "mps"
        elif requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"PII_DEVICE={requested_device} was requested but CUDA is unavailable.")
            device_name = requested_device
        elif requested_device == "auto":
            device_name = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            raise ValueError("PII_DEVICE must be one of: auto, cpu, mps, cuda, or cuda:<index>.")
        self.device = torch.device(device_name)
        if cpu_int8 and self.device.type != "cpu":
            raise ValueError("IndicNER cpu_int8 requires device=cpu")
        self.model.to(self.device)
        if cpu_int8:
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8)
        print(f"Model loaded on {self.device}")


    def predict_batch(self, texts, batch_size: int | None = None, *,
                      decoding_strategy="argmax", temperature=1.0):
        """Tokenize and score many short texts per forward pass.

        Chunking for sentence-level inference turns one call per transcript into
        tens of thousands of ~50-token calls, which on a GPU is dominated by
        kernel-launch overhead rather than compute. Keep CUDA batches bounded:
        a 64-chunk batch competes with GLiNER for a 4 GB GPU and can spill into
        shared system memory. Every chunk is still scored in its original order.
        """
        decoding_strategy, temperature, _ = validate_inference_controls(decoding_strategy, temperature)
        if not texts:
            return []
        if batch_size is None:
            batch_size = 8 if self.device.type == "cuda" else 64
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        all_results = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs).logits

            probabilities = torch.softmax(outputs / temperature, dim=2)
            input_ids = inputs["input_ids"].tolist()
            offsets_batch = offset_mapping.tolist()
            distributions_batch = probabilities.tolist()

            for text, ids, offsets, distributions in zip(batch, input_ids, offsets_batch, distributions_batch):
                all_results.append(self._token_predictions(text, ids, offsets, distributions, decoding_strategy))
        return all_results

    def _token_predictions(self, text, ids, offsets, distributions, strategy):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        real = [(offset, distribution) for token, offset, distribution in zip(tokens, offsets, distributions)
                if token not in self.tokenizer.all_special_tokens and offset[0] < offset[1]]
        predictions = decode_probabilities([distribution for _, distribution in real],
                                           self.model.config.id2label, strategy)
        return [{"text": text[offset[0]:offset[1]], "start": offset[0], "end": offset[1],
                 "label": self.model.config.id2label[prediction],
                 "confidence": float(distribution[prediction]),
                 "probabilities": {self.model.config.id2label[i]: float(p)
                                   for i, p in enumerate(distribution)}}
                for (offset, distribution), prediction in zip(real, predictions)]

    def predict(self, text: str, *, decoding_strategy="argmax", temperature=1.0):
        decoding_strategy, temperature, _ = validate_inference_controls(decoding_strategy, temperature)
        # Crucial: return_offsets_mapping=True is required for reconstruction
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            return_offsets_mapping=True
        )
        
        # Remove offset_mapping before passing to model
        offset_mapping = inputs.pop("offset_mapping")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        probabilities = torch.softmax(outputs / temperature, dim=2)
        return self._token_predictions(text, inputs["input_ids"][0].tolist(),
                                       offset_mapping[0].tolist(), probabilities[0].tolist(),
                                       decoding_strategy)
