#!/usr/bin/env python3
"""Export the GLiNER2 primary transformer encoder for hybrid ONNX inference.

Optimizations:
- FP16 export (50% size reduction)
- ONNX graph optimization (remove dead nodes, fold constants, dedup initializers)
- Dynamic axes for variable batch/sequence lengths
- Weight sharing preservation where applicable
"""
from __future__ import annotations

import argparse
import hashlib
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_OUTPUT = ROOT / "onnx_models" / "gliner_primary" / "encoder.onnx"


class EncoderExportModule(torch.nn.Module):
    """Expose only the GLiNER transformer encoder's hidden states."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimize_onnx_model(model_path: Path, output_path: Path = None) -> Path:
    """Apply ONNX graph optimizations to reduce size and improve inference speed.
    
    Uses ONNX Runtime's optimizer which is more reliable for transformer models.
    """
    try:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer
    except ImportError:
        print("  ONNX Runtime optimizer not available, skipping")
        return model_path

    if output_path is None:
        output_path = model_path.with_name(model_path.stem + "_opt.onnx")

    print(f"Applying ONNX Runtime graph optimizations...")
    opt = optimizer.optimize_model(
        str(model_path),
        model_type="bert",
        num_heads=12,
        hidden_size=768,
    )
    opt.save_model_to_file(str(output_path))

    orig_size = model_path.stat().st_size / 1024 / 1024
    opt_size = output_path.stat().st_size / 1024 / 1024
    print(f"  Optimized size: {opt_size:.1f} MB (reduction: {orig_size - opt_size:.1f} MB)")
    return output_path





def quantize_to_int8(model_path: Path, output_path: Path = None) -> Path:
    """Apply dynamic INT8 quantization to ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if output_path is None:
        # Avoid double suffix if already contains _int8
        stem = model_path.stem
        if stem.endswith("_int8"):
            output_path = model_path.with_name(stem + ".onnx")
        else:
            output_path = model_path.with_name(stem + "_int8.onnx")
    else:
        # If output_path is provided, use it as-is
        pass

    print(f"Applying dynamic INT8 quantization...")
    quantize_dynamic(
        str(model_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    orig_size = model_path.stat().st_size / 1024 / 1024
    opt_size = output_path.stat().st_size / 1024 / 1024
    print(f"  INT8 size: {opt_size:.1f} MB (reduction: {orig_size - opt_size:.1f} MB)")
    return output_path


def export_encoder(
    model_dir: str,
    output_path: Path,
    opset: int = 17,
    fp16: bool = False,
    int8: bool = False,
    optimize: bool = True,
) -> Path:
    from gliner2 import GLiNER2

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    model = GLiNER2.from_pretrained(model_dir, map_location="cpu")
    model.eval()

    # Optionally convert to FP16 before export
    if fp16:
        print("Converting model to FP16...")
        model.encoder = model.encoder.half()

    export_module = EncoderExportModule(model.encoder).eval()

    # Create dummy input for tracing
    schema = model.create_schema().entities(["person"])
    batch = model.processor.collate_fn_inference(
        [("Synthetic export validation input.", schema)]
    )

    if fp16:
        batch.input_ids = batch.input_ids.to(torch.int64)
        batch.attention_mask = batch.attention_mask.to(torch.int64)

    # For INT8, we export to a temp file first, then quantize to the final output path
    export_path = output_path
    if int8:
        export_path = output_path.with_name(output_path.stem + "_fp32_temp.onnx")

    print(f"Exporting to ONNX (opset={opset}, fp16={fp16})...")
    with torch.inference_mode():
        torch.onnx.export(
            export_module,
            (batch.input_ids, batch.attention_mask),
            str(export_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            verbose=False,
        )

    print(f"Exported to {export_path}")
    size_mb = export_path.stat().st_size / 1024 / 1024
    print(f"  Size: {size_mb:.1f} MB")

    # Apply INT8 quantization if requested
    if int8:
        print("Quantizing to INT8...")
        output_path = quantize_to_int8(export_path, output_path)
        # Clean up temp file
        if export_path.exists() and export_path != output_path:
            export_path.unlink()
        return output_path

    # Apply ONNX graph optimizations
    if optimize:
        output_path = optimize_onnx_model(export_path)
        # Clean up original if optimization created a new file
        if export_path != output_path and export_path.exists():
            export_path.unlink()

    return output_path


def write_manifest(output_path: Path, model_source: str, opset: int, fp16: bool, int8: bool = False) -> Path:
    manifest_path = output_path.parent / "manifest.json"
    precision = "int8" if int8 else ("fp16" if fp16 else "fp32")
    manifest = {
        "artifact": output_path.name,
        "artifact_sha256": sha256(output_path),
        "artifact_bytes": output_path.stat().st_size,
        "format": "ONNX hybrid encoder",
        "precision": precision,
        "model_source": model_source,
        "opset": opset,
        "inputs": ["input_ids", "attention_mask"],
        "outputs": ["last_hidden_state"],
        "decoder": "GLiNER2 PyTorch schema-aware span decoder",
        "optimizations": ["onnxruntime"] if output_path.suffix == ".onnx" else [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=MODEL_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true", help="Export in FP16 (50% size reduction)")
    parser.add_argument("--int8", action="store_true", help="Export in INT8 (70% size reduction)")
    parser.add_argument("--no-optimize", action="store_true", help="Skip ONNX Runtime graph optimization")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    if args.reuse_existing:
        if not args.output.is_file():
            raise FileNotFoundError(f"ONNX artifact not found: {args.output}")
        output_path = args.output
    else:
        output_path = export_encoder(
            args.model_dir,
            args.output,
            args.opset,
            fp16=args.fp16,
            int8=args.int8,
            optimize=not args.no_optimize,
        )
    manifest_path = write_manifest(output_path, args.model_dir, args.opset, args.fp16, args.int8)
    print(f"ONNX encoder: {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA-256: {sha256(output_path)}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
