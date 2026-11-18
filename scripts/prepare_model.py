#!/usr/bin/env python3
"""Download allegro/BiDi-eng-pol, export it to ONNX and quantize it to int8.

This is a developer-only step. Application users receive the finished files inside
the installer and never need Python or an internet connection.

Usage:
    python -m venv .venv
    .venv/bin/pip install torch "transformers>=4.45,<5" "optimum[onnxruntime]" \
        onnx onnxruntime sentencepiece sacremoses
    .venv/bin/python scripts/prepare_model.py --output models/bidi-eng-pol

To rewrite only model.json for existing model files:
    python3 scripts/prepare_model.py --output models/bidi-eng-pol --manifest-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

MODEL_ID = "allegro/BiDi-eng-pol"
RELEASE_ID = "bidi-eng-pol"
RELEASE_VERSION = "1"
MANIFEST_FILE = "model.json"
SOURCE_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "source.spm",
    "target.spm",
    "vocab.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
TOKENIZER_FILES = ("source.spm", "vocab.json")
ONNX_FILES = ("encoder_model.int8.onnx", "decoder_model_merged.int8.onnx")


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path) -> None:
    """Write the model version, file sizes and checksums to the model folder."""
    files = {
        name: {"size": (output / name).stat().st_size, "sha256": sha256(output / name)}
        for name in (*ONNX_FILES, *TOKENIZER_FILES)
    }
    manifest = {"id": RELEASE_ID, "version": RELEASE_VERSION, "files": files}
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_model(output: Path, revision: str) -> None:
    """Create the quantized ONNX and tokenizer files in the output folder."""
    from huggingface_hub import snapshot_download
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.exporters.onnx import main_export
    from optimum.onnx import merge_decoders

    source = Path(snapshot_download(MODEL_ID, revision=revision, allow_patterns=list(SOURCE_FILES)))

    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        export_dir = Path(temp)
        main_export(
            model_name_or_path=str(source),
            output=export_dir,
            task="text2text-generation-with-past",
            library_name="transformers",
        )
        print("Exported:", sorted(p.name for p in export_dir.glob("*.onnx*")))

        quantized = export_dir / "quantized"
        quantized.mkdir()
        for name in ("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"):
            quantize_dynamic(
                model_input=export_dir / name,
                model_output=quantized / name,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )

        # Merge after quantizing: both decoders then hold identical quantized
        # weights, which merge_decoders stores only once.
        shutil.copy2(quantized / "encoder_model.onnx", output / ONNX_FILES[0])
        merge_decoders(
            quantized / "decoder_model.onnx",
            quantized / "decoder_with_past_model.onnx",
            save_path=output / ONNX_FILES[1],
            strict=False,
        )

    for name in ONNX_FILES:
        print(f"{name}: {(output / name).stat().st_size / 1e6:.1f} MB")
    for name in TOKENIZER_FILES:
        shutil.copy2(source / name, output / name)


def main() -> None:
    """Create the model release files in the folder given by --output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    output: Path = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        build_model(output, args.revision)
    write_manifest(output)


if __name__ == "__main__":
    main()
