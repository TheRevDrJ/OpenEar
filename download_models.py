# OpenEar — Real-time AI captioning and translation for churches
# Copyright (c) 2026 TheRevDrJ
# Licensed under AGPL-3.0 — see LICENSE file for details
"""Download the AI models this machine's mode needs, ahead of the first start.

  captions      the Parakeet speech model only (~2.5 GB)
  translation   Parakeet plus the NLLB-200 translation model (~13 GB more: it is
                stored full-precision and quantized to int8 when loaded)

The mode comes from mode.json, which setup.bat writes immediately before calling
this — the same file and the same rules the server uses (openear_config.py). A
captions-only machine never fetches the translation model: it has no use for it,
and the whole point of that mode is a PC that carries nothing it does not run.

WHERE THEY LAND: the translation model goes in the OpenEar folder (models/), so any
Windows account can use it. Parakeet goes in the Hugging Face cache of the account
that RUNS SETUP — so run setup as the account that will run the server, or the
server downloads Parakeet again on its first start.

CALLED BY: setup.bat. Safe to run by hand; a model already present is not fetched
again.
"""
import os

# CUDA DLL fix for Microsoft Store Python
try:
    import nvidia.cublas
    p = os.path.join(os.path.dirname(nvidia.cublas.__path__[0]), "cublas", "bin")
    if os.path.isdir(p):
        os.add_dll_directory(p)
except Exception:
    pass

try:
    import nvidia.cudnn
    p = os.path.join(os.path.dirname(nvidia.cudnn.__path__[0]), "cudnn", "bin")
    if os.path.isdir(p):
        os.add_dll_directory(p)
except Exception:
    pass

import openear_config


def download_speech_model() -> None:
    """Parakeet — every mode needs it. onnx_asr fetches it on first load."""
    import onnx_asr

    print("  Downloading Parakeet speech model (~2.5GB)...")
    # CPU, as the server loads it. Left to its defaults, onnxruntime tries the GPU
    # providers first and prints a page of errors when their DLLs are absent.
    onnx_asr.load_model(openear_config.PARAKEET_MODEL, providers=["CPUExecutionProvider"])
    print("  Parakeet model ready.")


def download_translation_model() -> None:
    """NLLB-200, at the pinned revision — translation mode only."""
    from huggingface_hub import snapshot_download

    print("  Downloading NLLB-200 translation model (~13GB)...")
    snapshot_download(
        openear_config.NLLB_REPO,
        local_dir=str(openear_config.NLLB_MODEL_DIR),
        revision=openear_config.NLLB_REVISION,
    )
    print("  NLLB translation model ready.")


def main() -> None:
    # No launch flags here: setup.bat has just recorded the machine's choice, and
    # that recorded choice is what this download has to match.
    mode, source = openear_config.resolve_mode([])
    print(f"  Mode: {openear_config.describe(mode)}  (from {source})")

    download_speech_model()

    if mode == openear_config.TRANSLATION:
        download_translation_model()
    else:
        print("  Translation model skipped: this machine is set up for captions only.")

    print("  Done.")


if __name__ == "__main__":
    main()
