# OpenEar — Real-time AI captioning and translation for churches
# Copyright (c) 2026 TheRevDrJ
# Licensed under AGPL-3.0 — see LICENSE file for details
"""Download AI models for first-time setup.

Models are stored in the OpenEar install directory so they're accessible
regardless of which Windows user account runs the server.
"""
import os
from pathlib import Path

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

from huggingface_hub import snapshot_download
import onnx_asr

INSTALL_DIR = Path(__file__).parent

# Download Parakeet ASR model (~2.5GB, stored in HuggingFace cache)
print("  Downloading Parakeet ASR model (~2.5GB)...")
onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")
print("  Parakeet model ready.")

# Download NLLB-200 translation model (~3GB, stored in install dir)
nllb_dir = str(INSTALL_DIR / "models" / "nllb-3.3b-ct2")
print("  Downloading NLLB-200 translation model (~3GB)...")
snapshot_download("entai2965/nllb-200-3.3B-ctranslate2", local_dir=nllb_dir)
print("  NLLB translation model ready.")

print("  All models downloaded.")
