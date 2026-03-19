"""Download the Whisper model for first-time setup."""
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

from faster_whisper import WhisperModel

print("  Downloading model...")
WhisperModel(
    "large-v3",
    device="cpu",
    compute_type="int8",
    download_root=str(Path.home() / ".cache" / "whisper-models"),
)
print("  Model ready.")
