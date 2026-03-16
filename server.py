"""
OpenEar - Live Speech-to-Text Captioning Server
Receives audio chunks over WebSocket, transcribes with faster-whisper,
returns text in real time.
"""

import os
import time
import asyncio
import logging
import tempfile
from pathlib import Path

# Fix CUDA DLL discovery for Microsoft Store Python
try:
    import nvidia.cublas
    cublas_path = os.path.join(os.path.dirname(nvidia.cublas.__path__[0]), "cublas", "bin")
    if os.path.isdir(cublas_path):
        os.add_dll_directory(cublas_path)
except (ImportError, Exception):
    pass

try:
    import nvidia.cudnn
    cudnn_path = os.path.join(os.path.dirname(nvidia.cudnn.__path__[0]), "cudnn", "bin")
    if os.path.isdir(cudnn_path):
        os.add_dll_directory(cudnn_path)
except (ImportError, Exception):
    pass

from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
LANGUAGE = "en"
PORT = 8620

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openear")

app = FastAPI(title="OpenEar")

logger.info(f"Loading Whisper {MODEL_SIZE} model (first run downloads ~3GB)...")
t0 = time.time()
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=str(Path.home() / ".cache" / "whisper-models"),
)
logger.info(f"Model loaded on {DEVICE.upper()} in {time.time() - t0:.1f}s")


def transcribe_audio(audio_bytes: bytes) -> str:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        segments, info = model.transcribe(
            tmp_path,
            beam_size=BEAM_SIZE,
            language=LANGUAGE,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=200,
            ),
            condition_on_previous_text=False,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except:
                pass


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()

            if len(audio_bytes) < 1000:
                continue

            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, transcribe_audio, audio_bytes)

            if text:
                await websocket.send_json({"text": text})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SIZE, "device": DEVICE}


static_dir = Path(__file__).parent / "static"

@app.get("/")
async def serve_index():
    return FileResponse(static_dir / "index.html")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Server running at http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")