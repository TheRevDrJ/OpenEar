"""
OpenEar v0.2 - Live Speech-to-Text Captioning Server
Captures audio from a system input device, transcribes with faster-whisper,
broadcasts captions to all connected clients in real time.
"""

import os
import io
import time
import asyncio
import logging
import logging.handlers
import tempfile
import threading
import wave
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

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
LANGUAGE = "en"
PORT = 80
SAMPLE_RATE = 16000
CHUNK_DURATION = 3  # seconds

LOG_FILE = Path(__file__).parent / "openear.log"

logger = logging.getLogger("openear")
logger.setLevel(logging.INFO)

# Console handler
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_console)

# Rotating file handler — 5MB per file, keep 5 backups (~25MB max)
_file = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_file)

app = FastAPI(title="OpenEar")

# --- Model loading ---
logger.info(f"Loading Whisper {MODEL_SIZE} model (first run downloads ~3GB)...")
t0 = time.time()
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=str(Path.home() / ".cache" / "whisper-models"),
)
logger.info(f"Model loaded on {DEVICE.upper()} in {time.time() - t0:.1f}s")

# --- State ---
connected_clients: set[WebSocket] = set()
is_capturing = False
selected_device_id: int | None = None
audio_buffer: list[np.ndarray] = []
buffer_lock = threading.Lock()
capture_stream: sd.InputStream | None = None
transcription_task: asyncio.Task | None = None


def get_audio_devices() -> list[dict]:
    """List available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append({
                "id": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
            })
    return inputs


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """Called by sounddevice for each audio block. Accumulates into buffer."""
    if status:
        logger.warning(f"Audio status: {status}")
    with buffer_lock:
        audio_buffer.append(indata.copy())


def transcribe_audio_chunk(audio_data: np.ndarray) -> str:
    """Write audio to a temp WAV file, transcribe with Whisper."""
    tmp_path = None
    try:
        # Convert float32 numpy array to 16-bit PCM WAV
        pcm = (audio_data * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(buf.getvalue())
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
            except Exception:
                pass


async def broadcast(message: dict):
    """Send a message to all connected WebSocket clients."""
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def transcription_loop():
    """Continuously drain the audio buffer, transcribe, and broadcast."""
    loop = asyncio.get_event_loop()
    samples_per_chunk = SAMPLE_RATE * CHUNK_DURATION

    while is_capturing:
        await asyncio.sleep(CHUNK_DURATION)

        with buffer_lock:
            if not audio_buffer:
                continue
            chunk = np.concatenate(audio_buffer)
            audio_buffer.clear()

        # Only transcribe if we have enough audio
        if len(chunk) < SAMPLE_RATE:  # less than 1 second
            continue

        # Flatten to mono if needed
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        text = await loop.run_in_executor(None, transcribe_audio_chunk, chunk)

        if text:
            logger.info(f"Transcribed: {text[:80]}...")
            await broadcast({"type": "transcript", "text": text})


def start_capture(device_id: int):
    """Start capturing audio from the selected device."""
    global is_capturing, capture_stream, selected_device_id

    selected_device_id = device_id
    is_capturing = True

    with buffer_lock:
        audio_buffer.clear()

    capture_stream = sd.InputStream(
        device=device_id,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
    )
    capture_stream.start()
    logger.info(f"Audio capture started on device {device_id}")


def stop_capture():
    """Stop capturing audio."""
    global is_capturing, capture_stream

    is_capturing = False
    if capture_stream:
        capture_stream.stop()
        capture_stream.close()
        capture_stream = None

    with buffer_lock:
        audio_buffer.clear()

    logger.info("Audio capture stopped")


# --- API endpoints ---

@app.get("/api/devices")
async def list_devices():
    return {"devices": get_audio_devices()}


@app.post("/api/start")
async def api_start(body: dict):
    global transcription_task

    device_id = body.get("device_id")
    if device_id is None:
        return {"error": "device_id required"}, 400

    if is_capturing:
        stop_capture()

    start_capture(device_id)
    transcription_task = asyncio.create_task(transcription_loop())
    await broadcast({"type": "status", "capturing": True})
    return {"status": "capturing", "device_id": device_id}


@app.post("/api/stop")
async def api_stop():
    global transcription_task

    stop_capture()
    if transcription_task:
        transcription_task.cancel()
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass
        transcription_task = None

    await broadcast({"type": "status", "capturing": False})
    return {"status": "stopped"}


@app.get("/api/status")
async def api_status():
    return {
        "capturing": is_capturing,
        "device_id": selected_device_id,
        "clients": len(connected_clients),
    }


# --- WebSocket for clients ---

@app.websocket("/ws/captions")
async def websocket_captions(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    # Send current status immediately
    await websocket.send_json({"type": "status", "capturing": is_capturing})

    try:
        while True:
            # Keep connection alive; client doesn't send data
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected ({len(connected_clients)} total)")


# --- Health check ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "capturing": is_capturing,
    }


# --- Static files ---

static_dir = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    return FileResponse(static_dir / "index.html")


@app.get("/admin")
async def serve_admin():
    return FileResponse(static_dir / "admin.html")


app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Server running at http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
