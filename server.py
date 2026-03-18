"""
OpenEar v0.3 - Live Speech-to-Text Captioning Server

Architecture overview:
  1. A sounddevice InputStream continuously captures raw audio from a chosen input device
  2. Audio arrives in small blocks (~100ms) via a callback and accumulates in a thread-safe buffer
  3. Every CHUNK_DURATION seconds, the transcription loop drains the buffer and hands the audio
     to faster-whisper (Whisper AI running on the GPU) for speech-to-text
  4. Transcribed text is broadcast to all connected clients over WebSocket
  5. Clients are display-only — they just show captions. All audio capture happens server-side.

The admin page (admin.html) controls capture start/stop and device selection via REST API.
The client page (index.html) connects via WebSocket and renders incoming text.
"""

VERSION = "0.3.0"

import os
import io
import time
import asyncio
import logging
import logging.handlers
import socket
import tempfile
import threading
import wave
from pathlib import Path

# ============================================================================
# CUDA DLL DISCOVERY FIX
# ============================================================================
# When Python is installed from the Microsoft Store, it runs in a sandboxed
# environment that can't find CUDA DLLs installed via pip (nvidia-cublas-cu12,
# nvidia-cudnn-cu12). We manually tell Windows where those DLLs live so that
# faster-whisper can load the GPU acceleration libraries.
# If the nvidia packages aren't installed (e.g., CPU-only setup), these blocks
# silently do nothing.

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

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_SIZE = "large-v3"     # Whisper model variant — large-v3 is the most accurate
DEVICE = "cuda"             # "cuda" for GPU acceleration, "cpu" for fallback
COMPUTE_TYPE = "float16"    # Half-precision floats — faster on GPU, uses less VRAM
BEAM_SIZE = 5               # Beam search width — higher = more accurate but slower
LANGUAGE = "en"             # Fixed to English (skips language detection overhead)
PORT = 80                   # Default HTTP port — no :port needed in URLs
SAMPLE_RATE = 16000         # 16kHz — what Whisper expects. Audio is resampled to this.
CHUNK_DURATION = 3          # Seconds of audio to accumulate before transcribing.
                            # Shorter = more responsive but less context for Whisper.
                            # 3s is a good balance between latency and accuracy.

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Two log destinations: console (for live monitoring) and a rotating file
# (for post-service review). The file handler caps at 5MB and keeps 5 backups,
# so logs never consume more than ~25MB of disk space.

LOG_FILE = Path(__file__).parent / "openear.log"

logger = logging.getLogger("openear")
logger.setLevel(logging.INFO)

_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_console)

_file = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_file)

# ============================================================================
# FASTAPI APP & WHISPER MODEL
# ============================================================================

app = FastAPI(title="OpenEar")

# Load the Whisper model into GPU memory at startup. This takes a few seconds
# but only happens once. The model stays loaded for the lifetime of the server.
# On first run, it downloads ~3GB of model weights to the cache directory.
logger.info(f"Loading Whisper {MODEL_SIZE} model (first run downloads ~3GB)...")
t0 = time.time()
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=str(Path.home() / ".cache" / "whisper-models"),
)
logger.info(f"Model loaded on {DEVICE.upper()} in {time.time() - t0:.1f}s")

# ============================================================================
# SERVER STATE
# ============================================================================
# These globals track the current state of the server. They're modified by
# the API endpoints and read by the admin page's status polling.

connected_clients: set[WebSocket] = set()   # All active WebSocket connections
is_capturing = False                         # Whether we're currently recording audio
selected_device_id: int | None = None        # Which audio input device is active
audio_buffer: list[np.ndarray] = []          # Raw audio chunks waiting to be transcribed
buffer_lock = threading.Lock()               # Protects audio_buffer (written by audio thread,
                                             # read by async transcription loop)
capture_stream: sd.InputStream | None = None # The active sounddevice input stream
transcription_task: asyncio.Task | None = None  # The running async transcription loop
current_audio_level: float = 0.0             # RMS audio level (0.0-1.0) for the admin meter
audio_clipping: bool = False                 # True if audio peaks are hitting the ceiling


# ============================================================================
# AUDIO DEVICE DISCOVERY
# ============================================================================

def get_audio_devices() -> list[dict]:
    """List available audio input devices, filtering duplicates across APIs.

    On Windows, each physical device appears multiple times — once for each
    audio API (WASAPI, DirectSound, MME). We group by device name and pick
    the best API for each. WASAPI is preferred because it has the lowest
    latency and most reliable behavior.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    # Map each host API index to its human-readable name
    api_names = {i: api["name"] for i, api in enumerate(hostapis)}

    # Group all input-capable devices by their name
    by_name: dict[str, list[dict]] = {}
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            entry = {
                "id": i,                    # sounddevice's internal device index
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
                "api": api_names.get(d["hostapi"], ""),
            }
            by_name.setdefault(d["name"], []).append(entry)

    # For each physical device, keep only the best API version
    api_priority = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}
    inputs = []
    for name, entries in by_name.items():
        entries.sort(key=lambda e: api_priority.get(e["api"], 99))
        best = entries[0]
        inputs.append({
            "id": best["id"],
            "name": best["name"],
            "channels": best["channels"],
            "sample_rate": best["sample_rate"],
        })

    return inputs


# ============================================================================
# AUDIO CAPTURE
# ============================================================================

def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    """Called by sounddevice on a background thread for each audio block (~100ms).

    This runs on a separate thread from the main async loop, which is why we
    use a threading.Lock to safely append to the shared audio_buffer.

    Also computes the RMS (root-mean-square) level for the admin page's
    audio meter. RMS is the standard way to measure "loudness" — it's the
    square root of the average of squared sample values. We scale it up by
    3x because raw RMS values for speech are typically quite small (0.01-0.1),
    and we want the meter to be visually useful.
    """
    global current_audio_level, audio_clipping
    if status:
        # sounddevice reports issues like buffer overflows here
        logger.warning(f"Audio status: {status}")

    # Calculate RMS level for the visual meter on the admin page
    rms = float(np.sqrt(np.mean(indata ** 2)))
    current_audio_level = min(rms * 6.0, 1.0)  # Scale up for visibility, cap at 1.0

    # Clipping detection — if any sample exceeds 70% of max amplitude,
    # the input gain is too hot and risks distortion
    audio_clipping = float(np.max(np.abs(indata))) > 0.70

    # Thread-safe append to the buffer that the transcription loop will drain
    with buffer_lock:
        audio_buffer.append(indata.copy())


# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe_audio_chunk(audio_data: np.ndarray) -> str:
    """Convert a chunk of raw audio into text using Whisper.

    faster-whisper requires a file path (not raw bytes), so we:
    1. Convert the float32 numpy array to 16-bit PCM (standard WAV format)
    2. Write it to a temporary .wav file
    3. Run Whisper inference on it
    4. Clean up the temp file

    VAD (Voice Activity Detection) is enabled to skip silent segments,
    which dramatically reduces hallucination on quiet audio. Without VAD,
    Whisper tends to "hear" words in silence.

    condition_on_previous_text=False prevents Whisper from using its own
    prior output as context, which can cause it to get stuck in loops
    or carry forward mistakes.
    """
    tmp_path = None
    try:
        # float32 [-1.0, 1.0] → int16 [-32767, 32767] (standard PCM encoding)
        pcm = (audio_data * 32767).astype(np.int16)

        # Build a WAV file in memory, then write to disk
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)          # Mono
            wf.setsampwidth(2)          # 2 bytes per sample (16-bit)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        # Write to a temp file (faster-whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(buf.getvalue())
            tmp_path = f.name

        # Run Whisper inference
        segments, info = model.transcribe(
            tmp_path,
            beam_size=BEAM_SIZE,
            language=LANGUAGE,
            vad_filter=True,            # Skip silent regions
            vad_parameters=dict(
                min_silence_duration_ms=300,  # How long silence must be to split
                speech_pad_ms=200,            # Padding around detected speech
            ),
            condition_on_previous_text=False,  # Don't carry context between chunks
        )

        # Combine all detected segments into a single string
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

    finally:
        # Always clean up the temp file, even on error
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


async def broadcast(message: dict):
    """Send a JSON message to every connected WebSocket client.

    If a client has disconnected (broken pipe, network error, etc.), we catch
    the exception and collect it for removal. We can't modify the set while
    iterating it, so we batch the removals.
    """
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)


async def transcription_loop():
    """Main transcription loop — runs as an async task while capturing is active.

    Every CHUNK_DURATION seconds, this:
    1. Drains all accumulated audio from the buffer
    2. Skips if there's less than 1 second of audio (not worth transcribing)
    3. Hands the audio to Whisper via run_in_executor (so the GPU work doesn't
       block the async event loop — other WebSocket messages can still flow)
    4. Broadcasts any resulting text to all clients

    Note on privacy: we log THAT a transcription happened, not WHAT was said.
    Sermon content stays ephemeral — it never hits disk.
    """
    loop = asyncio.get_event_loop()
    samples_per_chunk = SAMPLE_RATE * CHUNK_DURATION

    while is_capturing:
        # Sleep for the chunk duration, letting audio accumulate
        await asyncio.sleep(CHUNK_DURATION)

        # Drain the buffer under lock — this is the handoff point between
        # the audio capture thread and the async transcription loop
        with buffer_lock:
            if not audio_buffer:
                continue
            chunk = np.concatenate(audio_buffer)  # Merge all small blocks into one array
            audio_buffer.clear()

        # Skip tiny chunks — less than 1 second isn't useful for Whisper
        if len(chunk) < SAMPLE_RATE:
            continue

        # If the input device was stereo, average the channels to mono
        # (Whisper only accepts mono audio)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)

        # Run transcription on a thread pool so the GPU inference doesn't
        # block the async event loop (WebSocket connections would stall)
        text = await loop.run_in_executor(None, transcribe_audio_chunk, chunk)

        if text:
            logger.info("Transcription sent to clients")
            await broadcast({"type": "transcript", "text": text})


# ============================================================================
# CAPTURE CONTROL
# ============================================================================

def start_capture(device_id: int):
    """Open an audio input stream on the chosen device and begin recording.

    The InputStream runs on its own thread managed by sounddevice/PortAudio.
    It calls audio_callback() for every ~100ms block of audio. We don't
    process audio here — we just accumulate it in the buffer for the
    transcription loop to consume.
    """
    global is_capturing, capture_stream, selected_device_id

    selected_device_id = device_id
    is_capturing = True

    # Clear any stale audio from a previous capture session
    with buffer_lock:
        audio_buffer.clear()

    capture_stream = sd.InputStream(
        device=device_id,
        samplerate=SAMPLE_RATE,   # Resample to 16kHz (what Whisper expects)
        channels=1,               # Mono capture
        dtype="float32",          # Samples as floats in [-1.0, 1.0]
        callback=audio_callback,  # Called on the audio thread for each block
        blocksize=int(SAMPLE_RATE * 0.1),  # 1600 samples = 100ms blocks
    )
    capture_stream.start()
    logger.info(f"Audio capture started on device {device_id}")


def stop_capture():
    """Stop the audio stream and clean up."""
    global is_capturing, capture_stream

    is_capturing = False
    if capture_stream:
        capture_stream.stop()
        capture_stream.close()
        capture_stream = None

    # Discard any unprocessed audio
    with buffer_lock:
        audio_buffer.clear()

    logger.info("Audio capture stopped")


# ============================================================================
# REST API ENDPOINTS
# ============================================================================
# These are called by the admin page (admin.html) to control the server.
# The client page (index.html) doesn't use REST — it only uses WebSocket.

@app.get("/api/devices")
async def list_devices():
    """Return a list of available audio input devices for the admin dropdown."""
    return {"devices": get_audio_devices()}


@app.post("/api/start")
async def api_start(body: dict):
    """Start capturing audio from the specified device and begin transcription.

    If already capturing, stops the current session first (allows switching
    devices without a separate stop call).
    """
    global transcription_task

    device_id = body.get("device_id")
    if device_id is None:
        return {"error": "device_id required"}, 400

    # Stop any existing capture before starting a new one
    if is_capturing:
        stop_capture()

    start_capture(device_id)

    # Launch the transcription loop as an async task running alongside
    # the web server — it will keep running until stop is called
    transcription_task = asyncio.create_task(transcription_loop())

    # Notify all connected clients that captioning is now active
    await broadcast({"type": "status", "capturing": True})
    return {"status": "capturing", "device_id": device_id}


@app.post("/api/stop")
async def api_stop():
    """Stop audio capture and transcription."""
    global transcription_task

    stop_capture()

    # Cancel the transcription loop task and wait for it to finish
    if transcription_task:
        transcription_task.cancel()
        try:
            await transcription_task
        except asyncio.CancelledError:
            pass
        transcription_task = None

    # Notify all clients that captioning has stopped — they'll show
    # the "OpenEar Disabled" banner
    await broadcast({"type": "status", "capturing": False})
    return {"status": "stopped"}


@app.get("/api/status")
async def api_status():
    """Return current server state. Polled by the admin page every 500ms
    to update the UI (audio level meter, client count, capture state).
    """
    return {
        "capturing": is_capturing,
        "device_id": selected_device_id,
        "clients": len(connected_clients),
        "audio_level": round(current_audio_level, 3),
        "clipping": audio_clipping,
    }


@app.get("/api/server-info")
async def server_info():
    """Return server hostname, LAN IP, and the .local URL for QR code generation.

    The IP detection uses a UDP socket trick: we "connect" to a public IP
    (8.8.8.8 / Google DNS) without actually sending any data. The OS picks
    the network interface that would route to that destination, and we read
    back our local IP from it. This reliably finds the real LAN IP even when
    Docker or WSL virtual adapters are present (which would otherwise get
    picked by gethostbyname).

    The .local URL uses mDNS (Bonjour) — iOS, macOS, and most modern systems
    resolve these automatically. This means users can type a friendly hostname
    instead of an IP address.
    """
    hostname = socket.gethostname()
    ip = "unknown"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
    except Exception:
        # Fallback if there's no default route (air-gapped network?)
        try:
            ip = socket.gethostbyname(hostname)
        except Exception:
            pass
    return {
        "hostname": hostname,
        "ip": ip,
        "url": f"http://{hostname.lower()}.local",
        "port": PORT,
    }


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/captions")
async def websocket_captions(websocket: WebSocket):
    """WebSocket connection handler for client devices.

    Each client (phone/tablet viewing captions) maintains a persistent
    WebSocket connection. When a new client connects:
    1. We accept the connection and add it to our set of clients
    2. We immediately send the current capture state so the client knows
       whether to show "Waiting for captions..." or "OpenEar Disabled"
    3. We keep the connection open by waiting for messages (the client
       doesn't actually send any, but WebSocket requires us to read)
    4. When the client disconnects, we remove it from the set

    The broadcast() function sends transcribed text to all clients in this set.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    # Tell the new client whether captioning is currently active
    await websocket.send_json({"type": "status", "capturing": is_capturing})

    try:
        while True:
            # Block here waiting for client messages (keeps connection alive).
            # Clients don't send data, but if the connection drops, this
            # will raise WebSocketDisconnect.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected ({len(connected_clients)} total)")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health():
    """Simple health check endpoint for monitoring tools."""
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "device": DEVICE,
        "capturing": is_capturing,
    }


# ============================================================================
# STATIC FILE SERVING
# ============================================================================
# Explicit routes for / and /admin so they serve the HTML files directly.
# Everything in the static/ directory is also served at /static/ (CSS, JS,
# images, favicon, manifest, etc.)

static_dir = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    """Serve the client caption display page."""
    return FileResponse(static_dir / "index.html")


@app.get("/admin")
async def serve_admin():
    """Serve the admin control page."""
    return FileResponse(static_dir / "admin.html")


app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # The admin page polls /api/status every 500ms for the audio meter.
    # Without this filter, every single poll would show up in the uvicorn
    # access log — that's 7,200 log lines per hour of just status checks.
    # This filter silently drops those so the log stays useful.
    class QuietAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "/api/status" in msg or "/api/server-info" in msg:
                return False
            return True

    logging.getLogger("uvicorn.access").addFilter(QuietAccessFilter())

    logger.info(f"Server running at http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
