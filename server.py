# OpenEar — Real-time AI captioning and translation for churches
# Copyright (c) 2026 TheRevDrJ
# Licensed under AGPL-3.0 — see LICENSE file for details
"""
OpenEar v0.5 - Live Speech-to-Text Captioning Server with NLLB Translation

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

VERSION = "0.5.0"

import os
import sys
import io
import json
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
# TRANSLATION (NLLB-200 via CTranslate2)
# ============================================================================
# Meta's NLLB-200 (No Language Left Behind) provides high-quality offline
# translation across 200 languages. We use the 3.3B parameter model quantized
# to INT8 via CTranslate2, which uses ~3GB VRAM on GPU or runs on CPU.
# This is the same CTranslate2 engine that powers faster-whisper, so no new
# runtime dependencies are needed.

import ctranslate2
import sentencepiece as spm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Text logging — enabled with --log-text flag. Writes transcription and
# translation output to separate files for quality evaluation.
LOG_TEXT = "--log-text" in sys.argv
TEXT_LOG_DIR = Path(__file__).parent / "text-logs"
if LOG_TEXT:
    TEXT_LOG_DIR.mkdir(exist_ok=True)

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
# NLLB TRANSLATION MODEL
# ============================================================================
# NLLB uses BCP-47-style codes with script suffixes (e.g., kor_Hang for Korean).
# We map simple ISO 639-1 codes (what clients send) to NLLB's format.

NLLB_LANG_MAP = {
    "af": ("afr_Latn", "Afrikaans"), "am": ("amh_Ethi", "Amharic"),
    "ar": ("arb_Arab", "Arabic"), "az": ("azj_Latn", "Azerbaijani"),
    "be": ("bel_Cyrl", "Belarusian"), "bg": ("bul_Cyrl", "Bulgarian"),
    "bn": ("ben_Beng", "Bengali"), "bs": ("bos_Latn", "Bosnian"),
    "ca": ("cat_Latn", "Catalan"), "cs": ("ces_Latn", "Czech"),
    "cy": ("cym_Latn", "Welsh"), "da": ("dan_Latn", "Danish"),
    "de": ("deu_Latn", "German"), "el": ("ell_Grek", "Greek"),
    "es": ("spa_Latn", "Spanish"), "et": ("est_Latn", "Estonian"),
    "fa": ("pes_Arab", "Persian"), "fi": ("fin_Latn", "Finnish"),
    "fr": ("fra_Latn", "French"), "ga": ("gle_Latn", "Irish"),
    "gl": ("glg_Latn", "Galician"), "gu": ("guj_Gujr", "Gujarati"),
    "ha": ("hau_Latn", "Hausa"), "he": ("heb_Hebr", "Hebrew"),
    "hi": ("hin_Deva", "Hindi"), "hr": ("hrv_Latn", "Croatian"),
    "hu": ("hun_Latn", "Hungarian"), "hy": ("hye_Armn", "Armenian"),
    "id": ("ind_Latn", "Indonesian"), "ig": ("ibo_Latn", "Igbo"),
    "is": ("isl_Latn", "Icelandic"), "it": ("ita_Latn", "Italian"),
    "ja": ("jpn_Jpan", "Japanese"), "ka": ("kat_Geor", "Georgian"),
    "kk": ("kaz_Cyrl", "Kazakh"), "km": ("khm_Khmr", "Khmer"),
    "kn": ("kan_Knda", "Kannada"), "ko": ("kor_Hang", "Korean"),
    "lo": ("lao_Laoo", "Lao"), "lt": ("lit_Latn", "Lithuanian"),
    "lv": ("lvs_Latn", "Latvian"), "mk": ("mkd_Cyrl", "Macedonian"),
    "ml": ("mal_Mlym", "Malayalam"), "mn": ("khk_Cyrl", "Mongolian"),
    "mr": ("mar_Deva", "Marathi"), "ms": ("zsm_Latn", "Malay"),
    "my": ("mya_Mymr", "Myanmar"), "ne": ("npi_Deva", "Nepali"),
    "nl": ("nld_Latn", "Dutch"), "no": ("nob_Latn", "Norwegian"),
    "pa": ("pan_Guru", "Punjabi"), "pl": ("pol_Latn", "Polish"),
    "pt": ("por_Latn", "Portuguese"), "ro": ("ron_Latn", "Romanian"),
    "ru": ("rus_Cyrl", "Russian"), "si": ("sin_Sinh", "Sinhala"),
    "sk": ("slk_Latn", "Slovak"), "sl": ("slv_Latn", "Slovenian"),
    "so": ("som_Latn", "Somali"), "sq": ("als_Latn", "Albanian"),
    "sr": ("srp_Cyrl", "Serbian"), "sv": ("swe_Latn", "Swedish"),
    "sw": ("swh_Latn", "Swahili"), "ta": ("tam_Taml", "Tamil"),
    "te": ("tel_Telu", "Telugu"), "tg": ("tgk_Cyrl", "Tajik"),
    "th": ("tha_Thai", "Thai"), "tl": ("tgl_Latn", "Filipino"),
    "tr": ("tur_Latn", "Turkish"), "uk": ("ukr_Cyrl", "Ukrainian"),
    "ur": ("urd_Arab", "Urdu"), "uz": ("uzn_Latn", "Uzbek"),
    "vi": ("vie_Latn", "Vietnamese"), "yo": ("yor_Latn", "Yoruba"),
    "zh": ("zho_Hans", "Chinese (Simplified)"),
    "zu": ("zul_Latn", "Zulu"),
}

NLLB_MODEL_DIR = str(Path.home() / ".cache" / "nllb-3.3b-ct2")

# Load NLLB translation model
logger.info("Loading NLLB-200 translation model...")
t0 = time.time()
try:
    nllb_translator = ctranslate2.Translator(
        NLLB_MODEL_DIR,
        device=DEVICE,
        compute_type="int8",
    )
    nllb_sp = spm.SentencePieceProcessor(os.path.join(NLLB_MODEL_DIR, "sentencepiece.bpe.model"))
    logger.info(f"NLLB translation model loaded in {time.time() - t0:.1f}s")
except Exception as e:
    logger.warning(f"NLLB translation model not found: {e}")
    logger.warning("Translation will be unavailable. Run download_models.py to install.")
    nllb_translator = None
    nllb_sp = None

# ============================================================================
# SERVER STATE
# ============================================================================
# These globals track the current state of the server. They're modified by
# the API endpoints and read by the admin page's status polling.

# Client tracking: maps each WebSocket to its preferred language code.
# "en" means no translation needed. Any other code triggers NLLB translation.
client_languages: dict[WebSocket, str] = {}  # {websocket: "en", websocket2: "ko", ...}
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
# ENABLED LANGUAGES (admin-controlled visibility)
# ============================================================================
# Admins toggle which languages appear on client devices. Persisted to disk.
# English is always enabled and cannot be disabled.

LANGUAGES_FILE = Path(__file__).parent / "languages.json"

def load_enabled_languages() -> set[str]:
    """Load enabled language codes from disk, defaulting to English only."""
    try:
        with open(LANGUAGES_FILE) as f:
            data = json.load(f)
            codes = set(data.get("enabled", ["en"]))
            codes.add("en")  # English always enabled
            return codes
    except (FileNotFoundError, json.JSONDecodeError):
        return {"en"}

def save_enabled_languages(codes: set[str]):
    """Persist enabled language codes to disk."""
    codes.add("en")  # English always enabled
    with open(LANGUAGES_FILE, "w") as f:
        json.dump({"enabled": sorted(codes)}, f, indent=2)

enabled_languages: set[str] = load_enabled_languages()


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


def get_available_languages() -> list[dict]:
    """Return all languages NLLB can translate to.

    NLLB supports all 200 languages with a single model — no per-language
    packs to install. We expose a curated subset of the most useful ones
    for church contexts.
    """
    languages = [
        {"code": code, "name": name}
        for code, (nllb_code, name) in sorted(NLLB_LANG_MAP.items(), key=lambda x: x[1][1])
    ]
    return languages


def translate_text(text: str, target_lang: str) -> str:
    """Translate English text to the target language using NLLB-200.

    Returns the original text if translation fails or target is English.
    """
    if not text or target_lang == "en":
        return text
    if not nllb_translator or not nllb_sp:
        return text

    nllb_code = NLLB_LANG_MAP.get(target_lang, (None, None))[0]
    if not nllb_code:
        logger.warning(f"No NLLB mapping for language code: {target_lang}")
        return text

    try:
        tokens = nllb_sp.encode(text, out_type=str)
        tokens = ["eng_Latn"] + tokens + ["</s>"]
        results = nllb_translator.translate_batch(
            [tokens],
            target_prefix=[[nllb_code]],
            max_batch_size=1,
            beam_size=4,
        )
        output_tokens = results[0].hypotheses[0][1:]  # skip language token
        return nllb_sp.decode(output_tokens)
    except Exception as e:
        logger.warning(f"Translation to {target_lang} failed: {e}")
        return text


async def broadcast(message: dict):
    """Send a JSON message to every connected WebSocket client.

    For transcript messages, translates the text to each client's preferred
    language before sending. English clients get the original text with no
    added latency. Translation results are cached per-broadcast so that if
    5 clients all want Spanish, we only translate once.

    For non-transcript messages (status updates), sends identically to all.
    """
    disconnected = set()

    if message.get("type") == "transcript":
        # Cache translations so each language is only translated once per chunk
        translation_cache: dict[str, str] = {"en": message["text"]}
        english_text = message["text"]

        # Log original English text
        if LOG_TEXT:
            with open(TEXT_LOG_DIR / "source-en.txt", "a", encoding="utf-8") as f:
                f.write(english_text + "\n")

        for client in connected_clients:
            lang = client_languages.get(client, "en")
            try:
                if lang not in translation_cache:
                    # Run translation in executor to not block the event loop
                    loop = asyncio.get_event_loop()
                    translated = await loop.run_in_executor(
                        None, translate_text, english_text, lang
                    )
                    translation_cache[lang] = translated

                    # Log translated text
                    if LOG_TEXT:
                        with open(TEXT_LOG_DIR / f"translated-{lang}.txt", "a", encoding="utf-8") as f:
                            f.write(translated + "\n")

                await client.send_json({
                    "type": "transcript",
                    "text": translation_cache[lang],
                    "lang": lang,
                })
            except Exception:
                disconnected.add(client)
    else:
        # Non-transcript messages go to everyone identically
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
    # Count how many clients are using each language
    lang_counts: dict[str, int] = {}
    for lang in client_languages.values():
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    return {
        "capturing": is_capturing,
        "device_id": selected_device_id,
        "clients": len(connected_clients),
        "audio_level": round(current_audio_level, 3),
        "clipping": audio_clipping,
        "languages": lang_counts,
    }


# ============================================================================
# LANGUAGE / TRANSLATION API ENDPOINTS
# ============================================================================
# NLLB handles all 200 languages with a single model — no per-language packs
# to install or remove. The API just returns the available language list.

@app.get("/api/languages")
async def list_languages():
    """Return all NLLB languages and which ones are enabled for clients.

    'installed' = all available languages (for admin toggle list)
    'enabled' = languages visible to clients (admin-controlled)
    """
    available = get_available_languages()
    return {
        "installed": available,
        "enabled": sorted(enabled_languages),
    }

@app.post("/api/languages/enable")
async def enable_language(body: dict):
    """Enable a language so it appears on client devices."""
    code = body.get("code", "")
    if code not in NLLB_LANG_MAP and code != "en":
        return {"error": f"Unknown language code: {code}"}, 400
    enabled_languages.add(code)
    save_enabled_languages(enabled_languages)
    return {"enabled": sorted(enabled_languages)}

@app.post("/api/languages/disable")
async def disable_language(body: dict):
    """Disable a language so it no longer appears on client devices."""
    code = body.get("code", "")
    if code == "en":
        return {"error": "English cannot be disabled"}, 400
    enabled_languages.discard(code)
    save_enabled_languages(enabled_languages)
    return {"enabled": sorted(enabled_languages)}


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
    client_languages[websocket] = "en"  # Default to English
    logger.info(f"Client connected ({len(connected_clients)} total)")

    # Tell the new client whether captioning is currently active,
    # and send only the admin-enabled languages for the dropdown
    all_langs = get_available_languages()
    visible = [l for l in all_langs if l["code"] in enabled_languages]
    await websocket.send_json({"type": "status", "capturing": is_capturing})
    await websocket.send_json({
        "type": "languages",
        "languages": [{"code": "en", "name": "English"}] + visible,
    })

    try:
        while True:
            # Clients can now send messages to set their language preference.
            # Message format: {"type": "set_language", "lang": "fr"}
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if msg.get("type") == "set_language" and msg.get("lang"):
                    old_lang = client_languages.get(websocket, "en")
                    client_languages[websocket] = msg["lang"]
                    logger.info(f"Client switched language: {old_lang} -> {msg['lang']}")
            except (json.JSONDecodeError, Exception):
                pass  # Ignore malformed messages
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        client_languages.pop(websocket, None)
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

    # When running under pythonw (headless/no console), sys.stdout is None.
    # Uvicorn's default log formatter calls sys.stdout.isatty() which crashes.
    # Disable uvicorn's log config in headless mode — our own logging still works.
    headless = sys.stdout is None
    log_config = None if headless else uvicorn.config.LOGGING_CONFIG

    logger.info(f"Server running at http://0.0.0.0:{PORT}")
    if LOG_TEXT:
        logger.info(f"Text logging enabled — output to {TEXT_LOG_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info", log_config=log_config)
