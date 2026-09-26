# OpenEar — Real-time AI captioning and translation for churches
# Copyright (c) 2026 TheRevDrJ
# Licensed under AGPL-3.0 — see LICENSE file for details
"""
OpenEar — live captioning and translation server. THE whole application.

Version lives in VERSION below and nowhere else.

⚠ THIS HEADER WAS FALSE UNTIL 2026-08-23 and had been for months: it said
"v0.5", described faster-whisper running on the GPU, and referenced a
CHUNK_DURATION constant that does not exist. None of that had been true since
the Parakeet switch. Rewritten rather than annotated (index.md's standing rule:
a stacked correction leaves the false line standing, and the next reader takes
the top sentence).

Architecture:
  1. A sounddevice InputStream captures audio from a chosen input device.
  2. Audio arrives in ~100ms blocks via a callback into a thread-safe buffer.
  3. transcription_loop() cuts on VOICE ACTIVITY, not a fixed interval: it
     accumulates at least MIN_CHUNK_DURATION (5s), then cuts at the next trailing
     silence, hard-capping at MAX_CHUNK_DURATION (10s). That window is the delay
     between speech and caption, and it is deliberate — the recogniser needs a
     complete phrase to be accurate and to punctuate. Measured in a real
     sanctuary: median chunk 7.7s, and 39% run to the 10s cap because preaching
     rarely pauses for half a second. The first word of a chunk waits the whole
     chunk; the last waits almost none.
  4. **NVIDIA Parakeet** (onnx_asr) transcribes, on the CPU. Not Whisper. It is
     chosen for punctuation consistency, which the translator depends on.
  5. broadcast() sends each transcribed chunk to English clients the moment it
     exists. Separately, it buffers the text until a sentence ends and — only
     for clients who chose another language — translates that whole sentence
     with **NLLB-200 3.3B** (CTranslate2, int8, CUDA). Both sides of the text
     log are stamped with a shared segment id so quality scoring pairs them.
  6. Clients are display-only. All capture happens server-side.

TWO MODES, chosen per machine by setup.bat and read from mode.json at startup
(openear_config.py holds the rules and the reasons):
  captions      Parakeet only. The translation model is never imported or
                loaded, so the process takes no video memory at all.
  translation   Parakeet plus NLLB-200 on the GPU (about 4.6 GB).
A launch flag, --captions-only or --translation, overrides the file for one run.

  admin.html  — capture start/stop, device selection, languages, mode (REST).
  index.html  — the congregant view; WebSocket in, text out.
"""

VERSION = "0.12.0"

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
# CTranslate2 can load them for translation. Only the translation model uses
# them; captions run on the CPU.
# If the nvidia packages aren't installed, these blocks silently do nothing.

try:
    import nvidia.cublas
    cublas_path = os.path.join(os.path.dirname(nvidia.cublas.__path__[0]), "cublas", "bin")
    if os.path.isdir(cublas_path):
        os.add_dll_directory(cublas_path)
        os.environ["PATH"] = cublas_path + os.pathsep + os.environ.get("PATH", "")
except (ImportError, Exception):
    pass

try:
    import nvidia.cudnn
    cudnn_path = os.path.join(os.path.dirname(nvidia.cudnn.__path__[0]), "cudnn", "bin")
    if os.path.isdir(cudnn_path):
        os.add_dll_directory(cudnn_path)
        os.environ["PATH"] = cudnn_path + os.pathsep + os.environ.get("PATH", "")
except (ImportError, Exception):
    pass

import numpy as np
import sounddevice as sd
import onnx_asr
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import openear_config

# ctranslate2 and sentencepiece (the translation stack) are imported ONLY inside
# load_translation_model(), and only in translation mode. A captions-only server
# never imports them, so it needs no CUDA libraries to start.

# ============================================================================
# CONFIGURATION
# ============================================================================

# Text logging — enabled with --log-text flag. Writes transcription and
# translation output to separate files for quality evaluation.
LOG_TEXT = "--log-text" in sys.argv
TEXT_LOG_DIR = Path(__file__).parent / "text-logs"
if LOG_TEXT:
    TEXT_LOG_DIR.mkdir(exist_ok=True)

ASR_MODEL = openear_config.PARAKEET_MODEL  # NVIDIA Parakeet — natively punctuated output
PORT = 80                   # Default HTTP port — no :port needed in URLs
SAMPLE_RATE = 16000         # 16kHz — what Parakeet expects. Audio is resampled to this.
MIN_CHUNK_DURATION = 5      # Don't cut before 5s — too little context for Parakeet.
MAX_CHUNK_DURATION = 10     # Hard cap — always cut here even mid-speech.
                            # WER tested at fixed intervals: 3s=14.4%, 5s=4.1%, 10s=3.0%.
SILENCE_THRESHOLD  = 0.005  # RMS below this = silence. Raise if cutting mid-speech;
                            # lower if not cutting at natural pauses. Range: 0.003–0.05.
                            # 0.005 optimal for close-mic'd SM58/lapel/headset.
                            # If room ambient mic is used, try 0.015–0.02.
SILENCE_WINDOW     = 0.5    # Seconds of trailing audio to check for silence.

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
# MODE — captions only, or captions + translation
# ============================================================================
# Resolved once, here, before any model loads, because it decides which models
# load. The rules (flag beats file, absent file means captions) live in
# openear_config.py so the downloader and the launcher apply the same ones.

try:
    MODE, MODE_SOURCE = openear_config.resolve_mode(sys.argv[1:])
except openear_config.ModeError as e:
    logger.error(f"Cannot start: {e}")
    sys.exit(2)

TRANSLATION_MODE = MODE == openear_config.TRANSLATION

# When this process started. The admin page watches it to notice a restart even
# when the mode did not change - a restart can still change WHY translation is off.
SERVER_STARTED = int(time.time())
logger.info(f"OpenEar v{VERSION} starting - mode: {openear_config.describe(MODE)} (from {MODE_SOURCE})")

# ============================================================================
# FASTAPI APP & SPEECH MODEL
# ============================================================================

app = FastAPI(title="OpenEar")

# Load the Parakeet ASR model at startup, ON THE CPU — pinned, not left to chance.
# It is fast enough there to need no graphics card (measured on an i7-12700K: 8s
# of audio in 0.37s, about 22x real time), and a captions-only machine takes no
# video memory.
#
# The pin is load-bearing. onnxruntime-gpu advertises TensorRT and CUDA ahead of
# the CPU, and onnx_asr tries them in that order. Parakeet only ever landed on
# the CPU because both GPU providers FAILED to load (a missing nvinfer / cufft
# DLL) — printing a wall of errors at every start. Any machine that happened to
# have those DLLs would have put speech on the GPU without a word, and "captions
# only uses no video memory" would have quietly stopped being true.
# On first run, it downloads ~2.5GB of model weights from HuggingFace.
logger.info("Loading Parakeet ASR model on the CPU (first run downloads ~2.5GB)...")
t0 = time.time()
asr_model = onnx_asr.load_model(ASR_MODEL, providers=["CPUExecutionProvider"])
logger.info(f"ASR model loaded in {time.time() - t0:.1f}s")

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

NLLB_MODEL_DIR = str(openear_config.NLLB_MODEL_DIR)

nllb_translator = None
nllb_sp = None

# Why translation is not running, in a sentence an operator can act on — or None
# when it is running. The admin page shows this verbatim, so it must never claim
# translation works when it does not: that is the difference between a machine
# that lost translation and says so, and one that serves English to a Spanish
# reader while the language list still offers Spanish.
translation_unavailable_reason: str | None = None


def load_translation_model():
    """Load NLLB-200 onto the GPU. Called once at startup, in translation mode only.

    Never raises. On any failure translation stays off and
    translation_unavailable_reason says why; the server still starts and
    captions still work.
    """
    global nllb_translator, nllb_sp, translation_unavailable_reason

    logger.info("Loading NLLB-200 translation model...")
    t0 = time.time()
    try:
        import ctranslate2
        import sentencepiece as spm

        # Check the FILES, not just the folder: an interrupted download leaves the
        # folder in place with model.bin missing, and a folder check trusts it.
        #
        # ⛔ AND NEVER DOWNLOAD HERE. This runs before the server opens port 80, so
        # fetching the 13 GB model at startup keeps CAPTIONS down for the whole
        # download - openear.bat gives up after three minutes, and starting again
        # kills the process mid-download. Captions must always come up fast; a
        # missing model is reported at once, and setup.bat (which resumes an
        # interrupted download) is where it gets fetched.
        missing = [f for f in NLLB_REQUIRED_FILES if not (Path(NLLB_MODEL_DIR) / f).is_file()]
        if missing:
            raise FileNotFoundError(f"translation model files missing: {', '.join(missing)}")
        # Present is not complete: a copy cut short, or a disk that filled, leaves a
        # short model.bin that CTranslate2 rejects with an error naming neither
        # cause. Its size is pinned with the revision, so it can be checked here.
        size = (Path(NLLB_MODEL_DIR) / "model.bin").stat().st_size
        if size != openear_config.NLLB_MODEL_BIN_BYTES:
            raise FileNotFoundError(
                f"translation model file incomplete: model.bin is {size:,} bytes, "
                f"expected {openear_config.NLLB_MODEL_BIN_BYTES:,}")

        nllb_translator = ctranslate2.Translator(
            NLLB_MODEL_DIR,
            device="cuda",
            compute_type="int8",
        )
        nllb_sp = spm.SentencePieceProcessor(os.path.join(NLLB_MODEL_DIR, "sentencepiece.bpe.model"))
        logger.info(f"NLLB translation model loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        nllb_translator = None
        nllb_sp = None
        translation_unavailable_reason = f"Captions are running in English. {translation_failure_hint(e)} The error was: {e}"
        logger.error(f"NLLB translation model unavailable: {e}")
        logger.error("Translation is OFF for this session. Captions continue in English only.")


# The files a usable NLLB-200 CTranslate2 folder must hold. model.bin is 13 GB and
# the one an interrupted download most often lacks.
NLLB_REQUIRED_FILES = ("model.bin", "config.json", "shared_vocabulary.json", "sentencepiece.bpe.model")


def translation_failure_hint(e: Exception) -> str:
    """One sentence on what to do about a translation model that would not load.

    Each branch is a failure seen in practice. The fallback does not guess: blaming
    the driver for a missing file sends an operator to nvidia.com for nothing.
    """
    text = str(e).lower()
    if "out of memory" in text:
        return ("The graphics card is out of video memory — something else on this PC is "
                "using it. Translation needs about 4.6 GB free; it works best on a PC of its own.")
    if any(w in text for w in ("cuda", "cublas", "cudnn", "no cuda-capable", "driver")):
        return ("Translation needs an NVIDIA graphics card and the full driver from "
                "nvidia.com — Windows Update's basic driver is not enough.")
    if isinstance(e, (FileNotFoundError, OSError)) or "no such file" in text or "not found" in text:
        return ("The translation model's files are missing or incomplete. Run setup.bat "
                "again with an internet connection to finish downloading them, then "
                "restart OpenEar.")
    return "Translation could not start."


if TRANSLATION_MODE:
    load_translation_model()
elif MODE_SOURCE.endswith("flag"):
    # A one-run test override. This PC's own setting may well be translation, so
    # telling the admin "translation isn't installed" would be false.
    translation_unavailable_reason = (
        "Captions only for this run - OpenEar was started with --captions-only. "
        "Restart it without that flag to use this PC's own setting."
    )
    logger.info("Captions only (launch flag) - the translation model is not loaded and no video memory is used.")
elif MODE_SOURCE == "mode.json":
    # Captions only because someone CHOSE it in setup.
    translation_unavailable_reason = (
        "Translation isn't installed on this PC. To add it, run setup.bat again, "
        "choose translation, then restart OpenEar. It needs an NVIDIA graphics card "
        "with 6 GB or more."
    )
    logger.info("Captions only - the translation model is not loaded and no video memory is used.")
else:
    # Captions only because NOTHING valid was chosen: no mode.json (an install from
    # before modes existed), or one that could not be read. Say that, rather than
    # dress a missing choice up as a deliberate one — a PC that used to translate
    # must be able to tell what happened.
    translation_unavailable_reason = (
        f"No mode has been chosen on this PC ({MODE_SOURCE.removesuffix(openear_config.DEFAULT_SUFFIX)}), so it runs "
        "captions only. To add translation, run setup.bat, choose translation, then "
        "restart OpenEar."
    )
    logger.warning(f"Captions only because no valid mode is recorded ({MODE_SOURCE}). Run setup.bat to choose.")


def translation_available() -> bool:
    """True only when the translation model is actually loaded and usable."""
    return nllb_translator is not None and nllb_sp is not None

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
is_monitoring: bool = False                  # Monitor mode: level meter only, no transcription
monitor_stream: sd.InputStream | None = None # The monitor-only audio stream

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


def effective_languages() -> set[str]:
    """The languages clients can actually receive right now.

    The admin's list when translation is running; English alone when it is not —
    whether because this machine is captions-only or because the model failed to
    load. languages.json is never rewritten to match, so a machine switched back
    to translation gets the admin's list back exactly as it was.
    """
    return set(enabled_languages) if translation_available() else {"en"}


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


def monitor_callback(indata: np.ndarray, frames: int, time_info, status):
    """Audio callback for monitor mode — updates level meter only, no transcription.

    Same math as audio_callback but skips the buffer append, so the audio
    is measured and discarded. This lets the admin verify the right device
    is live before starting a real capture session.
    """
    global current_audio_level, audio_clipping
    if status:
        logger.warning(f"Monitor status: {status}")
    rms = float(np.sqrt(np.mean(indata ** 2)))
    current_audio_level = min(rms * 6.0, 1.0)
    audio_clipping = float(np.max(np.abs(indata))) > 0.70


# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe_audio_chunk(audio_data: np.ndarray) -> str:
    """Convert a chunk of raw audio into text using Parakeet-TDT.

    Parakeet accepts numpy arrays directly — no temp files needed.
    It produces natively punctuated, capitalized text, which is
    critical for downstream translation quality.
    """
    try:
        result = asr_model.recognize(audio_data)
        return result.text.strip() if hasattr(result, 'text') else str(result).strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


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


# Translation sentence buffer — accumulates transcribed chunks until a sentence
# boundary is found, so translation (and the text log) work on whole sentences.
# It feeds translation clients only; English clients are sent each chunk as it
# arrives and never wait on it. See broadcast().
import re
_sentence_end_re = re.compile(r'[.!?][\s]*$')
_translation_buffer: str = ""

# Monotonic ID stamped onto each completed sentence and every translation of it,
# so the text logs pair exactly instead of being re-guessed by a scoring tool.
# Resets per server run; the logs are append-only per session.
_segment_counter: int = 0

# For a phone that has left English mid-sentence: which sentence it left in, and how
# many characters of it it had already read. Lets a return to English send only the
# part it has not seen. Written and read by the websocket handler.
_english_left_at: dict = {}


async def broadcast(message: dict):
    """Send a JSON message to every connected WebSocket client.

    A transcript message has two audiences, served in this order:

    1. ENGLISH clients get each chunk the moment Parakeet produces it. The chunk
       IS the deliberate delay — 5 to 10 seconds of speech, because Parakeet
       needs a whole phrase to transcribe and punctuate accurately (3 s chunks
       measured 14.4% word error against 4.1% at 5 s). Nothing is added on top:
       Parakeet's output is already capitalized and punctuated, and holding it
       longer would not improve it.

    2. TRANSLATION clients get whole sentences. The text is buffered until it
       ends in . ! or ?, then translated once per language and sent. NLLB needs
       the complete sentence to get word order right — Korean and Japanese put
       the verb last — so this second wait is worth paying, but only by them.

    English used to wait on the sentence buffer as well: the whole transcript
    path sat inside the sentence-complete branch, so an English reader saw
    nothing until a LATER chunk supplied a period, and a thought cut off before a
    hymn did not arrive at all. This docstring said "real-time, no delay" the
    entire time.

    English for a chunk is sent BEFORE any translation of it starts. Translation
    still runs inside this call, and transcription_loop() awaits the call, so the
    loop is not watching for the next cut while it runs. Audio keeps buffering, so
    nothing is lost. The cost is timing, and it adds up across languages: about
    0.3-0.4 s per sentence for each language some phone is reading, on a desktop
    GPU, more on a small one. Enabled languages nobody has chosen cost nothing. When the loop resumes it checks only the last half-second for silence, so
    a pause the speaker took WHILE translation ran is missed, and the cut waits for
    the next pause or the 10 s cap - it can come seconds late, not just by the time
    translation took. Captions-only machines never pay this.

    Non-transcript messages (status updates) go to everyone identically.
    """
    global _translation_buffer, _segment_counter
    disconnected = set()

    if message.get("type") == "transcript":
        english_text = message["text"]

        # 1. English readers: this chunk, now.
        #
        # Who counts as an English reader is fixed HERE, before the first await,
        # and the chunk joins the sentence buffer before any send. Both matter for
        # a phone that changes language while these sends are in flight: one that
        # switches TO English is not in this list, and the catch-up in the
        # websocket handler serves it from the buffer, which already holds this
        # chunk - so it gets the chunk exactly once, rather than twice or never.
        english_now = [c for c in connected_clients if client_languages.get(c, "en") == "en"]
        _translation_buffer += (" " if _translation_buffer else "") + english_text
        for client in english_now:
            try:
                await client.send_json({"type": "transcript", "text": english_text, "lang": "en"})
            except Exception:
                disconnected.add(client)

        # 2. Translation readers and the text log: whole sentences only.

        if _sentence_end_re.search(_translation_buffer):
            sentence_raw = _translation_buffer        # unstripped: read-offsets index into this
            complete_text = _translation_buffer.strip()
            _translation_buffer = ""

            # ── Segment ID: the unit of comparison for quality scoring ────────
            #
            # WHY THE SOURCE IS LOGGED *HERE* AND NOT WHERE THE FRAGMENT ARRIVED.
            #
            # This used to log every incoming ASR fragment to source-en.txt, while
            # translations were logged once per COMPLETED SENTENCE below. Those are
            # two different units: the buffer merges several fragments into one
            # sentence before translating. So the two log files were never 1:1 --
            # a 10-fragment stretch would produce 8 translated sentences, and
            # nothing said so.
            #
            # That silently destroyed the pairing at write time, and no downstream
            # tool could recover it: score_translation.py had to guess which source
            # line produced which translation using character length, which cannot
            # see meaning and got it wrong on real data.
            #
            # Now both sides log the SAME unit -- the completed sentence -- stamped
            # with a shared, monotonic segment ID. Pairing is exact by construction
            # rather than reconstructed by heuristic afterwards. Data you have to
            # consult is a rule; data baked into the artifact is a fact.
            _segment_counter += 1
            segment_id = _segment_counter

            if LOG_TEXT:
                with open(TEXT_LOG_DIR / "source-en.txt", "a", encoding="utf-8") as f:
                    f.write(f"{segment_id:05d}\t{complete_text}\n")

            # Translate once per language, send to each phone that was waiting for
            # this sentence. English phones were served above, chunk by chunk.
            #
            # The waiting list is fixed before the first await, but each phone gets
            # the sentence in the language it holds AT THE MOMENT OF SENDING. A phone
            # that switched to English while this sentence was being translated
            # would otherwise fall between the two paths - skipped here as English,
            # and too late for the catch-up, because the buffer is already empty -
            # and never see this sentence in either language.
            waiting = [c for c in connected_clients
                       if client_languages.get(c, "en") != "en" and c not in disconnected]
            # How much of this sentence each waiting phone had already read in
            # English before it left - snapshotted WITH the list, because the
            # handler drops that record the moment the phone returns to English.
            read_before = {c: _english_left_at.get(c) for c in waiting}
            translation_cache: dict[str, str] = {}
            for client in waiting:
                lang = client_languages.get(client, "en")
                try:
                    if lang == "en":
                        # Back in English while this sentence was being translated:
                        # send only what it has not already read in English, or a
                        # reader who briefly picked another language sees the start
                        # of the sentence twice.
                        rec = read_before.get(client)
                        seen = rec[1] if rec and rec[0] == segment_id - 1 else 0
                        unread = sentence_raw[seen:].strip()
                        if unread:
                            await client.send_json({"type": "transcript", "text": unread, "lang": "en"})
                        continue
                    if lang not in translation_cache:
                        loop = asyncio.get_event_loop()
                        translated = await loop.run_in_executor(
                            None, translate_text, complete_text, lang
                        )
                        translation_cache[lang] = translated

                        # Log translated text, stamped with the SAME segment ID as
                        # its source above, so scoring pairs them exactly.
                        if LOG_TEXT:
                            with open(TEXT_LOG_DIR / f"translated-{lang}.txt", "a", encoding="utf-8") as f:
                                f.write(f"{segment_id:05d}\t{translated}\n")

                    await client.send_json({
                        "type": "transcript",
                        "text": translation_cache[lang],
                        "lang": lang,
                    })
                except Exception:
                    disconnected.add(client)
    else:
        # Non-transcript messages go to everyone identically. Iterate a COPY: each
        # send awaits, and a phone connecting during that await would otherwise
        # change the set mid-loop and raise.
        for client in list(connected_clients):
            try:
                await client.send_json(message)
            except Exception:
                disconnected.add(client)

    connected_clients.difference_update(disconnected)


async def transcription_loop():
    """Main transcription loop — runs as an async task while capturing is active.

    Uses Voice Activity Detection (VAD) to cut at natural speech boundaries
    rather than fixed time intervals. Every 0.25s it checks whether to cut:

    1. Below MIN_CHUNK_DURATION: keep accumulating, not enough context yet
    2. Between MIN and MAX: cut only if trailing audio is below SILENCE_THRESHOLD
       (speaker paused) — this gives Parakeet complete sentences
    3. At MAX_CHUNK_DURATION: hard cut regardless, prevents runaway buffering

    Cutting at silence vs. mid-word significantly reduces WER because Parakeet
    sees complete phrases instead of arbitrary slices.

    Note on privacy: we log THAT a transcription happened, not WHAT was said.
    Sermon content stays ephemeral — it never hits disk.
    """
    loop = asyncio.get_event_loop()
    silence_samples = int(SAMPLE_RATE * SILENCE_WINDOW)

    while is_capturing:
        try:
            # Check 4x/second — fast enough to catch natural pauses
            await asyncio.sleep(0.25)

            chunk = None
            cut_reason = ""

            with buffer_lock:
                if not audio_buffer:
                    continue

                # How much audio have we accumulated?
                total_samples = sum(len(b) for b in audio_buffer)
                accumulated = total_samples / SAMPLE_RATE

                if accumulated < MIN_CHUNK_DURATION:
                    continue  # Not enough context yet — keep accumulating

                # Check if the trailing SILENCE_WINDOW seconds is silent
                # Grab tail blocks without draining the buffer
                tail_chunks = []
                tail_count = 0
                for block in reversed(audio_buffer):
                    tail_chunks.append(block)
                    tail_count += len(block)
                    if tail_count >= silence_samples:
                        break
                tail = np.concatenate(tail_chunks)
                rms = float(np.sqrt(np.mean(tail ** 2)))
                at_silence = rms < SILENCE_THRESHOLD

                if accumulated >= MAX_CHUNK_DURATION:
                    cut_reason = f"max cap ({accumulated:.1f}s)"
                elif at_silence:
                    cut_reason = f"silence (rms={rms:.4f}, {accumulated:.1f}s)"
                else:
                    continue  # Speech still in progress — wait for a pause

                # Cut here — drain the buffer
                chunk = np.concatenate(audio_buffer)
                audio_buffer.clear()

            # Mono conversion (stereo input devices)
            if chunk.ndim > 1:
                chunk = chunk.mean(axis=1)

            # Run transcription on a thread pool so inference doesn't
            # block the async event loop (WebSocket connections would stall)
            text = await loop.run_in_executor(None, transcribe_audio_chunk, chunk)

            if text:
                logger.info(f"Transcription sent to clients [{cut_reason}]")
                await broadcast({"type": "transcript", "text": text})

        except asyncio.CancelledError:
            raise  # Let stop_capture() cancellation propagate normally
        except Exception as e:
            logger.error(f"Transcription loop error (continuing): {e}", exc_info=True)


# ============================================================================
# MONITOR CONTROL
# ============================================================================

def start_monitor(device_id: int):
    """Open an audio stream for level monitoring only — no buffering, no transcription.

    The flag is set only AFTER the stream is running. It used to be set first,
    so a device that refused to open left /api/status reporting monitoring:true
    while nothing was listening. Raises if the device cannot be opened.
    """
    global is_monitoring, monitor_stream
    stream = sd.InputStream(
        device=device_id,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=monitor_callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    )
    try:
        stream.start()
    except Exception:
        stream.close()
        raise
    monitor_stream = stream
    is_monitoring = True
    logger.info(f"Audio monitor started on device {device_id}")


def stop_monitor():
    """Stop the monitor stream and clean up."""
    global is_monitoring, monitor_stream
    is_monitoring = False
    current_audio_level_reset()
    if monitor_stream:
        monitor_stream.stop()
        monitor_stream.close()
        monitor_stream = None
    logger.info("Audio monitor stopped")


def current_audio_level_reset():
    global current_audio_level, audio_clipping
    current_audio_level = 0.0
    audio_clipping = False


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

    # Clear any stale audio from a previous capture session
    with buffer_lock:
        audio_buffer.clear()

    stream = sd.InputStream(
        device=device_id,
        samplerate=SAMPLE_RATE,   # Resample to 16kHz (what Parakeet expects)
        channels=1,               # Mono capture
        dtype="float32",          # Samples as floats in [-1.0, 1.0]
        callback=audio_callback,  # Called on the audio thread for each block
        blocksize=int(SAMPLE_RATE * 0.1),  # 1600 samples = 100ms blocks
    )
    try:
        stream.start()
    except Exception:
        stream.close()
        raise

    # The state is claimed only once the stream is really running. It used to be
    # set before InputStream() — so a device that refused 16 kHz (most 48 kHz-only
    # USB and Bluetooth mics do) raised, and left /api/status reporting
    # capturing:true while nothing was being captured at all.
    capture_stream = stream
    selected_device_id = device_id
    is_capturing = True
    logger.info(f"Audio capture started on device {device_id}")


def stop_capture():
    """Stop the audio stream and clean up."""
    global is_capturing, capture_stream, _translation_buffer

    is_capturing = False
    if capture_stream:
        capture_stream.stop()
        capture_stream.close()
        capture_stream = None

    # Discard any unprocessed audio and translation buffer - and the English
    # read-offsets into that buffer, which would otherwise index into whatever
    # sentence starts after the restart and hand a phone a garbled tail of it.
    with buffer_lock:
        audio_buffer.clear()
    _translation_buffer = ""
    _english_left_at.clear()

    logger.info("Audio capture stopped")


# ============================================================================
# REST API ENDPOINTS
# ============================================================================
# These are called by the admin page (admin.html) to control the server.
# The client page (index.html) doesn't use REST — it only uses WebSocket.

def api_error(message: str, status: int = 400) -> JSONResponse:
    """An error response the admin page can actually see.

    These endpoints used to `return {"error": ...}, 400`. FastAPI does not read
    that tuple as a status code: it serialises it as a JSON ARRAY with HTTP 200,
    so admin.html's `if (data.error)` checks never fired and every failure was
    silent. Always return errors through here.
    """
    return JSONResponse({"error": message}, status_code=status)


def device_error(device_id, e: Exception) -> JSONResponse:
    """Explain an audio device that would not open, in words an operator can act on."""
    message = f"Could not open audio device {device_id}: {e}"
    # The one failure that has cost real time: a 48 kHz-only USB or Bluetooth mic
    # cannot be opened at 16 kHz. The Windows Sound Mapper resamples for it.
    if "sample rate" in str(e).lower():
        message += " — this device will not run at 16 kHz. Choose 'Microsoft Sound Mapper' instead, which converts for it."
    logger.error(message)
    return api_error(message)


# ONE capture-control request at a time. api_start awaits the old transcription
# loop's cancellation, and without this a Stop - or a second Start - from another
# admin page could run in that gap: orphaning a transcription loop, or overwriting
# an audio stream that is never closed. Two admin pages open at once is ordinary.
_capture_lock = asyncio.Lock()


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
    async with _capture_lock:
        device_id = body.get("device_id")
        if device_id is None:
            return api_error("device_id required")

        # Stop monitor mode if active — capture and monitor can't share the same device
        if is_monitoring:
            stop_monitor()

        # Stop any existing capture before starting a new one - and its transcription
        # loop. start_capture() sets is_capturing back to True at once, so a loop left
        # running here would pass its own "while is_capturing" check and keep going
        # beside the new one: two loops cutting the same audio, their broadcasts
        # overlapping, chunks able to reach phones out of order.
        was_capturing = is_capturing
        if is_capturing:
            stop_capture()
        if transcription_task:
            transcription_task.cancel()
            try:
                await transcription_task
            except asyncio.CancelledError:
                pass
            transcription_task = None

        try:
            start_capture(device_id)
        except Exception as e:
            # If this was a device switch, the old capture is already gone — tell the
            # clients, or they keep showing "live" over a silent server.
            if was_capturing:
                await broadcast({"type": "status", "capturing": False})
            return device_error(device_id, e)

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
    async with _capture_lock:
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


@app.post("/api/monitor/start")
async def api_monitor_start(body: dict):
    """Start monitor mode: open audio stream for level metering only, no transcription.

    Lets the admin verify the correct device is live and at a good level
    before committing to a full capture session.
    """
    async with _capture_lock:
        device_id = body.get("device_id")
        if device_id is None:
            return api_error("device_id required")
        if is_capturing:
            return api_error("Cannot monitor while capturing")
        if is_monitoring:
            stop_monitor()
        try:
            start_monitor(device_id)
        except Exception as e:
            return device_error(device_id, e)
        return {"status": "monitoring", "device_id": device_id}


@app.post("/api/monitor/stop")
async def api_monitor_stop():
    """Stop monitor mode."""
    async with _capture_lock:
        stop_monitor()
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
        "monitoring": is_monitoring,
        "device_id": selected_device_id,
        "clients": len(connected_clients),
        "audio_level": round(current_audio_level, 3),
        "clipping": audio_clipping,
        "languages": lang_counts,
        "mode": MODE,
        "translation_available": translation_available(),
        "started": SERVER_STARTED,
    }


# ============================================================================
# LANGUAGE / TRANSLATION API ENDPOINTS
# ============================================================================
# NLLB handles all 200 languages with a single model — no per-language packs
# to install or remove. The API just returns the available language list.

@app.get("/api/languages")
async def list_languages():
    """Return all NLLB languages, which ones clients can receive, and why.

    'installed'   = every language the model knows (the admin's toggle list)
    'enabled'     = the languages clients can actually choose right now —
                    English alone whenever translation is not running
    'translation' = the mode, whether translation is running, and if not, the
                    reason, in words the admin page shows as-is
    """
    available = get_available_languages()
    return {
        "installed": available,
        "enabled": sorted(effective_languages()),
        "translation": {
            "mode": MODE,
            "available": translation_available(),
            "reason": translation_unavailable_reason,
        },
    }

@app.post("/api/languages/enable")
async def enable_language(body: dict):
    """Enable a language so it appears on client devices."""
    code = body.get("code", "")
    if code not in NLLB_LANG_MAP and code != "en":
        return api_error(f"Unknown language code: {code}")
    if code != "en" and not translation_available():
        # 409: the request is fine, this machine's state is what refuses it.
        return api_error(translation_unavailable_reason, status=409)
    enabled_languages.add(code)
    save_enabled_languages(enabled_languages)
    return {"enabled": sorted(effective_languages())}

@app.post("/api/languages/disable")
async def disable_language(body: dict):
    """Disable a language so it no longer appears on client devices."""
    code = body.get("code", "")
    if code == "en":
        return api_error("English cannot be disabled")
    enabled_languages.discard(code)
    save_enabled_languages(enabled_languages)
    return {"enabled": sorted(effective_languages())}


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

    # Tell the new client whether captioning is currently active, and send only
    # the languages it can actually receive — English alone when translation is
    # not running, so a phone is never offered a language it would not get.
    all_langs = get_available_languages()
    offered = effective_languages()
    visible = [l for l in all_langs if l["code"] in offered]
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
                    requested = msg["lang"]
                    # A phone can ask for a language it was never offered — most
                    # often one saved in its browser from an earlier visit. Serve
                    # English rather than translate into something the admin has
                    # not enabled, or that this machine cannot produce at all.
                    new_lang = requested if requested in effective_languages() else "en"
                    client_languages[websocket] = new_lang
                    if new_lang != requested:
                        logger.info(f"Client asked for {requested}, which is not offered - serving English")
                    else:
                        logger.info(f"Client switched language: {old_lang} -> {new_lang}")
                    # A phone moving between English and a translation mid-sentence.
                    # Leaving English: remember how much of this sentence it has
                    # already read. Returning to English: send the part of the
                    # sentence so far that it has NOT read - all of it, if it left
                    # during an earlier sentence or joined in another language.
                    # English chunks go only to English phones, so without this it
                    # would never see those words in any language.
                    if old_lang == "en" and new_lang != "en":
                        _english_left_at[websocket] = (_segment_counter, len(_translation_buffer))
                    elif new_lang == "en" and old_lang != "en":
                        seg, seen = _english_left_at.pop(websocket, (None, 0))
                        unread = _translation_buffer[seen if seg == _segment_counter else 0:].strip()
                        if unread:
                            await websocket.send_json({"type": "transcript", "text": unread, "lang": "en"})
            except (json.JSONDecodeError, Exception):
                pass  # Ignore malformed messages
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(websocket)
        client_languages.pop(websocket, None)
        _english_left_at.pop(websocket, None)
        logger.info(f"Client disconnected ({len(connected_clients)} total)")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health():
    """Simple health check endpoint for monitoring tools."""
    return {
        "status": "ok",
        "version": VERSION,
        "mode": MODE,
        "model": ASR_MODEL,
        "device": "cpu (ASR) / cuda (translation)" if translation_available() else "cpu (ASR)",
        "translation_available": translation_available(),
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
