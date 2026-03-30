#!/usr/bin/env python3
"""
test_pipeline.py — OpenEar ASR/Translation Pipeline Test Harness

This tool was developed alongside OpenEar to benchmark transcription accuracy
and translation quality offline — without needing a live microphone or running
server. It uses the same models and pipeline logic as the server (Parakeet ASR,
NLLB-200 translation) so results reflect real-world performance.

If you're evaluating OpenEar for your context, forking it, or experimenting
with alternative models, this is the right starting point. Drop in any audio
file with a known transcript and get a Word Error Rate in under a minute.

── Modes ────────────────────────────────────────────────────────────────────
  transcribe  Audio file → text (measures WER if reference transcript given)
  translate   Text file → translated text
  both        Audio file → translated text
  align       Audio file + full chapter text → trimmed reference text
              (useful when your transcript covers more than the audio does)

── Usage ────────────────────────────────────────────────────────────────────
  python test_pipeline.py transcribe sermon.wav
  python test_pipeline.py transcribe sermon.mp3 --wer reference.txt
  python test_pipeline.py transcribe sermon.mp3 --vad --wer reference.txt
  python test_pipeline.py translate transcript.txt -l ko
  python test_pipeline.py both sermon.wav -l ko -o korean.txt
  python test_pipeline.py align audio.mp3 full_chapter.txt -o trimmed.txt

── VAD chunking ─────────────────────────────────────────────────────────────
  By default, audio is split into fixed 10s chunks. --vad enables silence-
  detection chunking, which cuts at natural speech pauses instead. This
  significantly reduces WER by giving the ASR model complete phrases rather
  than arbitrary slices. Benchmarked on 124 minutes of Chesterton (LibriVox):

    Fixed  3s → 14.4% WER
    Fixed  5s →  4.1% WER
    Fixed 10s →  3.0% WER
    VAD 0.005 →  ~3.3% WER average (best on clean close-mic audio: 1.9%)

  --silence controls the RMS threshold. 0.005 is tuned for close-mic'd
  speech (SM58, lapel, headset). Raise to 0.015–0.02 for ambient room mics.

── Audio formats ─────────────────────────────────────────────────────────────
  WAV, FLAC, OGG natively via soundfile.
  MP3 and other formats require ffmpeg on PATH:
    ffmpeg -i sermon.mp3 sermon.wav
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
NLLB_MODEL_DIR = str(SCRIPT_DIR / "models" / "nllb-3.3b-ct2")
SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 10  # seconds — fixed-interval fallback
DEFAULT_MIN_CHUNK = 5        # VAD: don't cut before this many seconds
DEFAULT_MAX_CHUNK = 10       # VAD: always cut at this many seconds
DEFAULT_SILENCE_THRESHOLD = 0.005  # VAD: RMS below this = silence
DEFAULT_SILENCE_WINDOW = 0.5       # VAD: seconds of trailing audio to check

# Same language map as server.py
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
    "th": ("tha_Thai", "Thai"), "tk": ("tuk_Latn", "Turkmen"),
    "tl": ("tgl_Latn", "Filipino"), "tr": ("tur_Latn", "Turkish"),
    "uk": ("ukr_Cyrl", "Ukrainian"), "ur": ("urd_Arab", "Urdu"),
    "uz": ("uzn_Latn", "Uzbek"), "vi": ("vie_Latn", "Vietnamese"),
    "xh": ("xho_Latn", "Xhosa"), "yo": ("yor_Latn", "Yoruba"),
    "zh": ("zho_Hans", "Chinese (Simplified)"), "zu": ("zul_Latn", "Zulu"),
}


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(audio_path: str) -> np.ndarray:
    """Read audio file, return float32 mono array at SAMPLE_RATE.

    Handles WAV/FLAC/OGG natively via soundfile.
    For MP3 (and any other format), falls back to ffmpeg conversion.
    """
    import soundfile as sf

    path = Path(audio_path)
    tmp_wav = None

    try:
        if path.suffix.lower() in (".mp3", ".m4a", ".aac", ".wma", ".opus"):
            raise ValueError("use ffmpeg")
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        # Convert to temp WAV via ffmpeg
        print(f"Converting {path.suffix} to WAV via ffmpeg...")
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-f", "wav", tmp_wav.name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("ffmpeg failed. Is ffmpeg installed and on PATH?", file=sys.stderr)
            print(result.stderr[-500:], file=sys.stderr)
            sys.exit(1)
        audio, sr = sf.read(tmp_wav.name, dtype="float32", always_2d=False)
        os.unlink(tmp_wav.name)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != SAMPLE_RATE:
        print(f"Resampling {sr}Hz → {SAMPLE_RATE}Hz...")
        import scipy.signal
        n_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy.signal.resample(audio, n_samples)

    return audio


# ── WER ───────────────────────────────────────────────────────────────────────

def normalize(text: str) -> list[str]:
    """Lowercase and strip punctuation, return word list."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)
    return text.split()


def word_error_rate(hypothesis: str, reference: str) -> tuple[float, dict]:
    """Compute WER using dynamic programming (word-level edit distance)."""
    h = normalize(hypothesis)
    r = normalize(reference)

    if not r:
        return 0.0, {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_words": 0}

    # Edit distance matrix
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j],    # deletion
                                   d[i][j - 1],    # insertion
                                   d[i - 1][j - 1]) # substitution

    wer = d[len(r)][len(h)] / len(r)

    # Back-trace to count operation types
    i, j = len(r), len(h)
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and r[i-1] == h[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j-1] + 1:
            ins += 1; j -= 1
        else:
            dels += 1; i -= 1

    return wer, {"substitutions": subs, "deletions": dels,
                 "insertions": ins, "ref_words": len(r)}


# ── ASR ───────────────────────────────────────────────────────────────────────

def load_asr():
    import onnx_asr
    print("Loading Parakeet ASR model...")
    t0 = time.time()
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v2")
    print(f"ASR loaded in {time.time() - t0:.1f}s\n")
    return model


def vad_chunks(audio: np.ndarray,
               min_chunk: float = DEFAULT_MIN_CHUNK,
               max_chunk: float = DEFAULT_MAX_CHUNK,
               silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
               silence_window: float = DEFAULT_SILENCE_WINDOW) -> list[tuple[np.ndarray, str]]:
    """Split audio using Voice Activity Detection — cut at silence, not fixed intervals.

    Returns list of (chunk_array, cut_reason) tuples.
    """
    silence_samples = int(SAMPLE_RATE * silence_window)
    min_samples = int(SAMPLE_RATE * min_chunk)
    max_samples = int(SAMPLE_RATE * max_chunk)

    chunks = []
    pos = 0

    while pos < len(audio):
        remaining = len(audio) - pos
        if remaining < SAMPLE_RATE:  # sub-1s tail — skip
            break

        # Scan forward from min_chunk looking for silence
        window_start = pos + min_samples
        window_end   = pos + max_samples

        cut_at = None
        cut_reason = ""

        i = window_start
        while i < min(window_end, len(audio)):
            tail_start = max(0, i - silence_samples)
            tail = audio[tail_start:i]
            rms = float(np.sqrt(np.mean(tail ** 2)))
            if rms < silence_threshold:
                cut_at = i
                cut_reason = f"silence rms={rms:.4f}"
                break
            i += int(SAMPLE_RATE * 0.1)  # step 100ms at a time

        if cut_at is None or cut_at > pos + max_samples:
            cut_at = min(pos + max_samples, len(audio))
            cut_reason = f"max cap"

        chunk = audio[pos:cut_at]
        chunks.append((chunk, cut_reason))
        pos = cut_at

    return chunks


def transcribe_file(audio_path: str, asr_model,
                    chunk_duration: int = None,
                    use_vad: bool = False,
                    min_chunk: float = DEFAULT_MIN_CHUNK,
                    max_chunk: float = DEFAULT_MAX_CHUNK,
                    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD) -> list[str]:
    print(f"Reading: {audio_path}")
    audio = load_audio(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    total_seconds = len(audio) / SAMPLE_RATE

    if use_vad:
        chunk_list = vad_chunks(audio, min_chunk=min_chunk, max_chunk=max_chunk,
                                silence_threshold=silence_threshold)
        print(f"Duration: {total_seconds:.1f}s  |  VAD chunks: {len(chunk_list)}"
              f"  |  min={min_chunk}s  max={max_chunk}s  threshold={silence_threshold}\n")
    else:
        cd = chunk_duration or DEFAULT_CHUNK_DURATION
        chunk_size = SAMPLE_RATE * cd
        raw_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
        chunk_list = [(c, f"fixed {cd}s") for c in raw_chunks]
        print(f"Duration: {total_seconds:.1f}s  |  Fixed chunks: {len(chunk_list)}  ({cd}s each)\n")

    segments = []
    t0 = time.time()
    elapsed_audio = 0
    for chunk, reason in chunk_list:
        if len(chunk) < SAMPLE_RATE:
            elapsed_audio += len(chunk) / SAMPLE_RATE
            continue
        result = asr_model.recognize(chunk)
        text = result.text.strip() if hasattr(result, "text") else str(result).strip()
        ts = int(elapsed_audio)
        elapsed_audio += len(chunk) / SAMPLE_RATE
        if text:
            segments.append(text)
            print(f"  [{ts // 60:02d}:{ts % 60:02d}] ({reason}) {text}")

    elapsed = time.time() - t0
    speed = total_seconds / elapsed if elapsed > 0 else 0
    print(f"\nTranscribed {total_seconds:.0f}s in {elapsed:.1f}s  ({speed:.0f}x realtime)")
    return segments





# ── Text alignment ────────────────────────────────────────────────────────────

def align_text_to_audio(audio_path: str, full_text_path: str, asr_model,
                        anchor_words: int = 60) -> str:
    """Transcribe audio and extract the matching portion from a full chapter text.

    Uses word-anchor matching: takes the first and last N words of the transcript,
    slides them across the full text to find the best-matching window, then extracts
    the corresponding passage from the original (un-normalized) text.

    Handles ~2% WER noise in the transcript — matched by set overlap, not exact match.
    """
    print("Step 1/2 — Transcribing audio to find position in full text...")
    segments = transcribe_file(audio_path, asr_model, use_vad=True)
    transcript = " ".join(segments)
    trans_words = normalize(transcript)
    print(f"  Transcript: {len(trans_words)} words\n")

    full_text = Path(full_text_path).read_text(encoding="utf-8")

    # Tokenize full text, tracking each word's character position
    token_matches = list(re.finditer(r"[a-zA-Z']+", full_text))
    full_words = [m.group().lower() for m in token_matches]
    print(f"Step 2/2 — Searching {len(full_words)}-word chapter for match...")

    if len(trans_words) >= len(full_words):
        print("  Transcript covers full text — returning as-is.")
        return full_text

    # ── Find start: slide start anchor across full text ──
    start_anchor = set(trans_words[:anchor_words])
    best_start_score, best_start_idx = -1, 0
    for i in range(len(full_words) - anchor_words):
        score = len(start_anchor & set(full_words[i:i + anchor_words]))
        if score > best_start_score:
            best_start_score, best_start_idx = score, i

    # ── Find end: slide end anchor, searching forward from expected end ──
    end_anchor = set(trans_words[-anchor_words:])
    expected_end = best_start_idx + len(trans_words)
    search_from = max(best_start_idx + len(trans_words) - anchor_words * 3, best_start_idx)
    best_end_score, best_end_idx = -1, min(expected_end + anchor_words, len(full_words))
    for i in range(search_from, min(len(full_words) - anchor_words, expected_end + anchor_words * 3)):
        score = len(end_anchor & set(full_words[i:i + anchor_words]))
        if score > best_end_score:
            best_end_score, best_end_idx = score, i + anchor_words

    # ── Map word indices back to character offsets ──
    start_char = token_matches[best_start_idx].start()
    end_char   = token_matches[min(best_end_idx, len(token_matches) - 1)].end()

    # Extend to nearest sentence boundary for cleaner reference text
    preceding = full_text[:start_char]
    last_break = max(preceding.rfind(".\n"), preceding.rfind(". "), preceding.rfind("\n\n"))
    if last_break > max(0, start_char - 300):
        start_char = last_break + 1

    following = full_text[end_char:]
    next_break = following.find(". ")
    if 0 <= next_break < 300:
        end_char += next_break + 2

    extracted = full_text[start_char:end_char].strip()

    print(f"\n  Start anchor: {best_start_score}/{anchor_words} words matched")
    print(f"  End anchor:   {best_end_score}/{anchor_words} words matched")
    print(f"  Extracted {len(extracted.split())} words from {len(full_words)}-word chapter")
    print(f"\n  Begins: \"{extracted[:80]}...\"")
    print(f"  Ends:   \"...{extracted[-80:]}\"")

    return extracted


# ── Translation ───────────────────────────────────────────────────────────────

def load_translator():
    import ctranslate2
    import sentencepiece as spm

    print("\nLoading NLLB translation model...")
    t0 = time.time()
    translator = ctranslate2.Translator(NLLB_MODEL_DIR, device="auto")
    sp = spm.SentencePieceProcessor(os.path.join(NLLB_MODEL_DIR, "sentencepiece.bpe.model"))
    print(f"Translator loaded in {time.time() - t0:.1f}s\n")
    return translator, sp


def translate_lines(lines: list[str], target_lang: str, translator, sp) -> list[str]:
    entry = NLLB_LANG_MAP.get(target_lang)
    if not entry:
        print(f"Unknown language code: {target_lang}", file=sys.stderr)
        sys.exit(1)
    nllb_code, lang_name = entry
    print(f"Translating to {lang_name} ({target_lang})...\n")

    results = []
    for line in lines:
        if not line.strip():
            results.append("")
            continue
        tokens = sp.encode(line, out_type=str)
        input_tokens = ["eng_Latn"] + tokens + ["</s>"]
        output = translator.translate_batch(
            [input_tokens],
            target_prefix=[[nllb_code]],
            max_batch_size=1,
            beam_size=4,
        )
        output_tokens = output[0].hypotheses[0][1:]  # skip language token
        translated = sp.decode(output_tokens)
        results.append(translated)
        print(f"  {translated}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenEar pipeline test — transcribe and/or translate audio/text files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", choices=["transcribe", "translate", "both", "align"])
    parser.add_argument("input",
                        help="Audio file (transcribe/both/align) or text file (translate)")
    parser.add_argument("reference_text", nargs="?",
                        help="Full chapter text file (align mode only)")
    parser.add_argument("-l", "--lang", default="ko",
                        help="Target language code, e.g. ko, es, fr (default: ko)")
    parser.add_argument("-o", "--output",
                        help="Output file — if omitted, prints to stdout")
    parser.add_argument("--wer", metavar="REFERENCE",
                        help="Reference transcript file — prints Word Error Rate after transcription")
    parser.add_argument("--chunk", type=int, default=None,
                        help=f"Fixed chunk size in seconds (default: {DEFAULT_CHUNK_DURATION})")
    parser.add_argument("--vad", action="store_true",
                        help="Use VAD chunking (cut at silence) instead of fixed intervals")
    parser.add_argument("--min-chunk", type=float, default=DEFAULT_MIN_CHUNK,
                        help=f"VAD: minimum seconds before cutting (default: {DEFAULT_MIN_CHUNK})")
    parser.add_argument("--max-chunk", type=float, default=DEFAULT_MAX_CHUNK,
                        help=f"VAD: hard cut at this many seconds (default: {DEFAULT_MAX_CHUNK})")
    parser.add_argument("--silence", type=float, default=DEFAULT_SILENCE_THRESHOLD,
                        help=f"VAD: RMS silence threshold (default: {DEFAULT_SILENCE_THRESHOLD})")
    args = parser.parse_args()

    # ── Align mode: extract matching portion from full chapter text ──
    if args.mode == "align":
        if not args.reference_text:
            print("align mode requires a full chapter text file as second argument.", file=sys.stderr)
            sys.exit(1)
        asr = load_asr()
        extracted = align_text_to_audio(args.input, args.reference_text, asr)
        if args.output:
            Path(args.output).write_text(extracted, encoding="utf-8")
            print(f"\nTrimmed reference written to {args.output}")
        else:
            print("\n" + "=" * 60)
            print(extracted)
        return

    lines = []

    if args.mode in ("transcribe", "both"):
        asr = load_asr()
        lines = transcribe_file(args.input, asr,
                                chunk_duration=args.chunk,
                                use_vad=args.vad,
                                min_chunk=args.min_chunk,
                                max_chunk=args.max_chunk,
                                silence_threshold=args.silence)

    if args.mode == "translate":
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()

    if args.mode in ("translate", "both"):
        translator, sp = load_translator()
        lines = translate_lines(lines, args.lang, translator, sp)

    output_text = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"\nWritten to {args.output}")
    else:
        print("\n" + "=" * 60)
        print(output_text)

    # WER — only meaningful for transcribe mode with a reference
    if args.wer and args.mode in ("transcribe", "both"):
        reference = Path(args.wer).read_text(encoding="utf-8")
        wer, counts = word_error_rate(output_text, reference)
        print("\n" + "=" * 60)
        print(f"  Word Error Rate : {wer * 100:.1f}%")
        print(f"  Substitutions   : {counts['substitutions']}")
        print(f"  Deletions       : {counts['deletions']}")
        print(f"  Insertions      : {counts['insertions']}")
        print(f"  Reference words : {counts['ref_words']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
