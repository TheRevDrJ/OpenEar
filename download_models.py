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

HUGGING FACE THROTTLES ANONYMOUS DOWNLOADS, and every church downloads
anonymously. It answers "429 Too Many Requests" and the download dies part way.
So each download is retried with waits that grow to five minutes (HF's limits are
counted in five-minute windows), resuming where it stopped, until the model is
COMPLETE - the translation model is checked against its pinned size, not trusted
because a call returned. Only an answer that will never change - the model or
revision does not exist, or access is refused - stops the retrying early.
Without this, setup printed a traceback, said "setup complete", and left the user
to notice a warning and run it again. @decision:gold 2026-11-11

CALLED BY: setup.bat. Safe to run by hand; a model already present is not fetched
again. Exits non-zero if a model could not be completed.
"""
import os
import sys
import time

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

# Seconds to wait before each retry. Eight tries, about 24 minutes of waiting in
# all - long enough to outlast a throttled window several times over.
RETRY_WAITS = (30, 60, 120, 300, 300, 300, 300)

# Answers that will be the same on every try, so retrying only wastes the user's
# time. Matched by EXACT class name: LocalEntryNotFoundError - what a throttled
# download raises - is a subclass of EntryNotFoundError, and must be retried.
PERMANENT_ERRORS = {"RepositoryNotFoundError", "RevisionNotFoundError",
                    "EntryNotFoundError", "GatedRepoError"}
PERMANENT_STATUS = {401, 403, 404}


class IncompleteDownload(Exception):
    """The download call returned, but what is on disk is not the whole model."""


def _chain(exc):
    """The exception and everything it was raised from."""
    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


def is_permanent(exc) -> bool:
    for e in _chain(exc):
        if type(e).__name__ in PERMANENT_ERRORS:
            return True
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in PERMANENT_STATUS:
            return True
    return False


def describe_failure(exc) -> str:
    """One plain line about why a try failed - never a traceback."""
    for e in _chain(exc):
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 429:
            return "Hugging Face is limiting downloads right now (429 Too Many Requests)"
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:200]


def with_retries(what: str, attempt, waits=RETRY_WAITS, sleep=time.sleep) -> None:
    """Run attempt() until it succeeds, waiting between tries. Raises on a permanent
    error, or with the last error once every wait is spent."""
    tries = len(waits) + 1
    for n in range(1, tries + 1):
        try:
            attempt()
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if is_permanent(e):
                print(f"  {what} cannot be downloaded: {describe_failure(e)}")
                raise
            if n == tries:
                print(f"  {what}: still failing after {tries} tries: {describe_failure(e)}")
                raise
            wait = waits[n - 1]
            print(f"  {describe_failure(e)}.")
            print(f"  Waiting {wait} seconds, then trying again (try {n + 1} of {tries}). "
                  f"The download resumes where it stopped.")
            sys.stdout.flush()
            sleep(wait)


def download_speech_model() -> None:
    """Parakeet — every mode needs it. onnx_asr fetches it on first load."""
    import onnx_asr

    print("  Downloading Parakeet speech model (~2.5GB)...")
    # CPU, as the server loads it. Left to its defaults, onnxruntime tries the GPU
    # providers first and prints a page of errors when their DLLs are absent.
    with_retries("The speech model", lambda: onnx_asr.load_model(
        openear_config.PARAKEET_MODEL, providers=["CPUExecutionProvider"]))
    print("  Parakeet model ready.")


def download_translation_model() -> None:
    """NLLB-200, at the pinned revision — translation mode only."""
    from huggingface_hub import snapshot_download

    def attempt():
        snapshot_download(
            openear_config.NLLB_REPO,
            local_dir=str(openear_config.NLLB_MODEL_DIR),
            revision=openear_config.NLLB_REVISION,
        )
        # A returned call is not a finished model: check what is on disk with the
        # same test the server applies before loading it.
        problem = openear_config.translation_model_problem()
        if problem:
            raise IncompleteDownload(problem)

    print("  Downloading NLLB-200 translation model (~13GB)...")
    with_retries("The translation model", attempt)
    print("  NLLB translation model ready.")


def main() -> int:
    # No launch flags here: setup.bat has just recorded the machine's choice, and
    # that recorded choice is what this download has to match.
    mode, source = openear_config.resolve_mode([])
    print(f"  Mode: {openear_config.describe(mode)}  (from {source})")

    try:
        download_speech_model()
        if mode == openear_config.TRANSLATION:
            download_translation_model()
        else:
            print("  Translation model skipped: this machine is set up for captions only.")
    except Exception:
        # with_retries has already said why, in one line; setup.bat prints what to do.
        return 1

    print("  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
