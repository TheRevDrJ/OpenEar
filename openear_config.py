# OpenEar — Real-time AI captioning and translation for churches
# Copyright (c) 2026 TheRevDrJ
# Licensed under AGPL-3.0 — see LICENSE file for details
"""
openear_config.py — which half of OpenEar this machine runs, and which models.

OpenEar has two modes, chosen once per machine by setup.bat:

  captions      English captions only. Loads the speech model, which runs on
                the CPU. Loads NO translation model and never touches a
                graphics card, so it can share a PC with streaming or video
                software without competing for video memory.
  translation   Captions plus live translation. Also loads NLLB-200 onto an
                NVIDIA GPU, which takes about 4.6 GB of video memory.

THE CHOICE LIVES IN mode.json beside this file: {"translation": true} or
{"translation": false}. setup.bat writes it; every server start reads it. It is
per-machine, so it is gitignored. A boolean was chosen over a mode name because
a boolean cannot be misspelled.

ABSENT OR UNREADABLE MEANS CAPTIONS ONLY. That is deliberate. The failure this
file exists to prevent is a machine loading 4.6 GB of translation model it never
asked for and taking video memory from whatever else runs on it — so the safe
direction is the one that loads less. A machine that wanted translation and
lost it says so on the admin page and in the log; a machine that wanted captions
and silently got translation would say nothing at all.
@decision:silver 2026-09-25 — the default for an install that predates modes.
Everything else about how modes are chosen is @decision:gold 2026-09-25.

A launch flag overrides the file for a single run, for testing:
    --captions-only     --translation     (never both)

CALLED BY:
    server.py            resolve_mode() at startup
    download_models.py   decides whether to fetch the translation model
    setup.bat            --write, to record the choice after asking
    openear.bat          prints the mode a start is about to use

RUN DIRECTLY (standard library only, so it works before the venv exists):
    python openear_config.py [--captions-only | --translation]
        prints the mode a server started with those flags would use
    python openear_config.py --current
        prints exactly one word — captions, translation, or none — from
        mode.json alone, for setup.bat to read into a variable
    python openear_config.py --write captions|translation
        records the choice in mode.json

Console output is plain ASCII on purpose: it is printed into cmd.exe, whose
code page mangles anything else.
"""

import json
import sys
from pathlib import Path

INSTALL_DIR = Path(__file__).resolve().parent
MODE_FILE = INSTALL_DIR / "mode.json"

CAPTIONS = "captions"
TRANSLATION = "translation"

# Appended to the source when no valid choice exists. A constant, so the server can
# strip it back off exactly - splitting on a comma would cut an OS error message.
DEFAULT_SUFFIX = ", so captions only is the default"

CAPTIONS_FLAG = "--captions-only"
TRANSLATION_FLAG = "--translation"

# ── The models. One definition, read by the server AND the downloader. ───────
# These used to be written twice, and had already drifted: the downloader pinned
# NLLB to a commit while the server's fallback download fetched whatever the
# repo's main branch held that day. Two copies of a pin is no pin.

# NVIDIA Parakeet (onnx_asr). Natively punctuated and capitalized. CPU only.
PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"

# Meta NLLB-200 3.3B, int8, via CTranslate2. Needs CUDA.
# Pinned to a commit so an upstream change cannot alter production on its own
# schedule. To update: verify the new revision end to end, then change this.
# Commits: https://huggingface.co/entai2965/nllb-200-3.3B-ctranslate2/commits/main
NLLB_REPO = "entai2965/nllb-200-3.3B-ctranslate2"
NLLB_REVISION = "33acf12b9572e946facdf7f3cb3ebcf47ba52286"  # pinned 2026-05-03
# The size of model.bin at that revision, so a present-but-truncated copy is caught
# as incomplete rather than failing inside CTranslate2 with an error that names
# neither cause. ⛔ Change it together with NLLB_REVISION, or every start will
# report the new model as incomplete.
NLLB_MODEL_BIN_BYTES = 13_387_884_764
NLLB_MODEL_DIR = INSTALL_DIR / "models" / "nllb-3.3b-ct2"


class ModeError(Exception):
    """A mode request that cannot be honoured, such as both flags at once."""


def describe(mode: str) -> str:
    """The words a person reads for a mode, in the log, the console and admin."""
    return "captions + translation" if mode == TRANSLATION else "captions only"


def read_mode_file(path: Path = MODE_FILE):
    """Return (mode, where) from mode.json, or (None, why) if there is no usable choice.

    Never raises. Every way the file can be missing or wrong comes back as None
    plus a sentence saying which way, so the caller can report it honestly
    instead of guessing.
    """
    try:
        # utf-8-sig, so a file saved by Notepad with a byte-order mark still reads.
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, "no mode.json"
    except OSError as e:
        return None, f"mode.json could not be read ({e})"
    except ValueError:
        # Not UTF-8 at all — e.g. written by PowerShell 5.1's `>`, which is UTF-16.
        # UnicodeDecodeError is a ValueError, not an OSError; left uncaught it would
        # stop the server from starting at all.
        return None, "mode.json is not UTF-8 text"

    try:
        data = json.loads(text)
    except ValueError:
        return None, "mode.json is not valid JSON"

    value = data.get("translation") if isinstance(data, dict) else None
    # bool only: 1, "yes" and "true" are rejected rather than interpreted.
    if not isinstance(value, bool):
        return None, 'mode.json has no true/false "translation" value'

    return (TRANSLATION if value else CAPTIONS), "mode.json"


def resolve_mode(argv, path: Path = MODE_FILE):
    """Return (mode, source) for a server started with these arguments.

    Precedence: a launch flag, then mode.json, then captions only.
    Raises ModeError if both flags are given — there is no sensible reading of
    that, and picking one would hide the mistake.
    """
    wants_captions = CAPTIONS_FLAG in argv
    wants_translation = TRANSLATION_FLAG in argv

    if wants_captions and wants_translation:
        raise ModeError(f"{CAPTIONS_FLAG} and {TRANSLATION_FLAG} cannot be used together")
    if wants_captions:
        return CAPTIONS, f"the {CAPTIONS_FLAG} flag"
    if wants_translation:
        return TRANSLATION, f"the {TRANSLATION_FLAG} flag"

    mode, where = read_mode_file(path)
    if mode is not None:
        return mode, where
    return CAPTIONS, f"{where}{DEFAULT_SUFFIX}"


def write_mode(mode: str, path: Path = MODE_FILE) -> None:
    """Record this machine's mode in mode.json."""
    if mode not in (CAPTIONS, TRANSLATION):
        raise ModeError(f"unknown mode {mode!r}: use {CAPTIONS} or {TRANSLATION}")
    path.write_text(
        json.dumps({"translation": mode == TRANSLATION}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv) -> int:
    if "--current" in argv:
        # "none" when there is no usable choice, so setup.bat can tell a first
        # install from a machine that already chose.
        mode, _ = read_mode_file()
        print(mode or "none")
        return 0

    if "--write" in argv:
        i = argv.index("--write")
        value = argv[i + 1] if i + 1 < len(argv) else ""
        try:
            write_mode(value)
        except (ModeError, OSError) as e:
            print(f"  [FAIL] Could not record the mode: {e}")
            return 1
        print(f"  [OK] Mode recorded: {describe(value)}  (mode.json)")
        return 0

    try:
        mode, source = resolve_mode(argv)
    except ModeError as e:
        print(f"  ERROR: {e}")
        return 2
    print(f"  Mode: {describe(mode)}  (from {source})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
