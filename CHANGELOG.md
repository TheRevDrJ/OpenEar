# Changelog

All notable changes to OpenEar.

This file records **released history only** — what happened, not what's planned. Plans
live where plans are worked, not here.

Versioning is [semver](https://semver.org/): any new feature earns a **minor**, any fix
earns a **patch**. Versions are labels, not a budget — a stingy history hides the shape
of the work.

> **On earlier releases:** this changelog starts at 0.6.0. Versions 0.1 through 0.5.1
> predate it and are not reconstructed here — retrofitting a changelog from commit
> history arrives thinner than the truth and reads like a guess, because it is one. The
> git log is the record for that stretch.

---

## 0.7.0 — "Instrument"

*We can measure translation quality now, instead of having opinions about it.*

OpenEar has had translation accuracy figures before — roughly 97% for Spanish, 85% for
Korean. They were real, and they were produced by a person reading parallel text side by
side once. That meant they could not be re-run, could not be compared against a later
version, and could not answer the only question that matters when you change something:
*did that make it better or worse?*

This release turns that measurement into a tool that writes its answer to a file.

### Added
- **`score_translation.py`** — translation integrity scoring. Takes source and translated
  text, one segment per line (exactly what `--log-text` already writes), and reports how
  much of the source survived. Emits markdown and JSON.
  - **No dependencies.** Standard library only — no model, no network, no API key, and
    **no reference translation required.** It runs anywhere, forever.
  - Reports **delivery rate** (pure arithmetic on segment counts) separately from
    **structural integrity** (which depends on per-segment alignment), because the first
    is always trustworthy and the second sometimes is not.
  - Detects dropped segments, merged segments, suspiciously short output (the fingerprint
    of a lost clause), text left untranslated in the source script, and empty output.
- **`--self-test`** — nine calibration controls run against deliberately corrupted
  fixtures. If the tool stops detecting a defect it is built to catch, it fails loudly and
  declares its own numbers untrustworthy. An evaluation tool that has never been shown to
  catch a known defect is an opinion generator, not an instrument.

### Notes
- **First measured baseline.** Our own 2026-03-26 Korean service log scores **80.0%
  delivery, 60.0% structural integrity**. Ten English segments produced eight Korean
  ones, and the clause *"into the richness of a good creation that God is actively
  working to restore"* never reached the listener. Nothing reported a problem at the time.
- **Scope, stated honestly.** This measures *delivery, not quality*. A fluent
  mistranslation scores 100%. It is a floor, not a grade. Per-segment meaning judgment is
  the next layer.
- **A known limitation, measured rather than assumed.** When segment counts disagree,
  pairings are heuristic and can be wrong — character length cannot see meaning, and on
  the log above the correct alignment scores *worse* than an incorrect one. The tool says
  so in its own report rather than presenting guesses as findings.
- No change to captioning, translation, or the client. This is bench equipment.

---

## 0.6.0 — "Landfall"

*The project comes ashore on macOS.*

OpenEar was built Windows-only and had never been opened on a Mac. As of this release,
**macOS is a supported development platform**: the repo is edited, versioned, and
documented there, while the app itself continues to run on the CUDA box. Development and
execution now live on different shores, deliberately — the models require CUDA, and no
amount of unified memory substitutes for it.

This release is mostly about making that split safe rather than accidental.

### Added
- **macOS as a supported development platform.** Repo work, docs, and client editing all
  happen there now. The app remains Windows-only to run; that is by design, not neglect.
- **`.gitattributes`** pinning the line-ending policy for every platform.

### Fixed
- **Windows batch and PowerShell scripts could be checked out with LF line endings.**
  `openear.bat` and `setup.bat` were stored LF in the repository, so a fresh clone on a
  Windows machine without `core.autocrlf=true` produced files `cmd.exe` cannot parse.
  This does not fail cleanly: labels and `goto` break, execution falls through into
  whatever branch follows, and variables come back empty — which can turn a
  narrowly-targeted process kill into an indiscriminate one. `*.bat` and `*.ps1` are now
  forced to CRLF on checkout on every platform, so the hazard cannot recur regardless of
  how any individual machine's git is configured.
- **`kill_openear.ps1` was LF on disk** — the same hazard, already present in a script
  whose entire job is stopping processes. Now CRLF.
- **Whole-tree line-ending churn.** The working tree reported 2,902 changed lines across
  12 files containing zero actual content difference, an artifact of a CRLF working copy
  meeting an LF repository. Normalized.

### Changed
- `models/` (~12GB of downloaded model weights), `.DS_Store`, and `openear.pid` (runtime
  state written at server start) are no longer trackable.

### Notes
- No change to captioning, translation, the client, or the admin console. Behavior on
  Windows is identical to 0.5.1.
- Verified on the reference box: Parakeet ASR and NLLB-200 3.3B both load, CTranslate2
  reports 2 CUDA devices, and the server serves on port 80 to phones on the LAN.
