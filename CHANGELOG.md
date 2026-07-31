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

## 0.9.0 — Segment IDs make source/translation pairing exact

*Fixes the logging mismatch that forced translation scoring to guess. Adds Spanish
verification: 95.0% adequate, against Korean's 77.5%.*

> **Note on release headings.** From this release forward, headings describe the change
> rather than naming a theme (Jonathan's ruling, 2026-07-31). A changelog is read by
> someone deciding whether they care, and "Second Language" does not help them decide.
> Earlier entries keep their names — those releases had genuine themes.

### Fixed
- **The text logs were never 1:1, and nothing said so.** `source-en.txt` recorded every
  incoming speech-recognition *fragment*, while `translated-<lang>.txt` recorded one line
  per completed *sentence*. The translation buffer merges several fragments into a
  sentence before translating, so a ten-fragment stretch produced eight translations and
  the files silently disagreed. The pairing was destroyed at write time and no downstream
  tool could recover it — which is why scoring had to guess by character length, and why
  it guessed wrong on real data.

  Both sides now log the **same unit** — the completed sentence — each stamped with a
  shared, monotonic **segment ID**. Pairing is exact by construction rather than
  reconstructed afterwards.
- **A guard that was documented but never implemented.** `check_untranslated` claimed in
  its own docstring to skip same-script language pairs, and did not. Every test until now
  was English→Korean, where the scripts differ, so it had never fired. The first
  English→Spanish run flagged all eight segments as "possibly untranslated" — Latin text
  in a Latin-script target — and reported **0.0% structural integrity for a translation
  that scored 95% adequate.**

### Added
- **`--rebuffer-source`** — replays the server's sentence-buffering rule over a legacy
  fragment-level log, making historical runs comparable instead of unusable. This is what
  turned the March 2026 Korean log from "80% delivery, two segments vanished" into its
  true reading: 100% delivery, one damaged segment.
- Three more calibration controls covering same-script pairs and confirming cross-script
  detection still fires. **Twenty-one controls total.**

### Measured
Same source, same model, same parameters:

| Language | Delivery | Structural integrity | Adequacy |
|---|---|---|---|
| Spanish | 100% | 100% | **95.0%** |
| Korean | 100% | 87.5% | **77.5%** |

- The gap is the expected one: Spanish is close to English in structure, Korean is
  verb-final and must restructure — and restructuring is where clauses fall out. Spanish
  kept the clause Korean dropped, kept "shadow" in "valley of the shadow of death", and
  chose a register-neutral verb for "he's using us" where Korean chose one closer to
  *exploiting*.
- **The offline path was validated against production before any of this was trusted:**
  re-translating the same source to Korean reproduced the March service log byte for byte.

---

## 0.8.0 — "Accuracy Check"

*A percentage that actually means "how much of the sermon got through."*

0.7.0 built the gauge and pointed it at delivery — did every source segment produce
output of a plausible size? That catches content vanishing, but a fluent mistranslation
sails through it at 100%. This release measures the thing itself: **did the meaning
arrive?**

The judge deliberately lives *outside* the tool. The harness emits a blind worksheet, a
judge fills it in, the harness ingests and scores. That keeps the result auditable (the
questions and answers are files you can read), keeps the judge swappable (a capable model
today, a local one later, a native speaker if we ever get one), and keeps whoever built
the pipeline from being its own unblinded grader.

### Added
- **Blind judging worksheet** (`--worksheet-out`). Segments shuffled, system identity
  stripped, hidden calibration controls salted in. A judge scores each pair 0–100 with a
  category and a one-line note.
- **Worksheet ingest and scoring** (`--worksheet-in`). Produces an adequacy percentage,
  a category breakdown, and a per-segment list of everything that scored poorly.
- **Judge calibration that needs no gold translation.** We cannot manufacture a known-
  *correct* translation for an arbitrary language, but we can manufacture known-*wrong*
  ones from the data itself: a source paired with a different segment's translation
  (must score near zero), and a source paired with the first 40% of its own translation
  (must score well below the real segments). A judge that fails either is rubber-stamping
  or blind to omission, and **the run is voided rather than reported** — a number from a
  broken instrument is worse than no number.
- Five more self-test controls covering the validation itself: a judge scoring everything
  100 is rejected, a judge blind to truncation is rejected, a discriminating judge is
  accepted, and an unfinished worksheet is rejected. Fourteen controls total.

### Notes
- **First adequacy figure on real data:** the 2026-03-26 Korean service log scores
  **60.6% adequate**, judge calibration passed.
- **That figure is not translator accuracy, and the report says so.** Two segments scored
  `wrong` because the heuristic aligner handed the judge mis-paired source and target;
  the judge correctly scored a bad *pairing* near zero. Layer 2 inherits Layer 1's
  uncertainty. Excluding those, the same judgments give roughly 77%.
- **Defects found that no string metric can see:** `이용하다` for "he's using us" (adequate,
  but connotes *exploiting* — wrong register for a sermon); 은혜의 의롭다는 것은 garbling the
  theological term "justifying grace"; 어둠 for "shadow of death", losing the Psalm 23
  allusion.
- **The bottleneck is now alignment,** not judging. When segment counts disagree, pairings
  are guessed by character length, and length cannot see meaning. Fixing that is the next
  real piece of work — ahead of any further layers.
- No change to captioning, translation, or the client. Still bench equipment.

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
