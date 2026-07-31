#!/usr/bin/env python3
"""score_translation.py — measure how much of the source survived translation.

WHY THIS EXISTS
---------------
OpenEar has had translation quality numbers before ("Spanish ~97%, Korean ~85%"),
but they were produced by a human reading parallel text side by side and were
never written down or made repeatable. That is a real evaluation method — judging
whether meaning survived is exactly the right question — but as a one-off it
cannot be re-run, cannot be diffed against a later version, and cannot tell you
whether a change made things better or worse.

This tool makes that measurement repeatable, and writes the result to a FILE
rather than reporting it through a person.

THE HARD PART, AND WHY THIS IS LAYERED
--------------------------------------
Translation has no single correct answer. The same sentence can be rendered a
dozen valid ways, so comparing characters against one "gold" translation
punishes correct paraphrase and rewards imitation of whoever wrote the gold.
BLEU and chrF do exactly that. They are useful for detecting REGRESSION against
a fixed reference, and misleading as an absolute grade.

So the tool is built in layers, weakest assumptions first:

  Layer 1 (this file, today) — DETERMINISTIC STRUCTURAL INTEGRITY.
      No judgment, no model, no reference translation, no dependencies. Asks only:
      did every piece of the source produce a piece of output, and is that output
      plausibly the right size and script? This cannot tell you whether a
      translation is GOOD. It reliably tells you when content was silently LOST,
      which is the failure that actually bit us.

  Layer 2 (next) — ADEQUACY JUDGMENT, per segment, reference-free.
      Meaning preservation scored by an independent judge. Handles paraphrase
      natively because it evaluates meaning rather than strings.

  Layer 3 (next) — REGISTER / DOMAIN FLAGS.
      Word choices that are adequate but wrong for a sermon. Real example from
      our own logs: "he's using us" rendered with 이용하다, which carries a sense
      closer to *exploiting*. No string metric catches that.

THE FAILURE THAT MOTIVATED LAYER 1
----------------------------------
From a real service log (text-logs/, 2026-03-26), English source vs Korean output:

    10 source segments in  ->  8 Korean segments out

Three source lines collapsed into one, and the clause "into the richness of a
good creation that God is actively working to restore" was dropped entirely. The
theological payload of the sentence never reached the Korean-speaking listener,
and nothing anywhere reported a problem. Layer 1 catches that with arithmetic.

USAGE
-----
    python score_translation.py --source SRC.txt --target TGT.txt --target-lang ko
    python score_translation.py --source SRC.txt --target TGT.txt --target-lang ko \
                                --report reports/run.md --json reports/run.json

Input format is one segment per line, which is exactly what OpenEar's --log-text
flag already writes (text-logs/source-en.txt, text-logs/translated-<lang>.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ── Tuning ────────────────────────────────────────────────────────────────────
#
# These are heuristic thresholds, deliberately loose. Layer 1 exists to catch
# gross structural loss, not to nitpick. A flag here means "a human or a judge
# should look at this segment," never "this is wrong."

# A translated segment shorter than this fraction of its expected length is
# suspected of dropping content. Expected length is derived per-run from the
# corpus-wide character ratio, so it self-calibrates to the language pair
# (Korean runs shorter than English; German runs longer).
SHORT_SEGMENT_RATIO = 0.55

# ...and longer than this suggests duplication or hallucinated padding.
LONG_SEGMENT_RATIO = 2.00

# Fraction of a target segment's letters that may share the SOURCE script before
# we suspect the segment was passed through untranslated. Only meaningful for
# cross-script pairs (en->ko, en->ar, en->ru). See note in check_untranslated().
MAX_SOURCE_SCRIPT_SHARE = 0.60


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Finding:
    """One thing worth a human's attention. Severity drives report ordering."""
    severity: str          # "critical" | "warning" | "info"
    kind: str              # short machine-readable slug
    detail: str            # human sentence
    source_index: int | None = None
    target_index: int | None = None


@dataclass
class Pair:
    """An aligned source/target unit. Indices are 1-based for human reporting."""
    op: str                        # "match" | "merge" | "split" | "dropped" | "added"
    source_indices: list[int] = field(default_factory=list)
    target_indices: list[int] = field(default_factory=list)
    source_text: str = ""
    target_text: str = ""


# ── Input ─────────────────────────────────────────────────────────────────────

def load_segments(path: Path) -> list[str]:
    """Read one segment per line, dropping blanks but preserving order."""
    if not path.exists():
        sys.exit(f"ERROR: no such file: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


# ── Script analysis ───────────────────────────────────────────────────────────

def script_of(char: str) -> str:
    """Best-effort Unicode script name for a single character.

    Uses unicodedata.name() prefixes rather than a script table, because the
    stdlib has no script property. Crude but adequate: we only need to tell
    'Latin' from 'Hangul'/'Cyrillic'/'Arabic'/'Han', not to be a full ICU.
    """
    if not char.isalpha():
        return "NONE"
    try:
        name = unicodedata.name(char)
    except ValueError:
        return "UNKNOWN"
    for script in ("LATIN", "HANGUL", "CJK", "HIRAGANA", "KATAKANA",
                   "CYRILLIC", "ARABIC", "HEBREW", "GREEK", "DEVANAGARI",
                   "THAI", "HANGZHOU"):
        if name.startswith(script):
            return "CJK" if script in ("CJK", "HANGZHOU") else script
    return "OTHER"


def dominant_script(text: str) -> str:
    """The most common alphabetic script in a string, or NONE."""
    counts: dict[str, int] = {}
    for ch in text:
        s = script_of(ch)
        if s != "NONE":
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "NONE"
    return max(counts, key=counts.get)


def script_share(text: str, script: str) -> float:
    """Fraction of alphabetic characters in `text` belonging to `script`."""
    total = sum(1 for ch in text if script_of(ch) != "NONE")
    if total == 0:
        return 0.0
    hits = sum(1 for ch in text if script_of(ch) == script)
    return hits / total


# ── Alignment ─────────────────────────────────────────────────────────────────

def align(source: list[str], target: list[str]) -> list[Pair]:
    """Align source segments to target segments by character length.

    OpenEar broadcasts one translation per completed source sentence, so a
    healthy run is 1:1 and alignment is trivial. It is when the counts DISAGREE
    that we need this — and disagreement is itself the headline defect, so the
    alignment exists to localise the damage, not to excuse it.

    This is a simplified Gale & Church: dynamic programming over character
    lengths, permitting 1:1 (match), 1:2 (split), 2:1 (merge), 1:0 (dropped) and
    0:1 (added). Length is a decent proxy across languages once you normalise by
    the corpus-wide ratio, and it needs no dictionary, no model, and no network.

    Cost is squared relative length error, so a mild size difference is cheap and
    a wild one is expensive. Skips carry a flat penalty tuned so the aligner
    prefers explaining a size mismatch as a merge rather than as a deletion —
    deletions should be reported only when nothing else fits.
    """
    src_len = [len(s) for s in source]
    tgt_len = [len(t) for t in target]
    total_src, total_tgt = sum(src_len) or 1, sum(tgt_len) or 1
    ratio = total_tgt / total_src          # expected target chars per source char

    SKIP_PENALTY = 6.0

    def cost(s_chars: int, t_chars: int) -> float:
        expected = s_chars * ratio
        if expected <= 0:
            return SKIP_PENALTY
        return ((t_chars - expected) / expected) ** 2

    # Maximum segments that may be consumed on either side by a single operation.
    #
    # THIS WAS ORIGINALLY 2 AND IT PRODUCED CONFIDENTLY WRONG OUTPUT. In our own
    # 2026-03-26 Korean log, three consecutive source segments collapsed into one
    # target line. A 2:1 ceiling cannot represent a 3:1 event, so the aligner did
    # the next-cheapest thing: it smeared the mismatch across the neighbouring
    # pairs, mis-pairing four segments and pointing the blame one line away from
    # the actual damage. The count was right and every pairing after it was wrong.
    #
    # Real merges follow ASR sentence-boundary noise and can run several segments
    # long, so the ceiling must exceed the worst case we expect, not the typical
    # one. Widening costs a little compute (the DP is O(n*m*SPAN^2) on inputs of a
    # few hundred lines) and buys correct localisation.
    MAX_SPAN = 4

    n, m = len(source), len(target)
    INF = float("inf")
    # dp[i][j] = best cost having consumed i source and j target segments
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back: dict[tuple[int, int], tuple[int, int, str]] = {}
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == INF:
                continue
            base = dp[i][j]

            # General si:tj operations, si and tj each from 1..MAX_SPAN.
            # A penalty proportional to how far the span is from 1:1 keeps the
            # aligner honest: it will only claim a merge or split when the sizes
            # genuinely demand one.
            for si in range(1, MAX_SPAN + 1):
                if i + si > n:
                    break
                for tj in range(1, MAX_SPAN + 1):
                    if j + tj > m:
                        break
                    s_chars = sum(src_len[i:i + si])
                    t_chars = sum(tgt_len[j:j + tj])
                    span_penalty = (si - 1) + (tj - 1)
                    c = base + cost(s_chars, t_chars) + span_penalty
                    if c < dp[i + si][j + tj]:
                        dp[i + si][j + tj] = c
                        op = "match" if (si, tj) == (1, 1) else (
                            "merge" if tj == 1 else "split" if si == 1 else "tangle")
                        back[(i + si, j + tj)] = (i, j, op)

            # 1:0 — source segment produced nothing
            if i < n:
                c = base + SKIP_PENALTY
                if c < dp[i + 1][j]:
                    dp[i + 1][j] = c
                    back[(i + 1, j)] = (i, j, "dropped")
            # 0:1 — target segment with no source
            if j < m:
                c = base + SKIP_PENALTY
                if c < dp[i][j + 1]:
                    dp[i][j + 1] = c
                    back[(i, j + 1)] = (i, j, "added")

    # Walk the backpointers home.
    pairs: list[Pair] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, op = back[(i, j)]
        s_idx = list(range(pi + 1, i + 1))
        t_idx = list(range(pj + 1, j + 1))
        pairs.append(Pair(
            op=op,
            source_indices=s_idx,
            target_indices=t_idx,
            source_text=" ".join(source[k - 1] for k in s_idx),
            target_text=" ".join(target[k - 1] for k in t_idx),
        ))
        i, j = pi, pj
    pairs.reverse()
    return pairs


# ── Checks ────────────────────────────────────────────────────────────────────

def check_structure(source: list[str], target: list[str],
                    pairs: list[Pair]) -> list[Finding]:
    """Segment-count integrity — the check that would have caught our real bug."""
    findings: list[Finding] = []

    if len(source) != len(target):
        findings.append(Finding(
            severity="critical",
            kind="segment_count_mismatch",
            detail=(f"{len(source)} source segments produced {len(target)} target "
                    f"segments. OpenEar emits one translation per completed source "
                    f"sentence, so a healthy run is 1:1. A mismatch means content "
                    f"was merged or lost in transit."),
        ))

    # When the counts disagree, every pairing below is a GUESS. See the note on
    # alignment_confidence() for why length-based alignment cannot be trusted
    # here, and why we hedge rather than tune it until one example passes.
    exact = len(source) == len(target)
    guess_prefix = "" if exact else "Heuristic alignment suggests "
    op_severity = "warning" if exact else "info"

    for p in pairs:
        if p.op == "dropped":
            findings.append(Finding(
                severity="critical" if exact else "warning",
                kind="segment_dropped",
                detail=(f"{guess_prefix}no translation was produced for: "
                        f"\"{p.source_text[:120]}\""),
                source_index=p.source_indices[0] if p.source_indices else None,
            ))
        elif p.op == "added":
            findings.append(Finding(
                severity=op_severity, kind="segment_unsourced",
                detail=(f"{guess_prefix}this target segment has no matching source: "
                        f"\"{p.target_text[:120]}\""),
                target_index=p.target_indices[0] if p.target_indices else None,
            ))
        elif p.op in ("merge", "tangle"):
            findings.append(Finding(
                severity=op_severity, kind="segments_merged",
                detail=(f"{guess_prefix}source segments {p.source_indices} were "
                        f"collapsed into one target segment. Merging is where clauses "
                        f"go missing — compare this pair by eye."),
                source_index=p.source_indices[0],
                target_index=p.target_indices[0] if p.target_indices else None,
            ))
    return findings


def alignment_confidence(source: list[str], target: list[str]) -> str:
    """Whether the per-segment pairings can be trusted: "exact" or "heuristic".

    WHY THIS EXISTS — a real failure, measured, not hypothetical.

    OpenEar emits one translation per completed source sentence. When the counts
    match, pairing by index is correct by construction and needs no cleverness.

    When they DON'T match, we fall back to length-based dynamic programming, and
    on our own 2026-03-26 Korean log it got the answer wrong. The true alignment
    was 1:1 through segment 5, then a 3:1 merge of segments 6-8 into target 6.
    The aligner instead proposed (4,5)->4, (6,7)->5, 8->6 — four segments
    mis-paired, and the blame pointed one line away from the real damage.

    It was not a bug in the search. Scored under the cost function, the CORRECT
    alignment costs ~2.50 and the WRONG one ~2.34, so the aligner faithfully
    returned the cheaper answer. Character length simply cannot distinguish them;
    telling them apart requires knowing that target 6 MEANS "a journey through
    the valleys". That is semantic work, and it belongs to Layer 2.

    Widening the span limit did not help, and tuning the penalties until this one
    example came out right would be fitting the instrument to a sample of one.
    So instead we label the uncertainty and let the report hedge. A tool that
    reports confident pairings it cannot support is worse than one that reports
    a count and admits the rest is a guess.
    """
    return "exact" if len(source) == len(target) else "heuristic"


def check_lengths(pairs: list[Pair]) -> list[Finding]:
    """Flag matched pairs whose size is far from the corpus-wide norm.

    A short target is the fingerprint of a dropped clause — the segment survived,
    but part of its content did not. This is the check that would have caught the
    "richness of a good creation" omission even had the segment counts matched.
    """
    matched = [p for p in pairs if p.op == "match" and p.source_text and p.target_text]
    if len(matched) < 3:
        return []      # too little data for a meaningful norm

    total_src = sum(len(p.source_text) for p in matched) or 1
    total_tgt = sum(len(p.target_text) for p in matched)
    ratio = total_tgt / total_src

    findings: list[Finding] = []
    for p in matched:
        expected = len(p.source_text) * ratio
        if expected <= 0:
            continue
        actual = len(p.target_text) / expected
        if actual < SHORT_SEGMENT_RATIO:
            findings.append(Finding(
                severity="warning", kind="target_suspiciously_short",
                detail=(f"Target is {actual:.0%} of expected length — likely dropped "
                        f"content. EN: \"{p.source_text[:100]}\""),
                source_index=p.source_indices[0], target_index=p.target_indices[0],
            ))
        elif actual > LONG_SEGMENT_RATIO:
            findings.append(Finding(
                severity="info", kind="target_suspiciously_long",
                detail=(f"Target is {actual:.0%} of expected length — possible "
                        f"repetition. EN: \"{p.source_text[:100]}\""),
                source_index=p.source_indices[0], target_index=p.target_indices[0],
            ))
    return findings


def check_untranslated(pairs: list[Pair], source_script: str) -> list[Finding]:
    """Detect target segments left in the source's script.

    LIMITATION, STATED PLAINLY: this only works for CROSS-SCRIPT pairs — English
    to Korean, Arabic, Russian, Greek. For en->es or en->fr both sides are Latin
    and this check is meaningless, so it is skipped rather than reported as a
    pass. A check that cannot fail must never look like a check that passed.
    """
    findings: list[Finding] = []
    for p in pairs:
        if not p.target_text or p.op in ("dropped",):
            continue
        if script_share(p.target_text, source_script) > MAX_SOURCE_SCRIPT_SHARE:
            findings.append(Finding(
                severity="critical", kind="possibly_untranslated",
                detail=(f"Target is mostly {source_script.title()} script — it may "
                        f"have passed through untranslated: \"{p.target_text[:100]}\""),
                source_index=p.source_indices[0] if p.source_indices else None,
                target_index=p.target_indices[0] if p.target_indices else None,
            ))
        if not p.target_text.strip():
            findings.append(Finding(
                severity="critical", kind="empty_target",
                detail="Target segment is empty.",
                source_index=p.source_indices[0] if p.source_indices else None,
            ))
    return findings


# ── Scoring ───────────────────────────────────────────────────────────────────

def delivery_rate(source: list[str], target: list[str]) -> float:
    """Target segments per source segment, as a percentage. Fully deterministic.

    This is the one number in Layer 1 that involves no heuristic whatsoever: it
    is arithmetic on two line counts. When alignment confidence is "heuristic",
    THIS is the figure to trust and quote; structural integrity below depends on
    pairings that may be guesses. Capped at 100 — producing more target segments
    than source ones is its own problem, reported separately, not extra credit.
    """
    if not source:
        return 0.0
    return min(100.0, 100.0 * len(target) / len(source))


def structural_integrity(source: list[str], pairs: list[Pair],
                         findings: list[Finding]) -> float:
    """Percentage of source segments that arrived structurally intact.

    DELIBERATELY NOT CALLED "ACCURACY". This measures delivery, not quality: a
    segment counts as intact if it produced output of a plausible size in the
    right script. A fluent mistranslation scores 100% here. Layer 2 is what will
    judge whether the meaning survived — and the final headline number should
    come from there, with this as a floor.
    """
    if not source:
        return 0.0
    damaged: set[int] = set()
    for f in findings:
        if f.severity in ("critical", "warning") and f.source_index is not None:
            damaged.add(f.source_index)
    for p in pairs:
        if p.op in ("dropped", "merge"):
            damaged.update(p.source_indices)
    return 100.0 * (len(source) - len(damaged)) / len(source)


# ── Reporting ─────────────────────────────────────────────────────────────────

def render_report(meta: dict, pairs: list[Pair], findings: list[Finding],
                  score: float) -> str:
    order = {"critical": 0, "warning": 1, "info": 2}
    ranked = sorted(findings, key=lambda f: (order.get(f.severity, 9),
                                             f.source_index or 0))
    counts = {s: sum(1 for f in findings if f.severity == s)
              for s in ("critical", "warning", "info")}

    out: list[str] = []
    out.append("# Translation integrity report\n")
    out.append(f"- **Source:** `{meta['source_file']}` ({meta['source_segments']} segments)")
    out.append(f"- **Target:** `{meta['target_file']}` ({meta['target_segments']} segments)"
               f" — language `{meta['target_lang']}`")
    out.append(f"- **Source script:** {meta['source_script'].title()}")
    out.append("")
    conf = meta["alignment_confidence"]
    out.append(f"## Delivery rate: {meta['delivery_rate']:.1f}%  ·  "
               f"Structural integrity: {score:.1f}%\n")
    out.append("> **Delivery rate** is pure arithmetic on segment counts — no heuristic,")
    out.append("> always trustworthy. **Structural integrity** additionally depends on")
    out.append("> per-segment pairings.\n")
    out.append("> Both measure **delivery, not quality** — whether each source segment")
    out.append("> produced output of a plausible size in the expected script. A fluent")
    out.append("> mistranslation still scores 100%. Treat these as a floor, not a grade.\n")

    if conf == "exact":
        out.append("> **Alignment: exact.** Source and target segment counts match, so")
        out.append("> pairings are correct by construction.\n")
    else:
        out.append("> ⚠️ **Alignment: HEURISTIC — pairings below are guesses.** The counts")
        out.append("> disagree, so segments were matched by character length, which cannot")
        out.append("> see meaning. On our own 2026-03-26 Korean log this method mis-paired")
        out.append("> four segments and pointed blame one line from the real damage. Trust")
        out.append("> the delivery rate and the count mismatch; verify every pairing by eye.\n")

    out.append(f"**{counts['critical']} critical · {counts['warning']} warning · "
               f"{counts['info']} info**\n")

    if ranked:
        out.append("## Findings\n")
        out.append("| Severity | Source line | Issue |")
        out.append("|---|---|---|")
        for f in ranked:
            loc = str(f.source_index) if f.source_index else "—"
            detail = f.detail.replace("|", "\\|").replace("\n", " ")
            out.append(f"| {f.severity} | {loc} | {detail} |")
        out.append("")
    else:
        out.append("## Findings\n\nNone. Every source segment produced plausible output.\n")

    out.append("## Aligned segments\n")
    out.append("| # | Op | Source | Target |")
    out.append("|---|---|---|---|")
    for p in pairs:
        s = (p.source_text or "—").replace("|", "\\|")
        t = (p.target_text or "—").replace("|", "\\|")
        idx = ",".join(map(str, p.source_indices)) or "—"
        out.append(f"| {idx} | {p.op} | {s} | {t} |")
    out.append("")
    return "\n".join(out)


# ── Self-test ─────────────────────────────────────────────────────────────────
#
# CALIBRATION CONTROLS. An evaluation tool that has never been shown to detect a
# known defect is not an instrument, it is an opinion generator. These fixtures
# are deliberately corrupted in specific, known ways; if the checks stop catching
# them, every number this tool produces is void and `--self-test` says so loudly.
#
# Run it before trusting a report, and always after touching the thresholds.

_GOOD_EN = [
    "Grace meets you exactly where you are.",
    "It does not wait for you to become worthy first.",
    "And then it refuses to leave you as it found you.",
    "That is the whole scandal of the gospel.",
]
_GOOD_KO = [
    "은혜는 당신이 있는 바로 그 자리에서 당신을 만납니다.",
    "당신이 먼저 합당해지기를 기다리지 않습니다.",
    "그리고 당신을 발견한 그대로 두기를 거부합니다.",
    "그것이 복음의 모든 스캔들입니다.",
]


def _run_case(src: list[str], tgt: list[str]) -> tuple[float, float, list[Finding], str]:
    conf = alignment_confidence(src, tgt)
    pairs = align(src, tgt)
    script = dominant_script(" ".join(src))
    findings = (check_structure(src, tgt, pairs)
                + check_lengths(pairs)
                + check_untranslated(pairs, script))
    return delivery_rate(src, tgt), structural_integrity(src, pairs, findings), findings, conf


def self_test() -> int:
    """Return 0 if every control behaved as expected, 1 otherwise."""
    failures: list[str] = []

    def check(name: str, condition: bool, why: str) -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {name}")
        if not condition:
            failures.append(f"{name}: {why}")

    print("Calibration controls:")

    # 1. Clean 1:1 translation must come back spotless.
    d, s, f, conf = _run_case(_GOOD_EN, _GOOD_KO)
    check("clean 1:1 scores 100% delivery", d == 100.0, f"got {d}")
    check("clean 1:1 alignment is exact", conf == "exact", f"got {conf}")
    check("clean 1:1 raises no critical findings",
          not [x for x in f if x.severity == "critical"],
          f"got {[x.kind for x in f if x.severity == 'critical']}")

    # 2. A dropped segment must be caught by arithmetic alone.
    d, s, f, conf = _run_case(_GOOD_EN, _GOOD_KO[:3])
    check("dropped segment lowers delivery to 75%", d == 75.0, f"got {d}")
    check("dropped segment raises a critical count mismatch",
          any(x.kind == "segment_count_mismatch" for x in f), "no mismatch finding")
    check("dropped segment marks alignment heuristic", conf == "heuristic", f"got {conf}")

    # 3. A segment left in the source language must be caught by script analysis.
    leaky = list(_GOOD_KO)
    leaky[2] = "And then it refuses to leave you as it found you."
    d, s, f, conf = _run_case(_GOOD_EN, leaky)
    check("untranslated segment is flagged",
          any(x.kind == "possibly_untranslated" for x in f),
          f"got {[x.kind for x in f]}")

    # 4. A truncated segment must be caught by the length norm even though the
    #    counts still match — this is the "dropped clause" fingerprint.
    truncated = list(_GOOD_KO)
    truncated[1] = "기다리지 않습니다."
    d, s, f, conf = _run_case(_GOOD_EN, truncated)
    check("truncated segment is flagged as suspiciously short",
          any(x.kind == "target_suspiciously_short" for x in f),
          f"got {[x.kind for x in f]}")
    check("truncated segment still reports 100% delivery (counts match)",
          d == 100.0, f"got {d} — delivery must not silently absorb content loss")

    print()
    if failures:
        print(f"{len(failures)} CONTROL(S) FAILED — this tool's output cannot be trusted:")
        for x in failures:
            print(f"  - {x}")
        return 1
    print("All controls passed. Detection is working.")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure structural integrity of a translation (Layer 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="Run built-in calibration controls and exit. Do this before "
                         "trusting a report, and after changing any threshold.")
    ap.add_argument("--source", type=Path,
                    help="Source text, one segment per line.")
    ap.add_argument("--target", type=Path,
                    help="Translated text, one segment per line.")
    ap.add_argument("--target-lang", default="?", help="Target language code, for the report.")
    ap.add_argument("--report", type=Path, help="Write a markdown report here.")
    ap.add_argument("--json", type=Path, help="Write machine-readable results here.")
    ap.add_argument("--fail-under", type=float, default=None,
                    help="Exit non-zero if structural integrity is below this percentage.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.source or not args.target:
        ap.error("--source and --target are required (or use --self-test)")

    source = load_segments(args.source)
    target = load_segments(args.target)
    src_script = dominant_script(" ".join(source))

    pairs = align(source, target)
    findings = (check_structure(source, target, pairs)
                + check_lengths(pairs)
                + check_untranslated(pairs, src_script))
    score = structural_integrity(source, pairs, findings)

    meta = {
        "source_file": str(args.source), "target_file": str(args.target),
        "source_segments": len(source), "target_segments": len(target),
        "target_lang": args.target_lang, "source_script": src_script,
        "alignment_confidence": alignment_confidence(source, target),
        "delivery_rate": delivery_rate(source, target),
    }

    report = render_report(meta, pairs, findings, score)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"Report written: {args.report}")
    else:
        print(report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "meta": meta, "structural_integrity": score,
            "findings": [asdict(f) for f in findings],
            "pairs": [asdict(p) for p in pairs],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON written: {args.json}")

    print(f"Delivery rate: {meta['delivery_rate']:.1f}%  ·  "
          f"Structural integrity: {score:.1f}%  "
          f"({sum(1 for f in findings if f.severity == 'critical')} critical, "
          f"alignment {meta['alignment_confidence']})")

    if args.fail_under is not None and score < args.fail_under:
        sys.exit(f"FAIL: {score:.1f}% is below the --fail-under threshold of {args.fail_under}%")


if __name__ == "__main__":
    main()
