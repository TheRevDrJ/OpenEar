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

_ID_LINE = __import__("re").compile(r"^(\d+)\t(.*)$")


def load_segments(path: Path) -> tuple[list[str], list[str | None]]:
    """Read one segment per line, returning (texts, ids).

    Two accepted formats:

      "00007\tSanctifying grace is the journey..."   <- ID-stamped (preferred)
      "Sanctifying grace is the journey..."          <- legacy, no ID

    OpenEar stamps a shared segment ID onto each completed sentence and every
    translation of it (see server.py broadcast()). When both files carry IDs the
    pairing is EXACT and no alignment guessing happens at all. Legacy logs written
    before that change have no IDs and fall back to the length heuristic, which is
    why the heuristic still exists rather than being deleted.
    """
    if not path.exists():
        sys.exit(f"ERROR: no such file: {path}")
    texts: list[str] = []
    ids: list[str | None] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _ID_LINE.match(line)
        if m:
            ids.append(m.group(1))
            texts.append(m.group(2).strip())
        else:
            ids.append(None)
            texts.append(line.strip())
    return texts, ids


def rebuffer(fragments: list[str]) -> list[str]:
    """Reconstruct the sentences the translator actually saw, from raw fragments.

    Legacy logs (before segment IDs) recorded one line per incoming ASR FRAGMENT
    on the source side, while the target side recorded one line per completed
    SENTENCE. Comparing them directly is meaningless — different units.

    This replays server.py's exact buffering rule: accumulate fragments until the
    buffer ends in sentence-terminating punctuation, then emit. Applied to an old
    source log it reproduces the units NLLB was actually given, which makes
    historical data comparable instead of unusable.

    MUST STAY IN LOCKSTEP with `_sentence_end_re` and the accumulation in
    server.py broadcast(). If that buffering rule changes, change it here too, or
    every score computed from a legacy log silently drifts.
    """
    import re as _re
    sentence_end = _re.compile(r"[.!?][\s]*$")
    out: list[str] = []
    buf = ""
    for frag in fragments:
        buf += (" " if buf else "") + frag
        if sentence_end.search(buf):
            out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


def align_by_id(source: list[str], src_ids: list[str | None],
                target: list[str], tgt_ids: list[str | None]) -> list[Pair]:
    """Pair segments by their shared ID. Exact by construction, no heuristics.

    A source ID with no matching target ID is a genuine drop — the sentence was
    transcribed and never translated. Unlike the length-based path, that is a
    fact rather than an inference, so the report can state it plainly.
    """
    tgt_by_id = {tid: (i, txt) for i, (tid, txt) in enumerate(zip(tgt_ids, target), 1)
                 if tid is not None}
    used: set[str] = set()
    pairs: list[Pair] = []

    for s_pos, (sid, s_txt) in enumerate(zip(src_ids, source), 1):
        if sid is not None and sid in tgt_by_id:
            t_pos, t_txt = tgt_by_id[sid]
            used.add(sid)
            pairs.append(Pair(op="match", source_indices=[s_pos],
                              target_indices=[t_pos],
                              source_text=s_txt, target_text=t_txt))
        else:
            pairs.append(Pair(op="dropped", source_indices=[s_pos],
                              target_indices=[], source_text=s_txt, target_text=""))

    for t_pos, (tid, t_txt) in enumerate(zip(tgt_ids, target), 1):
        if tid is not None and tid not in used:
            pairs.append(Pair(op="added", source_indices=[], target_indices=[t_pos],
                              source_text="", target_text=t_txt))
    return pairs


def has_ids(ids: list[str | None]) -> bool:
    return bool(ids) and all(i is not None for i in ids)


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

    # SAME-SCRIPT PAIRS: SKIP ENTIRELY.
    #
    # This guard was described in the docstring above from the first version and
    # was NOT implemented — a comment promising behaviour the code did not have.
    # It went unnoticed because every test until now was English->Korean, where
    # the scripts differ. The first English->Spanish run flagged all 8 segments as
    # "possibly untranslated" (Latin text in a Latin-script target, exactly as
    # designed) and drove structural integrity to 0.0% on a translation that
    # scored 95% adequate.
    #
    # A check that cannot fail must never masquerade as a check that passed, and a
    # check that cannot succeed must never masquerade as a failure.
    target_script = dominant_script(" ".join(p.target_text for p in pairs))
    if target_script == source_script or target_script == "NONE":
        return findings

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
        method = meta.get("alignment_method", "")
        if method == "segment-id":
            out.append("> **Alignment: exact (segment IDs).** Both logs carry the shared")
            out.append("> ID OpenEar stamps on each completed sentence, so pairings are")
            out.append("> correct by construction and a missing translation is a *fact*,")
            out.append("> not an inference.\n")
        else:
            out.append("> **Alignment: exact.** Source and target segment counts match, so")
            out.append("> pairings are correct by construction.\n")
    else:
        out.append("> ⚠️ **Alignment: HEURISTIC — pairings below are guesses.** The counts")
        out.append("> disagree, so segments were matched by character length, which cannot")
        out.append("> see meaning. On our own 2026-03-26 Korean log this method mis-paired")
        out.append("> four segments and pointed blame one line from the real damage. Trust")
        out.append("> the delivery rate and the count mismatch; verify every pairing by eye.\n")

    adq = meta.get("adequacy")
    if adq:
        out.append("## Adequacy (Layer 2 — did the meaning survive?)\n")
        if adq["valid"] and adq["adequacy"] is not None:
            out.append(f"### {adq['adequacy']:.1f}% adequate  "
                       f"({adq['segments_judged']} segments judged)\n")
            if conf != "exact":
                # Layer 2 inherits Layer 1's uncertainty and this must be said out
                # loud. The judge scores the PAIRS it is handed. If the aligner
                # handed it a mis-paired source and target, the judge correctly
                # scores that pair near zero -- and the resulting figure measures
                # ALIGNMENT failure, not TRANSLATION failure. The two are
                # indistinguishable in the aggregate, so the number understates the
                # translator by an unknown amount. Do not quote it as a translation
                # accuracy; quote it as a pipeline figure, or fix the alignment.
                out.append("> ⚠️ **This figure is depressed by alignment error and is NOT a")
                out.append("> translator accuracy.** Segment counts disagreed, so pairings")
                out.append("> were guessed by length. A mis-paired segment is correctly")
                out.append("> scored near zero by the judge — but that penalises the")
                out.append("> *aligner*, not the translation. Segments marked `wrong` below")
                out.append("> are the ones to inspect first: check whether the candidate is")
                out.append("> a bad translation or simply a good translation of a")
                out.append("> *different* source line.\n")
            if adq["categories"]:
                out.append("| Category | Segments |")
                out.append("|---|---|")
                for c, n in sorted(adq["categories"].items(), key=lambda kv: -kv[1]):
                    out.append(f"| {c} | {n} |")
                out.append("")
        else:
            out.append("### VOID — the judge failed its calibration controls.\n")
            out.append("> No adequacy figure is reported. A number from an instrument that")
            out.append("> cannot tell right from wrong is worse than no number at all.\n")

        ctrl = adq["controls"]
        out.append("**Judge calibration**\n")
        out.append("| Control | Scores | Must be |")
        out.append("|---|---|---|")
        out.append(f"| Mismatched pairs | {ctrl['mismatch']} | ≤ 30 |")
        out.append(f"| Truncated candidates | {ctrl['truncated']} | well below real mean |")
        rm = ctrl["real_mean"]
        out.append(f"| Real segments (mean) | {rm:.0f} | — |" if rm is not None
                   else "| Real segments (mean) | — | — |")
        out.append("")

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


# ── Layer 2: blind adequacy judging ───────────────────────────────────────────
#
# Layer 1 asks "did output arrive?". Layer 2 asks "did the MEANING arrive?" — the
# question that actually decides whether a congregant understood the sermon.
#
# WHY A WORKSHEET INSTEAD OF CALLING A JUDGE DIRECTLY
# ---------------------------------------------------
# The obvious design is to have the tool call a model and return a score. We are
# not doing that, for three reasons:
#
#   1. AUDITABILITY. A number with no visible evidence is exactly what we are
#      replacing. Here the questions and the answers are files on disk: anyone can
#      read precisely what the judge was shown and precisely what it said.
#   2. THE JUDGE SHOULD NOT BE BAKED IN. Today the best available judge is a
#      capable LLM; tomorrow it may be a local model, or an actual Korean speaker.
#      Same protocol, comparable numbers, no rewrite.
#   3. CONFLICT OF INTEREST. Whoever built the translation pipeline should not be
#      the unblinded judge of its output. The worksheet strips system identity,
#      shuffles order, and salts in controls, so a judge cannot favour a system
#      even if it wanted to — including me favouring something I built.
#
# HOW THE CONTROLS WORK, AND WHY THEY NEED NO GOLD TRANSLATION
# -------------------------------------------------------------
# We cannot manufacture a known-CORRECT translation for an arbitrary language
# without a reference. But we can manufacture known-WRONG ones from the data
# itself, and negative controls are enough to prove a judge discriminates:
#
#   * MISMATCH  — a source segment paired with a DIFFERENT segment's translation.
#                 Must score near zero. If it doesn't, the judge is rubber-stamping.
#   * TRUNCATED — a source segment paired with the first ~40% of its own
#                 translation. Must score clearly below the real segments. If it
#                 doesn't, the judge is blind to omission, which is the exact
#                 failure mode we care most about.
#
# A judge that passes these has demonstrated it can tell wrong from right on THIS
# data. A judge that fails them voids the run — the report says so and reports no
# adequacy figure at all, because a number from a broken instrument is worse than
# no number.

RUBRIC = """\
You are scoring TRANSLATION ADEQUACY for a live church captioning system. A preacher
speaks English; congregants read the translation on their phones in real time.

For each item you are given a SOURCE segment and a CANDIDATE translation. Score how
much of the source's MEANING survived.

Score 0-100:
  100  Complete. Every idea in the source is present and correctly rendered.
   80  Minor loss. A nuance, connotation, or modifier is off, but a listener would
       take away the same point.
   50  Major loss. A clause, qualifier, or key idea is missing or altered. The
       listener gets a different or thinner point.
   20  Wrong. The candidate asserts something the source did not, or omits the
       central idea.
    0  Unusable. Untranslated, empty, or unrelated to the source.

Also assign one CATEGORY: complete | minor_loss | major_loss | wrong | unusable

RULES:
  * Judge MEANING, not wording. A completely different phrasing that carries the
    same meaning scores 100. There are many valid ways to say a thing.
  * Do NOT reward fluency. Beautiful text that dropped a clause is major_loss.
  * Omission is the failure that matters most here. If content in the source is
    absent from the candidate, say so explicitly in your note.
  * Register counts. A word that is technically adequate but carries the wrong
    connotation for a sermon (e.g. rendering "he is using us" with a verb closer
    to "exploiting us") is at best minor_loss, and say why.
  * The source may contain speech-recognition errors — duplicated words, broken
    sentence boundaries. Judge the translation against what the source SAYS, not
    against what you think the preacher meant. Note it if the source is damaged.
  * You do not know which system produced any candidate, and items are in random
    order. Do not try to infer it.

Fill in "score", "category", and a one-sentence "note" for EVERY item. Leave
everything else untouched.
"""


def build_comparison_worksheet(source: list[str], systems: dict[str, list[str]],
                               seed: int) -> tuple[dict, dict]:
    """Blind head-to-head worksheet: several systems' output for the same source.

    WHY HEAD-TO-HEAD IN ONE WORKSHEET, rather than scoring each system separately
    and comparing the two numbers:

    Scoring system A on Monday and system B on Tuesday produces two figures that
    LOOK comparable and are not. The judge's calibration drifts between sittings —
    different fatigue, different anchoring, a different sense of what "80" means.
    You then attribute that drift to the models.

    Interleaving both systems in one shuffled, unlabelled worksheet removes the
    problem by construction. Every judgment is made in the same sitting against
    the same internal yardstick, and the judge cannot know which system it is
    grading, so it cannot favour one. The difference between the resulting scores
    is then a difference between the systems, which is the only thing we wanted to
    measure.

    The same negative controls are salted in, and they now do double duty: they
    validate the judge AND they establish the floor against which both systems are
    read.
    """
    import random
    rng = random.Random(seed)

    items: list[dict] = []
    key: dict[str, dict] = {}

    def add(kind: str, system: str | None, src: str, cand: str, idx: int | None) -> None:
        item_id = f"item_{len(items) + 1:03d}"
        items.append({"id": item_id, "source": src, "candidate": cand,
                      "score": None, "category": None, "note": None})
        key[item_id] = {"kind": kind, "system": system, "source_index": idx}

    for name, outputs in systems.items():
        for i, (src, cand) in enumerate(zip(source, outputs), 1):
            if src and cand:
                add("real", name, src, cand, i)

    # Controls are drawn from the FIRST system's output, but they test the judge,
    # not the system — so they are attributed to no system and excluded from every
    # per-system average.
    first = next(iter(systems.values()))
    n = max(2, len(source) // 4)
    if len(source) >= 2:
        for _ in range(n):
            a, b = rng.sample(range(min(len(source), len(first))), 2)
            add("control_mismatch", None, source[a], first[b], None)
    for _ in range(n):
        i = rng.randrange(min(len(source), len(first)))
        cut = max(1, int(len(first[i]) * 0.4))
        add("control_truncated", None, source[i], first[i][:cut], None)

    rng.shuffle(items)
    return ({"_instructions": RUBRIC,
             "_note": ("Items from several systems are interleaved, plus deliberately "
                       "corrupted controls. You are not told which is which. Score each "
                       "item on its own merits."),
             "items": items},
            {"seed": seed, "key": key})


def ingest_comparison(worksheet: dict, keyfile: dict) -> dict:
    """Per-system adequacy from a head-to-head worksheet, with shared controls."""
    key = keyfile["key"]
    scored = {i["id"]: i for i in worksheet.get("items", []) if i.get("score") is not None}

    by_system: dict[str, list[float]] = {}
    mismatch, truncated = [], []
    for iid, item in scored.items():
        meta = key.get(iid, {})
        s = float(item["score"])
        if meta.get("kind") == "real":
            by_system.setdefault(meta["system"], []).append(s)
        elif meta.get("kind") == "control_mismatch":
            mismatch.append(s)
        elif meta.get("kind") == "control_truncated":
            truncated.append(s)

    all_real = [s for v in by_system.values() for s in v]
    valid = True
    reasons = []
    if len(scored) != len(key):
        valid = False
        reasons.append(f"{len(key) - len(scored)} item(s) unscored")
    if mismatch and max(mismatch) > 30:
        valid = False
        reasons.append(f"a mismatched control scored {max(mismatch):.0f} (must be <= 30)")
    if truncated and all_real:
        at, ar = sum(truncated) / len(truncated), sum(all_real) / len(all_real)
        if at >= ar - 15:
            valid = False
            reasons.append(f"truncated controls averaged {at:.0f} vs {ar:.0f} for real text")

    return {
        "valid": valid, "reasons": reasons,
        "systems": {k: {"adequacy": sum(v) / len(v), "segments": len(v)}
                    for k, v in sorted(by_system.items())},
        "controls": {"mismatch": mismatch, "truncated": truncated},
    }


def build_worksheet(pairs: list[Pair], seed: int) -> tuple[dict, dict]:
    """Produce a blind judging worksheet and its private answer key.

    The worksheet is what the judge sees; the key is what the harness uses to
    unshuffle, separate controls from real data, and score. They MUST stay in
    separate files — a judge who can see the key is not blind.
    """
    import random
    rng = random.Random(seed)

    real = [p for p in pairs if p.source_text and p.target_text
            and p.op not in ("dropped", "added")]

    items: list[dict] = []
    key: dict[str, dict] = {}

    def add(kind: str, source: str, candidate: str, src_idx: int | None) -> None:
        item_id = f"item_{len(items) + 1:03d}"
        items.append({"id": item_id, "source": source, "candidate": candidate,
                      "score": None, "category": None, "note": None})
        key[item_id] = {"kind": kind, "source_index": src_idx}

    for p in real:
        add("real", p.source_text, p.target_text,
            p.source_indices[0] if p.source_indices else None)

    # Negative controls: roughly one per four real items, minimum two of each,
    # so a run always carries enough signal to catch a rubber-stamping judge.
    n_controls = max(2, len(real) // 4)

    if len(real) >= 2:
        for _ in range(n_controls):
            a, b = rng.sample(range(len(real)), 2)
            add("control_mismatch", real[a].source_text, real[b].target_text, None)

    for _ in range(n_controls):
        p = real[rng.randrange(len(real))] if real else None
        if not p:
            break
        cut = max(1, int(len(p.target_text) * 0.4))
        add("control_truncated", p.source_text, p.target_text[:cut], None)

    rng.shuffle(items)

    worksheet = {
        "_instructions": RUBRIC,
        "_note": ("Some items are deliberately corrupted quality controls. You are not "
                  "told which. Score every item on its own merits."),
        "items": items,
    }
    return worksheet, {"seed": seed, "key": key}


def ingest_worksheet(worksheet: dict, keyfile: dict) -> tuple[dict, list[Finding]]:
    """Score a filled worksheet, validating the judge before trusting it."""
    key = keyfile["key"]
    findings: list[Finding] = []

    scored = {i["id"]: i for i in worksheet.get("items", [])
              if i.get("score") is not None}
    missing = [i["id"] for i in worksheet.get("items", []) if i.get("score") is None]

    def scores_for(kind: str) -> list[float]:
        return [float(scored[i]["score"]) for i in scored
                if key.get(i, {}).get("kind") == kind]

    real_scores = scores_for("real")
    mismatch = scores_for("control_mismatch")
    truncated = scores_for("control_truncated")

    valid = True

    if missing:
        valid = False
        findings.append(Finding(
            severity="critical", kind="worksheet_incomplete",
            detail=f"{len(missing)} item(s) were not scored: {', '.join(missing[:5])}"
                   f"{'...' if len(missing) > 5 else ''}. Adequacy cannot be computed."))

    # A mismatched pair is unrelated text. Anything above 30 means the judge is
    # not actually reading the pairing.
    if mismatch:
        worst = max(mismatch)
        if worst > 30:
            valid = False
            findings.append(Finding(
                severity="critical", kind="control_failed_mismatch",
                detail=(f"A mismatched control (source paired with an unrelated "
                        f"translation) scored {worst:.0f}. The judge is not "
                        f"discriminating; this run's adequacy figure is void.")))

    # Truncated candidates are missing ~60% of their content. They must land
    # clearly below the real segments, or the judge cannot see omission.
    if truncated and real_scores:
        avg_trunc = sum(truncated) / len(truncated)
        avg_real = sum(real_scores) / len(real_scores)
        if avg_trunc >= avg_real - 15:
            valid = False
            findings.append(Finding(
                severity="critical", kind="control_failed_truncation",
                detail=(f"Truncated controls averaged {avg_trunc:.0f} against "
                        f"{avg_real:.0f} for real segments. The judge is not "
                        f"detecting omission — the failure this tool exists to "
                        f"catch. Adequacy figure is void.")))

    for item_id, item in scored.items():
        if key.get(item_id, {}).get("kind") != "real":
            continue
        if float(item["score"]) < 60:
            findings.append(Finding(
                severity="warning", kind="low_adequacy_segment",
                detail=(f"Scored {float(item['score']):.0f} "
                        f"({item.get('category', '?')}): {item.get('note', '')}"),
                source_index=key[item_id].get("source_index")))

    result = {
        "valid": valid,
        "adequacy": (sum(real_scores) / len(real_scores)) if (real_scores and valid) else None,
        "segments_judged": len(real_scores),
        "controls": {
            "mismatch": mismatch,
            "truncated": truncated,
            "real_mean": (sum(real_scores) / len(real_scores)) if real_scores else None,
        },
        "categories": {},
    }
    for item_id, item in scored.items():
        if key.get(item_id, {}).get("kind") == "real" and item.get("category"):
            c = item["category"]
            result["categories"][c] = result["categories"].get(c, 0) + 1
    return result, findings


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

    # 4a. Same-script pairs (en->es) must NOT trip the untranslated check. This
    #     regression drove structural integrity to 0% on a 95%-adequate Spanish
    #     translation, because the guard was documented but never written.
    _GOOD_ES = [
        "La gracia te encuentra exactamente donde estás.",
        "No espera a que te vuelvas digno primero.",
        "Y luego se niega a dejarte como te encontró.",
        "Ese es todo el escándalo del evangelio.",
    ]
    d, s, f, conf = _run_case(_GOOD_EN, _GOOD_ES)
    check("same-script pair (en->es) is not flagged as untranslated",
          not any(x.kind == "possibly_untranslated" for x in f),
          f"got {[x.kind for x in f]}")
    check("same-script pair scores full structural integrity", s == 100.0, f"got {s}")
    check("cross-script check still fires when scripts differ",
          any(x.kind == "possibly_untranslated"
              for x in _run_case(_GOOD_EN, [_GOOD_KO[0], _GOOD_KO[1],
                                            "This line was never translated.",
                                            _GOOD_KO[3]])[2]),
          "cross-script detection broke while fixing the same-script case")

    # 4b. ID-based pairing must be exact, and must report a missing translation
    #     as a fact rather than inferring one from lengths.
    ids = ["1", "2", "3", "4"]
    p = align_by_id(_GOOD_EN, ids, _GOOD_KO, ids)
    check("segment IDs pair every segment exactly",
          all(x.op == "match" for x in p) and len(p) == 4,
          f"got {[x.op for x in p]}")
    check("ID pairing survives reordered target lines",
          all(x.op == "match" for x in align_by_id(
              _GOOD_EN, ids, list(reversed(_GOOD_KO)), list(reversed(ids)))),
          "reordering the target file broke ID pairing")
    p = align_by_id(_GOOD_EN, ids, _GOOD_KO[:2], ids[:2])
    check("a source ID with no translation is reported dropped",
          sum(1 for x in p if x.op == "dropped") == 2,
          f"got {[x.op for x in p]}")
    check("has_ids rejects a partially-stamped file",
          not has_ids(["1", None, "3"]) and has_ids(["1", "2"]),
          "has_ids is wrong")

    # 5-7. The judge-validation must itself be validated. A worksheet scored by a
    #      rubber-stamping judge has to come back VOID, or Layer 2 is decoration.
    pairs = align(_GOOD_EN, _GOOD_KO)
    worksheet, keyfile = build_worksheet(pairs, seed=1)

    def score_all(fn) -> dict:
        ws = json.loads(json.dumps(worksheet))     # deep copy
        for item in ws["items"]:
            kind = keyfile["key"][item["id"]]["kind"]
            item["score"] = fn(kind)
            item["category"] = "complete"
            item["note"] = "test"
        return ws

    rubber = score_all(lambda kind: 100)
    res, _ = ingest_worksheet(rubber, keyfile)
    check("rubber-stamp judge (all 100s) is rejected", not res["valid"],
          "a judge scoring unrelated pairs 100 was accepted")

    blind_to_omission = score_all(lambda kind: 20 if kind == "control_mismatch" else 95)
    res, _ = ingest_worksheet(blind_to_omission, keyfile)
    check("judge blind to truncation is rejected", not res["valid"],
          "a judge scoring 60%-truncated text as high as complete text was accepted")

    good = score_all(lambda kind: {"control_mismatch": 5,
                                   "control_truncated": 40}.get(kind, 95))
    res, _ = ingest_worksheet(good, keyfile)
    check("discriminating judge is accepted", res["valid"] and res["adequacy"] is not None,
          f"a correctly-discriminating judge was rejected: {res}")
    check("accepted judge reports an adequacy figure",
          res["adequacy"] is not None and abs(res["adequacy"] - 95.0) < 0.01,
          f"got {res['adequacy']}")

    incomplete = score_all(lambda kind: 95)
    incomplete["items"][0]["score"] = None
    res, _ = ingest_worksheet(incomplete, keyfile)
    check("unfinished worksheet is rejected", not res["valid"],
          "a worksheet with an unscored item was accepted")

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
    ap.add_argument("--worksheet-out", type=Path,
                    help="LAYER 2 step 1: write a blind judging worksheet here. Also "
                         "writes <path>.key.json — keep that away from the judge.")
    ap.add_argument("--worksheet-in", type=Path,
                    help="LAYER 2 step 3: read a filled worksheet and score adequacy. "
                         "Expects <path>.key.json beside it.")
    ap.add_argument("--compare", action="append", metavar="NAME=FILE",
                    help="Head-to-head mode: add one system's output. Repeat per system, "
                         "e.g. --compare nllb=a.txt --compare madlad=b.txt. Combine with "
                         "--worksheet-out to build a blind worksheet, then --worksheet-in "
                         "to score every system in a single sitting.")
    ap.add_argument("--rebuffer-source", action="store_true",
                    help="Legacy logs only: re-apply server.py's sentence buffering to a "
                         "fragment-level source log, so it matches the sentence-level "
                         "target log. Not needed for ID-stamped logs.")
    ap.add_argument("--seed", type=int, default=1517,
                    help="Shuffle seed, so a worksheet is reproducible.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    # ── Head-to-head comparison mode ──────────────────────────────────────────
    if args.compare:
        if not args.source:
            ap.error("--compare requires --source")
        src, src_ids = load_segments(args.source)
        if args.rebuffer_source:
            src = rebuffer(src)

        systems: dict[str, list[str]] = {}
        for spec in args.compare:
            if "=" not in spec:
                ap.error(f"--compare expects NAME=FILE, got: {spec}")
            name, path = spec.split("=", 1)
            texts, _ = load_segments(Path(path))
            if len(texts) != len(src):
                ap.error(f"{name}: {len(texts)} segments but source has {len(src)}. "
                         f"Head-to-head requires every system to translate the same units.")
            systems[name] = texts

        if args.worksheet_out:
            ws, kf = build_comparison_worksheet(src, systems, args.seed)
            args.worksheet_out.parent.mkdir(parents=True, exist_ok=True)
            args.worksheet_out.write_text(json.dumps(ws, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
            kp = args.worksheet_out.with_suffix(args.worksheet_out.suffix + ".key.json")
            kp.write_text(json.dumps(kf, ensure_ascii=False, indent=2), encoding="utf-8")
            n_ctrl = sum(1 for v in kf["key"].values() if v["kind"] != "real")
            print(f"Head-to-head worksheet: {args.worksheet_out}")
            print(f"  {len(systems)} systems x {len(src)} segments + {n_ctrl} controls, "
                  f"shuffled and unlabelled (seed {args.seed})")
            print(f"  Answer key: {kp}  — keep away from the judge.")
            return

        if args.worksheet_in:
            kp = args.worksheet_in.with_suffix(args.worksheet_in.suffix + ".key.json")
            ws = json.loads(args.worksheet_in.read_text(encoding="utf-8"))
            kf = json.loads(kp.read_text(encoding="utf-8"))
            res = ingest_comparison(ws, kf)
            print("\nHead-to-head adequacy\n")
            if not res["valid"]:
                print("  VOID — the judge failed calibration:")
                for r in res["reasons"]:
                    print(f"    - {r}")
                print("  No figures reported.")
                sys.exit(1)
            width = max(len(k) for k in res["systems"])
            for name, v in sorted(res["systems"].items(),
                                  key=lambda kv: -kv[1]["adequacy"]):
                print(f"  {name.ljust(width)}  {v['adequacy']:6.1f}%   "
                      f"({v['segments']} segments)")
            print(f"\n  Controls — mismatch {res['controls']['mismatch']} (must be <=30), "
                  f"truncated {res['controls']['truncated']}")
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
                print(f"  JSON written: {args.json}")
            return

        ap.error("--compare needs --worksheet-out or --worksheet-in")

    if not args.source or not args.target:
        ap.error("--source and --target are required (or use --self-test)")

    source, src_ids = load_segments(args.source)
    target, tgt_ids = load_segments(args.target)

    if args.rebuffer_source:
        before = len(source)
        source = rebuffer(source)
        src_ids = [None] * len(source)
        print(f"Re-buffered source: {before} fragments -> {len(source)} sentences "
              f"(the units the translator actually received)")
    src_script = dominant_script(" ".join(source))

    # Exact pairing when both logs carry segment IDs; heuristic only for legacy.
    id_based = has_ids(src_ids) and has_ids(tgt_ids)
    pairs = (align_by_id(source, src_ids, target, tgt_ids) if id_based
             else align(source, target))
    findings = (check_structure(source, target, pairs)
                + check_lengths(pairs)
                + check_untranslated(pairs, src_script))
    score = structural_integrity(source, pairs, findings)

    meta = {
        "source_file": str(args.source), "target_file": str(args.target),
        "source_segments": len(source), "target_segments": len(target),
        "target_lang": args.target_lang, "source_script": src_script,
        "alignment_confidence": "exact" if id_based else alignment_confidence(source, target),
        "alignment_method": "segment-id" if id_based else "character-length heuristic",
        "delivery_rate": delivery_rate(source, target),
    }

    # ── Layer 2 step 1: emit a blind worksheet and stop ───────────────────────
    if args.worksheet_out:
        worksheet, keyfile = build_worksheet(pairs, args.seed)
        args.worksheet_out.parent.mkdir(parents=True, exist_ok=True)
        args.worksheet_out.write_text(
            json.dumps(worksheet, ensure_ascii=False, indent=2), encoding="utf-8")
        key_path = args.worksheet_out.with_suffix(args.worksheet_out.suffix + ".key.json")
        key_path.write_text(
            json.dumps(keyfile, ensure_ascii=False, indent=2), encoding="utf-8")
        n_real = sum(1 for v in keyfile["key"].values() if v["kind"] == "real")
        n_ctrl = len(keyfile["key"]) - n_real
        print(f"Worksheet written: {args.worksheet_out}")
        print(f"  {n_real} real segments + {n_ctrl} hidden controls, shuffled (seed {args.seed})")
        print(f"Answer key:        {key_path}")
        print("  DO NOT show the key to the judge. Have the judge fill in score/category/note")
        print(f"  for every item, then: --worksheet-in {args.worksheet_out}")
        return

    # ── Layer 2 step 3: ingest a filled worksheet ─────────────────────────────
    adequacy = None
    if args.worksheet_in:
        key_path = args.worksheet_in.with_suffix(args.worksheet_in.suffix + ".key.json")
        if not key_path.exists():
            sys.exit(f"ERROR: answer key not found beside the worksheet: {key_path}")
        worksheet = json.loads(args.worksheet_in.read_text(encoding="utf-8"))
        keyfile = json.loads(key_path.read_text(encoding="utf-8"))
        adequacy, judge_findings = ingest_worksheet(worksheet, keyfile)
        findings += judge_findings
        meta["adequacy"] = adequacy

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
