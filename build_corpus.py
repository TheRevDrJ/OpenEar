#!/usr/bin/env python3
"""build_corpus.py — turn sermon manuscripts into a translation test corpus.

WHY THIS IS SEPARATE FROM ANYTHING TO DO WITH AUDIO
---------------------------------------------------
Standing rule (Jonathan, 2026-07-31): **transcription and translation are always
tested in isolation.** They are two different systems that happen to be used
together, and a coupled test cannot tell you which one failed. So:

    ASR quality         -> WER, measured by test_pipeline.py against a reference
    Translation quality -> adequacy, measured by score_translation.py

This tool feeds the second one only. Its input is a clean written manuscript, and
nothing here ever touches a microphone or a recognizer. If a translation scores
badly on this corpus, the translator is at fault — there is no other suspect.

WHAT IT PRODUCES
----------------
An ID-stamped segment file (NNNNN<TAB>text) matching OpenEar's own --log-text
format, so it drops straight into score_translation.py and pairs exactly.

SEGMENTATION MATCHES PRODUCTION
-------------------------------
Segments are sentences, because a sentence is the unit OpenEar actually
translates: server.py buffers speech fragments until sentence-ending punctuation,
then translates the whole sentence. Feeding the translator paragraphs, or
half-sentences, would measure something the product never does.

PRIVACY
-------
Sermon manuscripts are pastoral material — they contain real people, hospital
visits, families in the congregation. Output belongs in a gitignored directory
(reports/), never in the repo. This tool will refuse to write inside a tracked
path it recognises as public.

Usage:
    python build_corpus.py sermon.txt --out reports/corpus.txt --limit 100
    python build_corpus.py sermons/*.txt --out reports/corpus.txt --limit 200
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

# Lines that are structure rather than speech. A manuscript is full of scaffolding
# the preacher never says aloud, and translating it would score the model on text
# no congregant ever hears.
SKIP_LINE = re.compile(
    r"""^\s*(
          [IVXLC]+\s*$                     # roman numeral section markers
        | \d+\s*$                          # bare numbers / page numbers
        | [-=_*#]{3,}\s*$                  # rules and dividers
        | \[[^\]]*\]\s*$                   # [PAUSE], [SLIDE], [stage directions]
        | \([^)]*\)\s*$                    # (parenthetical asides on their own line)
        | (SLIDE|HYMN|PRAYER|SCRIPTURE|READING|BENEDICTION|OFFERTORY|NOTE|TITLE)\b.*
    )""",
    re.X | re.I,
)

# Sentence boundary: terminator, optional quote/bracket, then whitespace + capital.
# Deliberately conservative — a missed split costs one long segment, a wrong split
# hands the translator a fragment and quietly penalises the model for our error.
SENTENCE_END = re.compile(r'(?<=[.!?])["\'”’)\]]*\s+(?=[A-Z"\'“(\[])')

ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "st.", "rev.", "fr.", "prof.", "sr.", "jr.",
    "vs.", "etc.", "e.g.", "i.e.", "cf.", "no.", "vol.", "ch.", "v.", "vv.",
}


def is_heading(line: str) -> bool:
    """A short line with no terminal punctuation — a title, section head, or citation.

    WHY THIS EXISTS. On the first real sermon, segment 1 came out as:

        "Hide and Seek Matthew 13:31-33, 44-52 The story My brother and I played
         a lot of games growing up on our farm in Indiana."

    — the title, the scripture citation, a section heading, and the opening
    sentence, welded together. None of those lines end in a full stop, so the
    paragraph-rejoining step (correctly) treated them as wrapped prose and glued
    them to the sentence that followed.

    That matters beyond tidiness: it would hand BOTH models a garbled first
    segment and then score them on the mess. A test corpus must never penalise a
    system for damage the harness introduced.

    Real sentences end in punctuation. Headings, titles and citations do not, and
    they are short. Both conditions together are a reliable signal; either one
    alone is not ("He was braver than I was." is short but punctuated, and a long
    unpunctuated line is more likely a genuine sentence missing its full stop).
    """
    if not line:
        return False
    if line[-1] in ".!?:;\"'”’)":
        return False
    return len(line.split()) <= 8


def clean_text(raw: str) -> str:
    """Normalise a manuscript into flowing prose.

    Manuscripts arrive hard-wrapped at some column, often with the preacher's own
    line breaks marking breath rather than sentence. Joining them back into
    paragraphs before splitting is what keeps a sentence from being torn in half.
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if SKIP_LINE.match(line) or is_heading(stripped):
            kept.append("")            # treat as a paragraph break
            continue
        kept.append(stripped)

    # Blank line = paragraph break; single newline = a wrap, so rejoin with a space.
    paragraphs, current = [], []
    for line in kept:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def split_sentences(paragraph: str) -> list[str]:
    """Split into sentences, rejoining splits that landed after an abbreviation."""
    parts = SENTENCE_END.split(paragraph)
    out: list[str] = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if not part:
            continue
        if out:
            last_word = out[-1].split()[-1].lower() if out[-1].split() else ""
            if last_word in ABBREVIATIONS:
                out[-1] = f"{out[-1]} {part}"
                continue
        out.append(part)
    return out


def is_usable(sentence: str, min_words: int, max_words: int) -> bool:
    """Filter to segments that resemble what the translator meets in production."""
    words = sentence.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not re.search(r"[a-z]", sentence):        # ALL CAPS headings
        return False
    letters = sum(c.isalpha() for c in sentence)
    if letters < len(sentence) * 0.5:            # mostly numbers/punctuation
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="Manuscript file(s).")
    ap.add_argument("--out", required=True, type=Path, help="ID-stamped corpus file.")
    ap.add_argument("--limit", type=int, default=100, help="Max segments (default 100).")
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1517,
                    help="Sampling seed, so a corpus is reproducible.")
    ap.add_argument("--sample", choices=["spread", "first", "random"], default="spread",
                    help="spread (default) takes segments evenly across the whole "
                         "manuscript, so the corpus is not all introduction.")
    args = ap.parse_args()

    # Privacy guard: refuse to write pastoral material somewhere tracked.
    out_str = str(args.out).replace("\\", "/")
    if not any(part in out_str for part in ("reports/", "text-logs/", "/tmp/", "scratch")):
        sys.exit(f"ERROR: refusing to write a sermon corpus to '{args.out}'.\n"
                 f"       Sermon manuscripts are pastoral material and must land in a\n"
                 f"       gitignored directory — use reports/ .")

    sentences: list[str] = []
    for path in args.inputs:
        if not path.exists():
            sys.exit(f"ERROR: no such file: {path}")
        cleaned = clean_text(path.read_text(encoding="utf-8", errors="replace"))
        found = 0
        for para in cleaned.split("\n\n"):
            for s in split_sentences(para):
                if is_usable(s, args.min_words, args.max_words):
                    sentences.append(s)
                    found += 1
        print(f"  {path.name}: {found} usable segments")

    if not sentences:
        sys.exit("ERROR: no usable segments found. Check the input format.")

    print(f"\n{len(sentences)} usable segments total")

    if len(sentences) > args.limit:
        if args.sample == "first":
            chosen = sentences[:args.limit]
        elif args.sample == "random":
            import random
            chosen = random.Random(args.seed).sample(sentences, args.limit)
            chosen.sort(key=sentences.index)
        else:                                    # spread
            step = len(sentences) / args.limit
            chosen = [sentences[int(i * step)] for i in range(args.limit)]
        print(f"Sampled {len(chosen)} using '{args.sample}'")
    else:
        chosen = sentences

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(f"{i:05d}\t{s}" for i, s in enumerate(chosen, 1)) + "\n",
        encoding="utf-8")

    words = sum(len(s.split()) for s in chosen)
    print(f"\nWrote {args.out}")
    print(f"  {len(chosen)} segments, {words} words, "
          f"{words / len(chosen):.1f} words per segment")


if __name__ == "__main__":
    main()
