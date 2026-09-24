# Model evaluation registry

**Purpose: never evaluate the same model twice.**

Every translation or ASR model we test gets an entry here — what it is, how it
licenses, what it scored, how fast it ran, and the verdict. Including the ones we
rejected. A model that was measured and set aside is a *settled question*, and the
point of this file is that nobody re-opens it a year from now on a hunch.

If you are about to download a model to try it, **read this file first.**

Results come from `score_translation.py` (adequacy) and `test_pipeline.py` (WER).
Method notes are at the bottom — read them before comparing any number here to a
figure published elsewhere.

---

## In production

| Stage | Model | License | Since |
|---|---|---|---|
| ASR | **NVIDIA Parakeet-TDT-0.6B-v2** | CC-BY-4.0 | 2026-03 |
| Translation | **NLLB-200 3.3B** (CTranslate2 int8) | CC-BY-NC-4.0 ¹ | 2026-03 |

¹ Non-commercial, and that is **fine here**: OpenEar does not ship the weights. Setup
*downloads* them, so AGPL code and NC weights meet at runtime on the user's own disk
as two separately-licensed works. **The operative rule is: never bundle the weights.**
Settled — see `FOB/LICENSING_REFERENCE.md` in Felix's briefcase; do not re-litigate.

---

## Translation models evaluated

### NLLB-200 3.3B — **IN PRODUCTION**

| | |
|---|---|
| Repo | `entai2965/nllb-200-3.3B-ctranslate2` |
| Size | ~5.5 GB (int8) |
| License | CC-BY-NC-4.0 |
| Adequacy — Spanish | **87.3%** |
| Adequacy — Korean | **72.9%** |
| Speed | 0.33 s/seg (es), 0.28 s/seg (ko) |
| Evaluated | 2026-07-31 |

**Verdict: keep.** Level with MADLAD in Spanish, ahead in Korean, and roughly twice
as fast.

**Known weaknesses:**
- **No em-dash in its vocabulary.** U+2014 becomes the unknown token and decodes as a
  literal `⁇`. Harmless in production — Parakeet emits only `.` `,` and `'` — but it
  will bite anything that feeds NLLB text of written origin.
- Renders "justifying grace" acceptably in Spanish (*gracia justificadora*) but garbles
  it in Korean.
- Drops clauses on long, ASR-damaged sentences.

---

### MADLAD-400 3B — **evaluated, rejected 2026-07-31**

| | |
|---|---|
| Repo | `Nextcloud-AI/madlad400-3b-mt-ct2-int8` |
| Size | 2.77 GB (int8) — half of NLLB |
| License | **Apache-2.0** (permissive — the main attraction) |
| Adequacy — Spanish | 87.4% |
| Adequacy — Korean | 70.7% |
| Speed | 0.49 s/seg (es), **0.73 s/seg (ko)** |
| Evaluated | 2026-07-31 |

**Verdict: no switch.** A statistical tie in Spanish (+0.1), behind in Korean (−2.2),
and **2.6× slower on Korean** — which matters for a live captioning product.

**Why it looked promising:** Apache-2.0 rather than CC-BY-NC, one model covering all
four enabled languages, and a smaller download. Recommended by Felix on those grounds.

**Why it lost:**
- Systematically mishandles compound theological terms — renders "justifying grace" as
  a verb phrase ("to justify the grace") in **both** Spanish and Korean.
- Drops into casual Korean speech level (반말) in a preaching context.
- The speed penalty is worst in the language that already performs worst.

**Note for anyone re-reading this:** an 8-segment pilot had MADLAD *ahead* in Korean
(79.1 vs 77.5). At 192 segments that **reversed**. Do not act on a small-corpus result.

**Prompting, if ever re-tested:** T5-style. The target-language token goes on the front
of the SOURCE (`<2es> text`), and there is **no** `target_prefix` — the opposite of
NLLB. Getting that backwards yields fluent output in the wrong language.

---

## Considered without testing

| Model | Why not tested |
|---|---|
| Whisper large-v3 / turbo (ASR) | Tested 2026-03 — see `BENCHMARK_REPORT.md`. Lost to Parakeet on punctuation consistency, which directly degrades translation quality downstream. |
| Apple Silicon / CoreML stack | Not a model question. Killed 2026-07-31 on **cost**, not capability — the $599 Mac mini that justified it was retired; 16 GB now ~$1300. See `STATE.md`. |

---

## Method — read before comparing these numbers to anything published

- **Adequacy, not BLEU/chrF.** String metrics punish correct paraphrase and reward
  imitation of one reference translation. These numbers come from per-segment
  judgment of whether *meaning survived*, which is the only question a congregation
  cares about.
- **Blind, interleaved, single sitting.** Both systems' output for the same source
  goes into one shuffled, unlabelled worksheet. Scoring A on Monday and B on Tuesday
  produces two figures that look comparable and are not — judge calibration drifts,
  and that drift then gets attributed to the models.
- **Every judge is validated.** Each worksheet chunk carries hidden negative controls
  (a source paired with a *different* segment's translation; a source paired with 40%
  of its own). A judge that scores those highly is rubber-stamping or blind to
  omission, and the chunk is **dropped and named**, not averaged in.
- **Corpus: a real sermon, whole.** 192 segments, 2,867 words, transcript-form (spoken
  register, no manuscript scaffolding). Transcription and translation are **always**
  tested in isolation — a coupled test cannot attribute a failure.
- **The 2026-07-31 run:** 24 judges, 960 judgments, **0 chunks rejected**.

## The finding that outranks the model choice

**Korean congregants receive materially worse translation than Spanish congregants —
72.9% vs 87.3%,** same source, same models, same judges.

The *kind* of failure differs too. Spanish fails on vocabulary (wrong word, true
sentence). Korean fails on **meaning** — the panel independently caught several clean
inversions where the translation asserts the opposite of the source. A wrong word is a
stumble; a reversed claim is a different sermon.

Two systematic Korean problems, neither solved by changing models:
1. **Speech level.** Roughly a third of otherwise-good output drops into 반말 — casual
   address, wrong for preaching. God frequently rendered 신 (generic deity) not 하나님.
2. **Key nouns transliterated** rather than translated, severing them from the imagery
   a sermon is built on.

Both are register problems, and register is where the next real work is.

**Scripture quotation scored near-perfectly in both languages** — that text appears in
training data at enormous frequency and is effectively recalled rather than translated.
It is the preacher's *own prose* that degrades.
