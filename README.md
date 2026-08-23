# LunBawang Translate

**Live demo:** https://translate.lunbawang.com/

A bidirectional **Lun Bawang ↔ English** machine translator, built by fine-tuning language models on the first parallel corpus ever assembled for the language. The web interface is deployed on Render and runs against a fine-tuned model served via the [Tinker](https://thinkingmachines.ai/tinker/) API. The production default is the **v2 dictionary retrain — Inkling-Small · Step 16,000**, warm-started from the earlier Inkling checkpoint-11,500 and continued on the OCR'd Kemaloh Lundayeh–English dictionary (22,038 pairs). See [Evaluation](#evaluation) for how it was selected and [Past experiments and versions](#past-experiments-and-versions) for the earlier Qwen3-8B and Inkling runs.

---

## Background

Lun Bawang (also called Lun Dayeh or Lundayeh) is an Austronesian language spoken by approximately 50,000–80,000 people across the highlands of Borneo — primarily in Sarawak (Malaysia), Kalimantan (Indonesia), and Brunei. It is severely under-resourced: no public parallel corpus existed before this project.

---

## How It Works

```
Browser (static/index.html)
  │  POST /api/translate
  ▼
FastAPI server (serve.py)
  │  OpenAI-compatible REST call
  ▼
Tinker inference API
  └─ Inkling-Small + LoRA (v2 dictionary retrain, checkpoint-16000, production default)
```

1. The user types Lun Bawang or English text in the browser.
2. The server auto-detects the source language from a vocabulary heuristic, then calls the Tinker API with a chat-format prompt.
3. The model generates a translation; `<think>…</think>` reasoning blocks are stripped before the response is returned. Every Inkling call passes `reasoning_effort="medium"` — see [The empty-output issue](#the-empty-output-issue-resolved) for why medium is the floor, and why a single retry at `"high"` is issued if a call comes back empty.
4. For the production default checkpoint, an exact full-phrase match against a small curated **phrasebook** bypasses the model entirely, for consistency with the on-page glossary and to sidestep the model's weakness on very short inputs. The comparison dropdown still shows raw model output.
5. Optionally, English input can be split into clauses and each clause translated independently, giving a secondary clause-by-clause translation alongside the whole-sentence result.

### Language auto-detection

The server maintains a curated set of ~250 common English function words, time words, nouns, verbs, and adjectives. If the input contains ≥2 English words, or ≥25% of its tokens match, it is classified as English (→ translate to Lun Bawang). Otherwise it is treated as Lun Bawang (→ translate to English). The user can override this at any time using the swap button.

### Clause-by-clause translation (English → Lun Bawang)

Long English sentences are split on commas, semicolons, and coordinating conjunctions (`and / but / or / so / yet / nor`). Conjunctions only trigger a split when both sides have ≥3 words, preventing over-splitting of short phrases like "bread and butter". Each clause is sent to the model separately, and both the whole-sentence and clause-level translations are shown side by side.

---

## Data

The training corpus combines a large Biblical parallel text with several smaller, higher-diversity everyday-language sources, plus the OCR'd Kemaloh Lundayeh dictionary added for the v2 retrain.

### 1. Lun Bawang Bible (primary corpus, ~30,000 verse pairs)

The only substantial public Lun Bawang text available is the full Bible translation published by the Sabah Theological Seminary. The PDF (`LunBawang-Bible.pdf`) was parsed with `parse_lun_bawang.py` into verse-level segments, then aligned verse-by-verse with the [World English Bible](https://worldenglish.bible) (WEB, public domain) using `build_parallel_corpus.py`, keyed on book code + chapter + verse. The result is `parallel_corpus.csv` (~30,000 matched verse pairs across 66 books).

**Train / val split:** 90% train / 10% val, stratified by Bible book.

### 2. Kemaloh Lundayeh–English Dictionary (v2 retrain corpus, 22,038 pairs)

A ~400-page scanned dictionary — Ganang, Crain & Pearson-Rounds, *Kemaloh Lundayeh–English Dictionary* — was OCR'd into structured entries via the Anthropic Batch API (Claude vision, forced tool-use into a strict entry schema: headword, homograph, senses, examples, variants, cross-references), then flattened into the corpus's `source,lun_bawang,english,type` schema by `corpus/build_dictionary_corpus.py`. The result is `corpus/dictionary_corpus.csv` with four row types:

| type | count | used for |
|------|-------|----------|
| `word` | 7,478 | headword ↔ gloss (both directions) |
| `definition` | 11,184 | headword → descriptive English gloss (**lb→en only**; a definition is a poor en→lb target) |
| `sentence` | 2,281 | example sentences (both directions, up-weighted 3×) |
| `redirect` | 1,095 | "see X" cross-references (dropped from training) |

This is the source that shifts the model from Bible-only fluency toward everyday vocabulary. See [Evaluation](#evaluation) for how it is weighted and gated.

### 3. Borneo Dictionary (word-level pairs, ~400 entries)

Word-level pairs from borneodictionary.com/lun-bawang/, parsed from repeating `LB_HEADWORD / English: … / Bahasa Malaysia: …` blocks by `build_aux_corpus.py`.

### 4. Longsemadoh WordPress (words + conversational sentences, ~350 entries)

A language-learning page from longsemadoh.wordpress.com using five different inline formats simultaneously (dialogue blocks, alternating EN/LB lines, numbered sentences, parenthetical LB, dash/equals-separated pairs). The parser handles all five and uses an English-word scoring heuristic to orient each pair. Entries with ≥3 Lun Bawang words are classified as `sentence`, shorter ones as `word`.

### 5. Mortensen (2021) — Laba' fairy tale (~54 sentence pairs)

Appendix A.1 of Mortensen's PhD dissertation "The Kemaloh Lun Bawang Language of Borneo" (University of Hawai'i) contains a Mouse-deer vs. Crocodile fairy tale in two-column parallel format — narrative dialogue, the domain farthest from the Biblical data. `parse_mortensen.py` uses pdfminer coordinate-based column separation to extract and align paragraph-level pairs. Used for non-commercial research purposes.

### Combined auxiliary corpus

`build_aux_corpus.py` combines sources 3, 4, and 5 into `aux_corpus.csv` (columns: `source`, `lun_bawang`, `english`, `type`). Split 80% train / 20% val, randomised per source. The dictionary corpus (source 2) is loaded and split separately (see [Evaluation](#evaluation)).

---

## Tinker Setup

[Tinker](https://thinkingmachines.ai/tinker/) is a hosted fine-tuning and inference service. It provides a Python SDK (`tinker`) and an OpenAI-compatible REST API for serving.

**Base model:** `thinkingmachines/Inkling-Small` (current production) or `Qwen/Qwen3-8B` (archived v0/v1 runs), selected with `--base-model`
**Adaptation:** LoRA, rank 16 · **Optimiser:** Adam, lr 5e-5 · **Batch size:** 8 · **Sequence length cap:** 384 tokens

```bash
export TINKER_API_KEY=your_key_here
```

The training client is created with `ServiceClient().create_lora_training_client(...)`. Each saved checkpoint has a `tinker://…` URI passed directly to the OpenAI-compatible inference endpoint as the `model` parameter.

### Training format

Every source pair produces datums using the model's chat template (native TML rendering for Inkling; the generic HF template for Qwen):

```
[system]  You are a translator specializing in the Lun Bawang language …
[user]    Translate to English:\n{lun_bawang_text}
[assistant] {english_text}
```

Only the assistant tokens contribute to the loss. Short-input hints (`Output only the translation of this word or phrase.`) are added for inputs ≤5 words at inference time.

### Training command

```bash
# Current recipe (Inkling base, dictionary corpus folded in):
python3.13 train_translator.py --train \
  --base-model thinkingmachines/Inkling-Small \
  --state-file tinker_state_inkling_v2.json
```

The script resumes automatically from the last checkpoint if the target state file exists, and full-resumes optimizer state where available. `--base-model thinkingmachines/*` requires `pip install tml-renderers torch` (these models use a native TML chat format, not the generic HF chat template). The v2 run was **warm-started** from Inkling checkpoint-11,500 (weights only; optimizer reset, with a 500-step LR warmup).

### Interactive CLI translation

```bash
python3.13 train_translator.py --translate --direction en2lb --text "In the beginning"
```

---

## Evaluation

Validation metrics are computed during training and written back to the state file. A fast `val_loss` (teacher-forced cross-entropy, no sampling) runs every 100 steps; the full bidirectional BLEU / exact-match / chrF eval runs every 2,000 steps against fixed, seeded held-out sets. All sampling-based eval uses `reasoning_effort="medium"`, matching serving.

### Current production model — v2 dictionary retrain (Inkling · Step 16,000)

The v2 run warm-starts from Inkling checkpoint-11,500 and continues for one epoch over the combined Bible + auxiliary + **dictionary** corpus. Recipe decisions (made with the maintainer):

- **Per-type direction gating** — `word`/`sentence` → both directions; `definition` → **lb→en only**; `redirect` → dropped.
- **Balanced weighting** — word 1×, definition 1×, sentence 3× → the dictionary is **~28% of the training mix**.
- **Long-definition trim** (>30 words) from train only, applied *after* the split so held-out rows still match eval.
- **Leakage guard** — drops any dict train row whose exact `(lb, en)` pair equals a held-out probe in any corpus.
- **Held-out dictionary eval slices** — new `dictword`/`dictsent` sets (both directions), seeded identically to the trainer's split so they are exactly the model's held-out rows. Word recall is scored with **chrF** (partial credit) alongside exact-match, because a single-word gold answer rarely matches token-for-token even when the meaning is right.

**Selection.** The goal was everyday English↔Lun Bawang quality, not literal Bible fidelity, so checkpoints were ranked by an everyday-quality blend — `dict-sentence BLEU (both directions) + general-sentence BLEU + 0.1·dict-word chrF` — excluding only any checkpoint where Bible BLEU catastrophically collapsed (a gibberish cliff; a moderate drop toward conversational register is expected and fine). The final call was made at `reasoning_effort="medium"` because at `low` a large fraction of single-word probes return empty completions and score zero, understating the dictionary metrics (see below).

**Contender comparison (medium effort):**

| Step | dict-sent lb→en | dict-sent en→lb | sentence lb→en | dict-word en→lb chrF | Bible lb→en | **blend** |
|------|-----------------|-----------------|----------------|----------------------|-------------|-----------|
| 8,000 | 13.3 | 7.3 | 24.6 | 24.6 | 60.2 | 47.0 |
| 12,000 | 12.3 | 10.5 | 25.2 | 24.8 | 59.3 | 49.6 |
| 14,000 | 12.7 | 3.6 | 26.6 | 20.6 | 57.2 | 44.5 |
| **16,000** | 12.6 | 10.0 | **27.9** | **27.1** | 58.9 | **52.3** |

**Step 16,000 is the best checkpoint of the run** — strongest general lb→en sentences and en→lb dictionary quality, with Bible fidelity holding (~57 BLEU, no collapse). It is now the served default; warmup, the phrasebook shortcut, and the dropdown "(default)" marker all follow it.

**Training trajectory (auto-evals, `low` effort — the curve, not the final selection):**

| Step | Val loss | Bible lb→en | dict-sent lb→en | dict-sent en→lb | dict-word chrF (lb / en) | Sent lb→en |
|------|----------|-------------|-----------------|-----------------|--------------------------|------------|
| 2,000 | 0.0081 | 64.1 | 12.9 | 1.9 | — | 26.5 |
| 4,000 | 0.0054 | 57.6 | 11.9 | 5.9 | — | 26.2 |
| 6,000 | 0.0048 | 66.3 | 10.4 | 7.8 | 17.0 / 20.5 | 24.8 |
| 8,000 | 0.0055 | 60.2 | 14.3 | 8.6 | 17.0 / 24.0 | 24.5 |
| 10,000 | 0.0022 | 64.5 | 12.4 | 8.8 | 17.4 / 25.2 | 22.5 |
| 12,000 | 0.0018 | 59.3 | 11.9 | 9.3 | 16.8 / 25.9 | 24.8 |
| 14,000 | 0.0021 | 57.2 | 13.3 | 7.0 | 16.2 / 16.5 | 26.6 |

The **en→lb direction improved steadily** across the run — the direction that was weakest and matters most for everyday use. Dictionary single-word *exact*-match stays low (~2–13%) throughout because single-word gold answers rarely match token-for-token; chrF is the metric to watch there. Raw per-translation outputs for every eval run live in `eval/eval_raw/*.jsonl` (merged into `eval/eval_outputs.csv`).

### The empty-output issue (resolved)

Inkling supports a `reasoning_effort` control. Training never includes an effort signal, so any value is somewhat out-of-distribution — and at `low` effort certain short inputs (single common words, short idioms) come back as a single end-of-message token with **zero generated content** rather than a wrong-but-present translation. Measured on 50 single-word lb→en probes:

| checkpoint | low | medium |
|------------|-----|--------|
| v1 · checkpoint-11,500 | 4% | 0% |
| v2 · checkpoint-16,000 | 36% | 0% |

The dictionary retrain did **not** fix this at the model level — v2 is actually *more* empty-prone at `low`, even though single words are what it trained on. At `medium` both collapse to 0%. **Mitigation shipped to production** (`serve.py`): every input starts at `reasoning_effort="medium"`, with a single retry at `"high"` on the rare empty. Eval uses `medium` too, so its numbers reflect what the served model actually does.

---

## Past experiments and versions

<details>
<summary><b>Qwen3-8B — v0 and v1 runs</b> (the original base model, superseded by Inkling)</summary>

All BLEU scores are sacrebleu corpus BLEU on fixed held-out sets. Bidirectional eval uses the same 50 Bible, 144 dictionary, and 40 sentence examples per checkpoint.

#### v0 — run 1 (model `42a2c780`, 16,000 steps)

| Step | Val loss | Bible LB→EN | Bible EN→LB | Dict LB→EN | Dict EN→LB | Sent LB→EN | Sent EN→LB |
|------|----------|------------|------------|-----------|-----------|-----------|-----------|
| 8,000 | 0.185 | 54.55 | 44.06 | 20.1% | 16.7% | 34.98² | 1.24 |
| 16,000 | 0.054 | 58.24 | — | 19.5% | — | 33.59¹ | — |

¹ Unidirectional (LB→EN only) eval; earlier eval code. ² Measured on the original sentence val set (longsemadoh only, before Mortensen prose was added in v1); with the current val set the score is 5.99.

Training stopped at step 16,000 (mid-epoch 3); val loss kept falling into epoch 3, indicating overfitting to Biblical register. Step 8,000 gave the best balance.

**Comparison with GPT models (v0 val set):**

| Model | Bible BLEU | Dict exact | Sentence BLEU | Avg ms/call |
|-------|-----------|-----------|---------------|-------------|
| **Our model (v0·checkpoint-8000)** | **51.70** | **20.3%** | **30.79** | ~5,400 |
| gpt-4o | 10.44 | 6.0% | 21.46 | ~716 |
| gpt-5-mini | 8.79 | 3.8% | 10.95 | ~18,300 |
| gpt-4o-mini | 7.22 | 2.3% | 11.84 | — |

The fine-tuned model scores ~5× higher on Bible BLEU and ~3× higher on Dict exact-match than the best general GPT model, confirming that Lun Bawang vocabulary is largely absent from general pre-training data.

#### v1 — run 2 (model `719fbcd8`, 11,500 steps)

v1 adds Mortensen (2021) narrative sentences and a small amount of user feedback. Sentence BLEU split by reference length: **sh** ≤10 words, **lg** >10 words.

| Step | Val loss | Bible LB→EN | Bible EN→LB | Dict LB→EN | Dict EN→LB | Sent LB→EN (sh / lg) | Sent EN→LB (sh / lg) |
|------|----------|------------|------------|-----------|-----------|----------------------|----------------------|
| 2,000 | 0.734 | 27.83 | 20.09 | 13.9% | 7.6% | 8.64 (13.2 / 7.6) | 6.33 (1.4 / 7.1) |
| 4,000 | 0.454 | 42.35 | 24.10 | 16.0% | 7.6% | 14.99 (15.3 / 14.7) | 9.65 (1.6 / 11.4) |
| 6,000 | 0.324 | 50.45 | 30.85 | 13.9% | 1.4%† | 15.70 (15.7 / 15.4) | 8.96 (3.1 / 9.9) |
| 8,000 | 0.173 | 48.49 | 47.47 | 22.2% | 12.5% | 14.90 (18.6 / 13.6) | 3.43 (5.3 / 3.5) |
| 10,000 | 0.147 | 46.88 | 46.70 | 23.6% | 16.7% | 21.72 (20.1 / 22.4) | 11.04 (2.3 / 12.8) |
| **11,500** | — | **58.82** | **56.48** | **24.3%** | 15.3% | **22.07 (20.3 / 22.2)** | **9.51 (4.2 / 10.3)** |

† Step 6,000 EN→LB dict anomaly: model was emitting `<think>` tokens at that checkpoint. **v1 · Step 11,500 was the best Qwen checkpoint**, superseded by Inkling.

</details>

<details>
<summary><b>Inkling-Small v1 — the step-matched 11,500 run</b> (predecessor to the v2 dictionary retrain)</summary>

[Inkling-Small](https://thinkingmachines.ai/) is Thinking Machines' small reasoning model, available on Tinker at lower per-token cost than Qwen3-8B. It was fine-tuned on the exact same corpus, LoRA config, and step count (11,500) as the archived v1 Qwen run, for a direct step-matched comparison. The v2 dictionary retrain warm-starts from this run's checkpoint-11,500.

Inkling uses a native TML chat format (`tml_renderers`) rather than a generic HF chat template. Training and inference prompts must be rendered through `tml_renderers.Renderer`, or the fine-tuned model produces empty output through the real serving API even though in-loop training-time eval looks fine. `train_translator.py` and `eval/eval_checkpoint.py` both auto-detect `thinkingmachines/*` base models and switch rendering paths accordingly.

#### Results by checkpoint (50 Bible, 144 dictionary, 40 sentence examples)

| Step | Val loss | Bible LB→EN | Bible EN→LB | Dict LB→EN | Dict EN→LB | Sent LB→EN | Sent EN→LB |
|------|----------|------------|------------|-----------|-----------|-----------|-----------|
| 2,000 | 0.465 | 48.68 | 27.73 | 17.4% | 12.5% | 20.36 | 6.92 |
| 4,000 | 0.224 | 52.25 | 45.43 | 18.1% | 14.6% | 16.64 | 7.98 |
| 6,000 | 0.125 | 45.88 | 54.96 | 18.8% | 22.2% | 19.51 | 11.92 |
| 8,000 | 0.089 | 58.14 | 61.97 | 9.7% | 13.2% | 23.74 | 3.12 |
| 10,000 | 0.051 | 66.46 | 67.00 | 16.0% | 12.5% | 18.87 | 5.75 |
| **11,500** | **0.044** | **59.11** | **71.59** | **17.4%** | **18.8%** | **24.57** | **9.85** |

#### Inkling-Small vs. Qwen3-8B v1 (step 11,500)

| Metric | Inkling-Small | Qwen v1 | Winner |
|--------|---------------|---------|--------|
| Bible BLEU LB→EN | 59.11 | 58.82 | ~tied |
| Bible BLEU EN→LB | **71.59** | 56.48 | Inkling (+15.1) |
| Dict exact LB→EN | 17.4% | **24.3%** | Qwen (+6.9pp) |
| Dict exact EN→LB | **18.8%** | 15.3% | Inkling (+3.5pp) |
| Sent BLEU LB→EN | **24.57** | 22.07 | Inkling (+2.5) |
| Sent BLEU EN→LB | **9.85** | 9.51 | ~tied |

Inkling-Small won or tied on 5 of 6 metrics and became the production default before the v2 dictionary retrain superseded it.

</details>

### BLEU context

A Bible BLEU above 50 for a rare language with ~30k training sentences is strong — Google Translate achieves ~40 BLEU for well-resourced pairs like French→English. Published results on similarly low-resource languages typically fall in the 20–35 range at equivalent data scales. The ceiling is partly set by reference quality and the domain gap: training data is primarily Biblical, while the sentence/dictionary eval sets are conversational and lexical.

---

## User Feedback Loop

The web UI collects thumbs up/down feedback on every translation, feeding future fine-tuning runs.

### How feedback is collected

- **Thumbs up** — records the translation as a correct example.
- **Thumbs down** — optionally prompts for a correction; if provided, the correction is used as the training target; thumbs-down with no correction is discarded.

Each entry records source text, direction, checkpoint, model output, rating, and correction. IPs are truncated to the `/24` prefix before being written anywhere.

### Storage

**Sole store:** `eval/feedback.csv` in the repo — if `GITHUB_TOKEN` is set, every submission triggers an async commit that appends the row. No local database; data survives Render redeploys.

### Reviewing and preparing training data

```bash
python3.13 eval/review_feedback.py --dry-run                # summary + QC flags
python3.13 eval/review_feedback.py --csv eval/feedback.csv  # writes feedback_corpus.csv
```

The QC script filters no-ops, self-copies, and empty corrections, and flags unusual submission volume. Output `feedback_corpus.csv` uses the same schema as `aux_corpus.csv`. `train_translator.py` loads it automatically and repeats feedback entries 10× (vs. 5× for aux) as high-confidence human signal.

---

## Running Locally

### Prerequisites

- Python 3.13 and a Tinker API key.

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `fastapi`, `uvicorn`, `openai`, `sacrebleu`, `requests`, `beautifulsoup4`.

### Run the web server

```bash
export TINKER_API_KEY=your_key_here
python3.13 serve.py            # → http://localhost:8000  (use --port to change)
```

The server reads `tinker_state.json` (and the per-run state files) to discover checkpoints. If none exist yet, the UI shows a "training in progress" notice.

### Rebuild the corpora (optional)

```bash
python3.13 corpus/parse_lun_bawang.py       # 1. Parse the Lun Bawang Bible PDF into verses
python3.13 corpus/build_parallel_corpus.py  # 2. Align with World English Bible
python3.13 corpus/parse_mortensen.py        # 3. Parse the Mortensen (2021) appendix (needs the PDF)
python3.13 corpus/build_aux_corpus.py       # 4. Build the auxiliary word/sentence corpus
python3.13 corpus/build_dictionary_corpus.py  # 5. Flatten OCR'd dictionary entries → dictionary_corpus.csv
python3.13 train_translator.py --train --base-model thinkingmachines/Inkling-Small \
  --state-file tinker_state_inkling_v2.json   # 6. Train (requires TINKER_API_KEY)
```

Dictionary OCR itself (scans → `corpus/dictionary_entries.jsonl`) is run by `corpus/ocr_dictionary.py` against the Anthropic Batch API and needs `ANTHROPIC_API_KEY` plus the page images.

---

## Deployment (Render)

The app is deployed on [Render](https://render.com) as a web service; on startup Render runs `python3.13 serve.py`. `TINKER_API_KEY` is a Render environment variable. The committed state files (`tinker_state*.json`) tell the server which checkpoints exist and where to find them on Tinker — no model weights are stored in the repo.

The checkpoint dropdown lets you compare any saved checkpoint across every run. **v2 · Step 16,000 is pre-selected as the default.**

---

## Project Structure

```
lunbawang-translate/
├── serve.py                     # FastAPI web server + translation API + phrasebook
├── train_translator.py          # Fine-tuning + in-loop evaluation
├── orchestrate_retrain.py       # Recycle-proof driver used to run the v2 retrain on an ephemeral host
├── cleanup_checkpoints.py       # Utility for pruning Tinker checkpoints
├── tinker_state.json            # Checkpoint metadata for the current run
├── tinker_state_v1.json         # Archived Qwen3-8B v1 run
├── tinker_state_inkling.json    # Inkling-Small v1 (step-matched 11,500 run)
├── tinker_state_inkling_v2.json # Inkling v2 dictionary retrain — production default (step 16,000)
├── feedback_corpus.csv          # Generated by eval/review_feedback.py; loaded by training
├── requirements.txt
├── static/index.html            # Single-page frontend
│
├── corpus/                      # Corpus building: source files, scripts, output data
│   ├── parse_lun_bawang.py      # Extract verses from LunBawang-Bible.pdf
│   ├── build_parallel_corpus.py # Align LB Bible verses with WEB English
│   ├── parse_mortensen.py       # Extract parallel pairs from the Mortensen (2021) PDF
│   ├── build_aux_corpus.py      # Parse borneodict.txt + longsemadoh.txt + Mortensen
│   ├── ocr_dictionary.py        # OCR the Kemaloh Lundayeh dictionary scans (Anthropic Batch API)
│   ├── build_dictionary_corpus.py  # Flatten OCR'd entries → dictionary_corpus.csv
│   ├── parallel_corpus.csv      # ~30k Bible verse pairs
│   ├── aux_corpus.csv           # ~800 word/sentence pairs from web + Mortensen
│   ├── dictionary_corpus.csv    # 22,038 pairs from the Kemaloh Lundayeh dictionary
│   └── dictionary_entries.jsonl # Structured OCR output (pre-flatten)
│
└── eval/                        # Evaluation + feedback review scripts and outputs
    ├── eval_checkpoint.py       # Standalone bidirectional BLEU/exact/chrF eval for a checkpoint
    ├── eval_openai.py           # Eval any OpenAI model on the same val set
    ├── merge_evals.py           # Merge eval_raw/*.jsonl → eval_outputs.csv
    ├── review_feedback.py       # QC + prepare feedback_corpus.csv
    ├── feedback.csv             # Sole feedback store
    ├── eval_outputs.csv         # Combined eval results across all runs
    └── eval_raw/*.jsonl         # Per-run raw model outputs (one JSON object per translation)
```

---

## Limitations

- **Single-word gloss recall:** the model gets sentence register and structure right more reliably than rare single-word translations, sometimes defaulting to a generic gloss ("a variety of tree") for a word it doesn't know. Dictionary exact-match stays low; chrF (partial credit) is the better signal there.
- **Sentence fidelity:** conversational translations are fluent and in the right register but don't always preserve exact meaning (BLEU ~12–13 on held-out dictionary sentences).
- **En→LB is harder to verify:** the weaker direction, and harder to check without a native speaker — though it improved most across the v2 run.
- **No morphological analysis:** Lun Bawang has productive affixation (nasal prefixes, infixes, reduplication), learned implicitly from examples rather than explicit structure.
- **Rare residual empty output:** ~1% of single-word inputs still return empty even at `medium` effort; the server retries once at `high`, and a repeat attempt almost always succeeds.

---

## Acknowledgements

- Lun Bawang Bible translation: Sabah Theological Seminary / Bible Society of Malaysia
- English reference: [World English Bible](https://worldenglish.bible) (public domain)
- Dictionary data: Ganang, R., Crain, J., & Pearson-Rounds, V. *Kemaloh Lundayeh–English Dictionary* (OCR'd for the v2 retrain, non-commercial research use); borneodictionary.com
- Phrasebook data: longsemadoh.wordpress.com
- Narrative parallel text: Mortensen, M. (2021). *The Kemaloh Lun Bawang Language of Borneo*. PhD dissertation, University of Hawai'i at Mānoa. Non-commercial research use.
- Fine-tuning infrastructure: [Tinker](https://thinkingmachines.ai/tinker/) by Thinking Machines
- Base models: Inkling-Small by Thinking Machines (current); [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by Alibaba Cloud (archived v0/v1)
