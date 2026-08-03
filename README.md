# LunBawang Translate

**Live demo:** https://translate.lunbawang.com/

A bidirectional **Lun Bawang ↔ English** machine translator, built by fine-tuning language models on the first parallel corpus ever assembled for the language. The web interface is deployed on Render and runs against a fine-tuned model served via the [Tinker](https://thinkingmachines.ai/tinker/) API. The production default is **Inkling-Small · Step 11,500** — see [Inkling-Small Experiment](#inkling-small-experiment) for the full comparison against the original Qwen3-8B run.

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
  └─ Inkling-Small + LoRA fine-tune (checkpoint-11500, production default)
```

1. The user types Lun Bawang or English text in the browser.
2. The server auto-detects the source language from a vocabulary heuristic, then calls the Tinker API with a chat-format prompt.
3. The model generates a translation; `<think>…</think>` reasoning blocks (from Qwen3's chain-of-thought mode) are stripped before the response is returned. For Inkling checkpoints, `reasoning_effort="low"` is passed on every call — see [Inkling-Small Experiment](#inkling-small-experiment) for why, and why a single retry at `"medium"` is issued if the first call comes back empty.
4. Optionally, English input can be split into clauses and each clause translated independently, giving a secondary clause-by-clause translation alongside the whole-sentence result.

### Language auto-detection

The server maintains a curated set of ~250 common English function words, time words, nouns, verbs, and adjectives. If the input contains ≥2 English words, or ≥25% of its tokens match, it is classified as English (→ translate to Lun Bawang). Otherwise it is treated as Lun Bawang (→ translate to English). The user can override this at any time using the swap button.

### Clause-by-clause translation (English → Lun Bawang)

Long English sentences are split on commas, semicolons, and coordinating conjunctions (`and / but / or / so / yet / nor`). Conjunctions only trigger a split when both sides have ≥3 words, preventing over-splitting of short phrases like "bread and butter". Each clause is sent to the model separately, and both the whole-sentence and clause-level translations are shown side by side.

---

## Data

Four sources were used, totalling ~31,000 training pairs.

### 1. Lun Bawang Bible (primary corpus, ~30,000 verse pairs)

The only substantial public Lun Bawang text available is the full Bible translation published by the Sabah Theological Seminary. The PDF (`LunBawang-Bible.pdf`) was parsed with `parse_lun_bawang.py` into verse-level segments, then aligned verse-by-verse with the [World English Bible](https://worldenglish.bible) (WEB, public domain) using `build_parallel_corpus.py`.

The alignment keys on book code + chapter + verse number. The result is `parallel_corpus.csv` (~30,000 matched verse pairs across 66 books).

**Train / val split:** 90% train / 10% val, stratified by Bible book, so every book appears in both sets.

### 2. Borneo Dictionary (word-level pairs, ~400 entries)

Word-level Lun Bawang ↔ English pairs from borneodictionary.com/lun-bawang/ were copied into `borneodict.txt`. The parser (`build_aux_corpus.py`) reads repeating blocks of the form:

```
LB_HEADWORD
English: DEFINITION
Bahasa Malaysia: BM_DEFINITION
```

Each headword / English definition pair becomes one training entry.

### 3. Longsemadoh WordPress (words + conversational sentences, ~350 entries)

A language learning page from longsemadoh.wordpress.com was copied into `longsemadoh.txt`. This source uses five different inline formats simultaneously (dialogue blocks, alternating EN/-LB lines, numbered sentences, parenthetical LB, and dash/equals-separated pairs). The parser handles all five formats and uses an English-word scoring heuristic to orient each pair correctly (LB side vs. English side).

Entries with ≥3 Lun Bawang words are classified as `sentence`; shorter entries as `word`.

### 4. Mortensen (2021) — Laba' fairy tale (~54 sentence pairs)

Appendix A.1 of Mortensen's PhD dissertation "The Kemaloh Lun Bawang Language of Borneo" (University of Hawai'i, UMI #10969) contains a full Mouse-deer vs. Crocodile fairy tale in two-column parallel format: Lun Bawang prose on the left, English translation on the right. This is narrative dialogue — the domain farthest from the Biblical training data — and is almost certainly absent from any LLM's pre-training corpus.

`parse_mortensen.py` uses pdfminer coordinate-based column separation (left column x < 250 = LB, right column x ≥ 250 = EN) to extract and align paragraph-level pairs. Footnote starters (lines matching `^\d+[A-Za-z]`) and their continuation paragraphs are filtered; inline footnote reference numbers (e.g., `em,1 uten`) are stripped from the LB text. Output: `mortensen_corpus.csv`.

Used for non-commercial research purposes.

### Combined auxiliary corpus

`build_aux_corpus.py` combines sources 2, 3, and 4 into `aux_corpus.csv` (columns: `source`, `lun_bawang`, `english`, `type`).

**Train / val split:** 80% train / 20% val, randomised per source, so each source appears in both train and val.

**Up-weighting:** Auxiliary training datums are repeated 5× in the training loop to compensate for their small size relative to the ~54,000 Bible datums (2 directions × 27,000 train verses).

---

## Tinker Setup

[Tinker](https://thinkingmachines.ai/tinker/) is a hosted fine-tuning and inference service. It provides a Python SDK (`tinker`) and an OpenAI-compatible REST API for serving.

**Base model:** `Qwen/Qwen3-8B` (archived v0/v1 runs) or `thinkingmachines/Inkling-Small` (current production default — see [Inkling-Small Experiment](#inkling-small-experiment)), selected with `--base-model`
**Adaptation:** LoRA, rank 16
**Optimiser:** Adam, lr 5e-5
**Batch size:** 8
**Sequence length cap:** 384 tokens (longer pairs are dropped)

To use Tinker you need an API key set as:

```bash
export TINKER_API_KEY=your_key_here
```

The training client is created with `ServiceClient().create_lora_training_client(...)`. Each saved checkpoint has a `tinker://…` URI that is passed directly to the OpenAI-compatible inference endpoint as the `model` parameter.

### Training format

Every source pair produces two datums — LB→EN and EN→LB — using Qwen3's standard chat template:

```
[system]  You are a translator specializing in the Lun Bawang language …
[user]    Translate to English:\n{lun_bawang_text}
[assistant] {english_text}
```

Only the assistant tokens contribute to the loss (weights = 0 on prompt, 1 on completion). Short-input hints (`Output only the translation of this word or phrase.`) are added for inputs ≤5 words at inference time.

### Training command

```bash
python3.13 train_translator.py --train
# Alternate base model / separate experiment, e.g. Inkling-Small:
python3.13 train_translator.py --train \
  --base-model thinkingmachines/Inkling-Small \
  --state-file tinker_state_inkling.json \
  --max-steps 11500   # match an existing run's step count for direct comparison
```

The script resumes automatically from the last checkpoint if the target state file exists. Progress is logged to stdout and checkpoint metadata is saved back to the state file after every 500 steps. `--base-model thinkingmachines/*` requires `pip install tml-renderers torch` (see [Inkling-Small Experiment](#inkling-small-experiment) — these models use a native TML chat format, not the generic HF chat template).

### Interactive CLI translation

```bash
python3.13 train_translator.py --translate
python3.13 train_translator.py --translate --direction en2lb --text "In the beginning"
```

---

## Evaluation

Three validation metrics are computed during training:

| Metric | Subset | Frequency | Method |
|--------|--------|-----------|--------|
| `val_loss` | 200 random Bible val datums | Every 500 steps | Cross-entropy forward pass, no sampling |
| `val_bleu_bible` | 50 sampled Bible val pairs (LB→EN) | Every 2,000 steps | sacrebleu corpus BLEU |
| `val_exact_dict` | All 133 dictionary val pairs (LB→EN) | Every 2,000 steps | Case-insensitive exact string match |
| `val_bleu_sentence` | All 16 sentence val pairs (LB→EN) | Every 2,000 steps | sacrebleu corpus BLEU |

`val_loss` is fast (no sampling, no cold start). BLEU and exact-match require spawning a sampling client, so they run less frequently.

### Results by checkpoint

All BLEU scores are sacrebleu corpus BLEU on fixed held-out sets. Bidirectional evaluation (LB→EN and EN→LB) uses the same 50 Bible, 144 dictionary, and 40 sentence examples for every checkpoint.

#### v0 — run 1 (model `42a2c780`, 16,000 steps)

| Step | Val loss | Bible LB→EN | Bible EN→LB | Dict LB→EN | Dict EN→LB | Sent LB→EN | Sent EN→LB |
|------|----------|------------|------------|-----------|-----------|-----------|-----------|
| 8,000 | 0.185 | 54.55 | 44.06 | 20.1% | 16.7% | 34.98² | 1.24 |
| 16,000 | 0.054 | 58.24 | — | 19.5% | — | 33.59¹ | — |

¹ Unidirectional (LB→EN only) eval; earlier eval code.
² Measured on the original sentence val set (longsemadoh conversational sentences only, before Mortensen narrative prose was added to the val set in v1). With the current val set the score is 5.99.

Training stopped at step 16,000 (mid-epoch 3). Val loss continued falling steeply into epoch 3, indicating overfitting to Biblical register. Step 8,000 gave the best balance between Bible BLEU and generalisation; step 16,000 was stronger on Biblical prose but weaker on conversational text.

##### Comparison with GPT models (v0 val set)

The same val sets were run against general-purpose OpenAI models to establish a baseline for what an off-the-shelf LLM can do with no Lun Bawang-specific training. Scripts: `eval/eval_openai.py`, `eval/eval_checkpoint.py`. Raw outputs saved to `eval/eval_raw/`; combined into `eval/eval_outputs.csv` via `eval/merge_evals.py`.

| Model | Bible BLEU | Dict exact match | Sentence BLEU | Avg ms/call | Notes |
|-------|-----------|-----------------|---------------|-------------|-------|
| **Our model (v0·checkpoint-8000)** | **51.70** | **20.3%** | **30.79** | ~5,400 | Qwen3-8B + LoRA, Tinker inference |
| gpt-4o | 10.44 | 6.0% | 21.46 | ~716 | |
| gpt-5-mini | 8.79 | 3.8% | 10.95 | ~18,300 | Reasoning model; slow despite "mini" label |
| gpt-4o-mini | 7.22 | 2.3% | 11.84 | — | |

Key takeaways:
- Our fine-tuned model scores **5× higher** on Bible BLEU and **3× higher** on Dict exact match than the best general GPT model (gpt-4o), despite being 8B parameters vs. GPT-4o's much larger scale.
- gpt-5-mini is a reasoning model and takes ~18s per API call — 25× slower than gpt-4o and 3× slower than our model on Tinker. Its BLEU scores are also worse, suggesting reasoning capability does not compensate for lack of Lun Bawang training data.
- All GPT models struggle with the dictionary exact-match task (single-word translations), confirming that Lun Bawang vocabulary is largely absent from general pre-training data.
- The ~5,400ms timing reflects Tinker's cold-start latency; warm-cache requests are faster.

#### v1 — run 2 (model `719fbcd8`, 11,500 steps)

v1 adds manually transcribed entries from the Mortensen (2021) dissertation appendix and a small amount of user feedback from the live site to the training set. The Mortensen narrative sentences also join the sentence val set, making v1 sentence BLEU scores not directly comparable to v0.

Sentence BLEU is split by reference length: **sh** = short (≤10 words), **lg** = long (>10 words).

| Step | Val loss | Bible LB→EN | Bible EN→LB | Dict LB→EN | Dict EN→LB | Sent LB→EN (sh / lg) | Sent EN→LB (sh / lg) |
|------|----------|------------|------------|-----------|-----------|----------------------|----------------------|
| 2,000 | 0.734 | 27.83 | 20.09 | 13.9% | 7.6% | 8.64 (13.2 / 7.6) | 6.33 (1.4 / 7.1) |
| 4,000 | 0.454 | 42.35 | 24.10 | 16.0% | 7.6% | 14.99 (15.3 / 14.7) | 9.65 (1.6 / 11.4) |
| 6,000 | 0.324 | 50.45 | 30.85 | 13.9% | 1.4%† | 15.70 (15.7 / 15.4) | 8.96 (3.1 / 9.9) |
| 8,000 | 0.173 | 48.49 | 47.47 | 22.2% | 12.5% | 14.90 (18.6 / 13.6) | 3.43 (5.3 / 3.5) |
| 10,000 | 0.147 | 46.88 | 46.70 | 23.6% | 16.7% | 21.72 (20.1 / 22.4) | 11.04 (2.3 / 12.8) |
| **11,500** | — | **58.82** | **56.48** | **24.3%** | 15.3% | **22.07 (20.3 / 22.2)** | **9.51 (4.2 / 10.3)** |

† Step 6,000 EN→LB dict anomaly: model was emitting `<think>` tokens at that checkpoint.

#### Best checkpoint comparison

All scores on the v1 val set (50 Bible, 144 dict, 40 sentence examples including Mortensen narrative prose). Sentence BLEU: **sh** = short (≤10 words), **lg** = long (>10 words).

| Metric | v0 · Step 8,000 | **v1 · Step 11,500** |
|--------|-----------------|----------------------|
| Bible BLEU LB→EN | 54.55 | **58.82** |
| Bible BLEU EN→LB | 44.06 | **56.48** |
| Dict exact LB→EN | 20.1% | **24.3%** |
| Dict exact EN→LB | 16.7% | 15.3% |
| Sent BLEU LB→EN | 5.99 (13.8 / 2.6) | **22.07 (20.3 / 22.2)** |
| Sent BLEU EN→LB | 1.24 (1.4 / 1.4) | **9.51 (4.2 / 10.3)** |

**v1 · Step 11,500 is the current default checkpoint.** It achieves the best Bible BLEU in both directions and best dictionary exact-match, and EN→LB is substantially improved across the board compared to v0. The sentence BLEU gap is particularly stark: v0 scores 5.99 LB→EN and 1.24 EN→LB on the v1 val set; v1 scores 22.07 and 9.51.

---

## Inkling-Small Experiment

[Inkling-Small](https://thinkingmachines.ai/) is Thinking Machines' small reasoning model, available on Tinker at a lower per-token cost than Qwen3-8B. It was fine-tuned on the exact same corpus, LoRA config (rank 16, lr 5e-5, batch size 8), and step count (11,500) as the archived v1 Qwen run, for a direct step-matched comparison.

### Training format differences

Inkling uses a native TML chat format (`tml_renderers`) rather than a generic HF chat template — `<|message_system|>`, `<|content_text|>`, `<|end_message|>`, etc. — which Tinker's serving endpoint expects server-side. Training and inference prompts must be rendered through `tml_renderers.Renderer` rather than `tokenizer.apply_chat_template()`, or the fine-tuned model produces empty output through the real serving API even though in-loop training-time eval looks fine (the two use different tokenization paths). `train_translator.py` and `eval/eval_checkpoint.py` both auto-detect `thinkingmachines/*` base models and switch rendering paths accordingly.

### Results by checkpoint

Same bidirectional eval as the v1 table above (50 Bible, 144 dictionary, 40 sentence examples).

| Step | Val loss | Bible LB→EN | Bible EN→LB | Dict LB→EN | Dict EN→LB | Sent LB→EN | Sent EN→LB |
|------|----------|------------|------------|-----------|-----------|-----------|-----------|
| 2,000 | 0.465 | 48.68 | 27.73 | 17.4% | 12.5% | 20.36 | 6.92 |
| 4,000 | 0.224 | 52.25 | 45.43 | 18.1% | 14.6% | 16.64 | 7.98 |
| 6,000 | 0.125 | 45.88 | 54.96 | 18.8% | 22.2% | 19.51 | 11.92 |
| 8,000 | 0.089 | 58.14 | 61.97 | 9.7% | 13.2% | 23.74 | 3.12 |
| 10,000 | 0.051 | 66.46 | 67.00 | 16.0% | 12.5% | 18.87 | 5.75 |
| **11,500** | **0.044** | **59.11** | **71.59** | **17.4%** | **18.8%** | **24.57** | **9.85** |

### Final comparison: Inkling-Small vs. Qwen3-8B v1 (step 11,500)

| Metric | Inkling-Small | Qwen v1 | Winner |
|--------|--------------:|--------:|--------|
| Bible BLEU LB→EN | 59.11 | 58.82 | ~tied |
| Bible BLEU EN→LB | **71.59** | 56.48 | Inkling (+15.1) |
| Dict exact LB→EN | 17.4% | **24.3%** | Qwen (+6.9pp) |
| Dict exact EN→LB | **18.8%** | 15.3% | Inkling (+3.5pp) |
| Sent BLEU LB→EN | **24.57** | 22.07 | Inkling (+2.5) |
| Sent BLEU EN→LB | **9.85** | 9.51 | ~tied |

Inkling-Small wins or ties on 5 of 6 metrics, with a decisive and consistent edge on Bible EN→LB throughout the run. Qwen retains a real advantage on dictionary LB→EN exact-match. **Inkling-Small · Step 11,500 is the new production default** (see [Deployment](#deployment-render)).

### The empty-output issue

Partway through this run, a serving-time failure mode surfaced: certain short inputs — mostly single common words or short idioms ("eat", "drink", "yes", "straight ahead", "turn left") — occasionally came back as a single end-of-message token with **zero generated content**, rather than a wrong-but-present translation. This section documents the investigation because it's a real, checkpoint-intrinsic quirk of Inkling that anyone else fine-tuning it should watch for.

**Root cause.** Inkling supports a `reasoning_effort` control (`none` / `low` / `medium` / `high` / …) that governs how much internal deliberation it does before answering. Training (`render_for_sft()`) never includes any effort signal, so *any* effort value used at inference time is somewhat out-of-distribution — but `reasoning_effort="none"` (the default when unset) was measured to reproduce a **100% empty-completion rate** on a curated set of known-hard short phrases. We initially tried to control this via a `"Thinking effort level: N"` system-message string (mirroring the native SDK's completion renderer) — this turned out to be **inert noise** over the OpenAI-compatible REST endpoint (varying its value from 0 to 0.99 gave inconsistent, non-monotonic empty rates, 12–18/27). The actual fix was the OpenAI client's real `reasoning_effort` parameter, passed outside the message list — `"low"` cut empty completions from 100% (at `"none"`) down to roughly 19% on the hardest known cases in isolated testing.

**Empty-output rate over the course of training** (measured directly from the full eval JSONL at each checkpoint, `reasoning_effort="low"` used from step 10,000 onward — earlier checkpoints used the inert text-message workaround, effectively `"none"`):

| Step | Overall | Dict LB→EN | Dict EN→LB | Sent LB→EN | Sent EN→LB |
|------|--------:|-----------:|-----------:|-----------:|-----------:|
| 2,000 | 0% | 0% | 0% | 0% | 0% |
| 4,000 | 13% | 5% | 34% | 0% | 12% |
| 6,000 | 1% | 3% | 0% | 0% | 2% |
| 8,000 | 30% | 41% | 50% | 0% | 25% |
| 10,000 | 15% | 13% | 31% | 0% | 15% |
| **11,500** | **4%** | **8%** | **4%** | **0%** | **2%** |

For comparison, **Qwen v1 has a 0% empty rate on the identical eval set at every checkpoint** — this is an Inkling-specific quirk, not a general artifact of fine-tuning on this corpus. It's also concentrated almost entirely in EN→LB and dictionary-style short inputs; Bible verses and LB→EN sentences never produced an empty result at any checkpoint measured.

**Mitigation shipped to production** (`serve.py`): every Inkling call uses `reasoning_effort="low"` by default (fast, and correct the overwhelming majority of the time). If a call comes back empty, the server automatically retries once at `reasoning_effort="medium"` before returning to the user — `"medium"`/`"high"` measured meaningfully lower failure rates than `"low"` on the hardest cases in testing. This escalation only ever triggers for Inkling checkpoints and only on the rare empty result, so it doesn't add latency to the common case. In manual testing against the ten hardest known-empty phrases from the final checkpoint's eval data, this brought the empty rate to 0/10.

### BLEU context

A Bible BLEU score above 50 for a rare language with ~30k training sentences is strong. Google Translate achieves ~40 BLEU for well-resourced language pairs like French→English with billions of sentence pairs. For comparison, published results on similarly low-resource languages (Swahili, Welsh, Basque at small data scales) typically fall in the 20–35 range with equivalent training set sizes.

The ceiling is partly set by reference translation quality and the domain gap: training data is primarily Biblical, while the sentence evaluation set is conversational and narrative.

---

## User Feedback Loop

The web UI collects thumbs up/down feedback on every translation. This data feeds back into future fine-tuning runs to improve quality on real user input — particularly conversational Lun Bawang, which the current training data (almost entirely Biblical prose) does not cover well.

### How feedback is collected

After each translation, a 👍 / 👎 widget appears below the output:

- **Thumbs up** — records the translation as a correct example
- **Thumbs down** — optionally prompts for a correction; if provided, the corrected translation is used as the training target instead of the model's output; thumbs-down with no correction is discarded (we know it's wrong but not what's right)

Each feedback entry records: source text, translation direction (LB→EN or EN→LB), checkpoint used, model output, rating, and correction (if provided). IPs are truncated to the `/24` prefix (e.g. `1.2.3.x`) before being written anywhere.

### Storage

- **Sole store:** `eval/feedback.csv` in the GitHub repo — if `GITHUB_TOKEN` is set, every submission triggers an async commit that fetches the current CSV, appends the new row, and writes it back. No local database or file is used; data survives Render redeploys automatically.

### Reviewing and preparing training data

```bash
python3.13 eval/review_feedback.py --dry-run                # summary + QC flags, no file written
python3.13 eval/review_feedback.py --csv eval/feedback.csv  # writes feedback_corpus.csv
```

The QC script filters no-ops (user submitted the same text as the correction), self-copies (correction matches source), and empty corrections. It flags IP addresses submitting an unusual volume of entries for manual review.

Output is `feedback_corpus.csv` with the same schema as `aux_corpus.csv` (`source`, `lun_bawang`, `english`, `type`).

### Using feedback in training

`train_translator.py` automatically loads `feedback_corpus.csv` if present. Feedback entries are repeated **10×** during training (vs. 5× for the aux corpus) — they represent high-confidence human signal and the dataset will be small relative to the ~30k Bible pairs.

```bash
python3.13 train_translator.py --train
# → "Loading feedback corpus… N feedback entries"
```

Feedback val entries are merged into the existing dict/sentence evaluation sets, so BLEU and exact-match scores automatically reflect improvement on user-corrected examples.

---

## Running Locally

### Prerequisites

- Python 3.13
- A Tinker API key

### Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `fastapi`, `uvicorn`, `openai`, `sacrebleu`, `requests`, `beautifulsoup4`

### Run the web server

```bash
export TINKER_API_KEY=your_key_here
python3.13 serve.py
# → http://localhost:8000
```

Custom port:

```bash
python3.13 serve.py --port 8080
```

The server reads `tinker_state.json` to discover available checkpoints. If `tinker_state.json` is absent or has no checkpoints yet, the UI shows a "training in progress" notice and polls every 20 seconds.

### Rebuild the corpora (optional)

These steps are only needed if you want to retrain from scratch:

```bash
# 1. Parse the Lun Bawang Bible PDF into verses
python3.13 corpus/parse_lun_bawang.py

# 2. Align with World English Bible
python3.13 corpus/build_parallel_corpus.py

# 3. Parse Mortensen (2021) dissertation appendix (requires the PDF)
python3.13 corpus/parse_mortensen.py

# 4. Build the auxiliary word/sentence corpus (includes Mortensen if present)
python3.13 corpus/build_aux_corpus.py

# 5. Train (requires TINKER_API_KEY)
python3.13 train_translator.py --train
```

---

## Deployment (Render)

The app is deployed on [Render](https://render.com) as a web service. On startup Render runs:

```
python3.13 serve.py
```

`TINKER_API_KEY` is set as a Render environment variable. The committed state files (`tinker_state.json`, `tinker_state_v1.json`, `tinker_state_inkling.json`, …) tell the server which checkpoints exist and where to find them on Tinker's infrastructure — no model weights are stored in the repo.

The checkpoint dropdown in the UI lets you compare any saved checkpoint across every run. **Inkling · Step 11,500 is pre-selected as the default** — see [Inkling-Small Experiment](#inkling-small-experiment) for why.

---

## Project Structure

```
raretranslator/
├── serve.py                  # FastAPI web server + translation API
├── train_translator.py       # Fine-tuning + evaluation script
├── tinker_state.json         # Checkpoint metadata for the current run
├── tinker_state_v1.json      # Archived Qwen3-8B v1 run (comparison baseline)
├── tinker_state_inkling.json # Inkling-Small run — production default (see README)
├── feedback_corpus.csv       # Generated by eval/review_feedback.py; loaded by training
├── requirements.txt
├── static/
│   └── index.html            # Single-page frontend
│
├── corpus/                   # Corpus building: source files, scripts, and output data
│   ├── parse_lun_bawang.py   # Extract verses from LunBawang-Bible.pdf
│   ├── build_parallel_corpus.py  # Align LB Bible verses with WEB English
│   ├── parse_mortensen.py    # Extract parallel pairs from Mortensen (2021) dissertation PDF
│   ├── build_aux_corpus.py   # Parse borneodict.txt + longsemadoh.txt + Mortensen
│   ├── LunBawang-Bible.pdf   # Source: full Lun Bawang Bible translation
│   ├── Mortensen_hawii_0085A_10969.pdf  # Source: Mortensen (2021) dissertation
│   ├── borneodict.txt        # Copied from borneodictionary.com
│   ├── longsemadoh.txt       # Copied from longsemadoh.wordpress.com
│   ├── parallel_corpus.csv   # ~30k Bible verse pairs (LB + EN)
│   ├── aux_corpus.csv        # ~800 word/sentence pairs from web + Mortensen sources
│   └── mortensen_corpus.csv  # ~54 sentence pairs from Mortensen (2021) fairy tale
│
└── eval/                     # Evaluation + feedback review scripts and outputs
    ├── eval_checkpoint.py    # Standalone BLEU/exact-match eval for a single checkpoint
    ├── eval_openai.py        # Eval any OpenAI model on the same val set
    ├── merge_evals.py        # Merge per-run eval_raw/*.jsonl into eval_outputs.csv
    ├── review_feedback.py    # QC + prepare feedback_corpus.csv for training
    ├── feedback.csv          # sole feedback store — appended to on every submission
    ├── eval_outputs.csv      # Combined eval results across all runs
    ├── eval_results_openai.json
    └── eval_raw/             # Per-run JSONL output files
        └── *.jsonl
```

---

## Limitations

- **Domain bias:** Training data is primarily Biblical prose. The model handles conversational input reasonably well but may produce archaic or overly formal Lun Bawang for casual text.
- **Small vocabulary:** The combined corpus covers only a fraction of Lun Bawang vocabulary. Uncommon words are often hallucinated or approximated.
- **En→LB is harder:** The model was evaluated primarily in the LB→EN direction. English→Lun Bawang output is harder to verify without a native speaker.
- **No morphological analysis:** Lun Bawang has productive affixation (nasal prefixes, infixes, reduplication). The model learns these patterns implicitly from examples rather than through explicit linguistic structure.
- **Occasional empty output (Inkling default):** ~4% of short/single-word translations came back empty in the final checkpoint's held-out eval, mitigated in production by an automatic retry at higher reasoning effort (see [Inkling-Small Experiment](#inkling-small-experiment)). Not fully eliminated — a repeat translation attempt on the same input usually succeeds if you hit it.

---

## Acknowledgements

- Lun Bawang Bible translation: Sabah Theological Seminary / Bible Society of Malaysia
- English reference: [World English Bible](https://worldenglish.bible) (public domain)
- Dictionary data: borneodictionary.com
- Phrasebook data: longsemadoh.wordpress.com
- Narrative parallel text: Mortensen, M. (2021). *The Kemaloh Lun Bawang Language of Borneo*. PhD dissertation, University of Hawai'i at Mānoa. Used for non-commercial research purposes.
- Fine-tuning infrastructure: [Tinker](https://thinkingmachines.ai/tinker/) by Thinking Machines
- Production base model: Inkling-Small by Thinking Machines
- Archived base model (v0/v1): [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by Alibaba Cloud
