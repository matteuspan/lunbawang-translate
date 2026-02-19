# LunBawang Translate

**Live demo:** https://lunbawang-translate.onrender.com/

A bidirectional **Lun Bawang ↔ English** machine translator, built by fine-tuning Qwen3-8B on the first parallel corpus ever assembled for the language. The web interface is deployed on Render and runs against a fine-tuned model served via the [Tinker](https://thinkingmachines.ai/tinker/) API.

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
  └─ Qwen3-8B + LoRA fine-tune (checkpoint-8000, best BLEU)
```

1. The user types Lun Bawang or English text in the browser.
2. The server auto-detects the source language from a vocabulary heuristic, then calls the Tinker API with a chat-format prompt.
3. The model generates a translation; `<think>…</think>` reasoning blocks (from Qwen3's chain-of-thought mode) are stripped before the response is returned.
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

**Base model:** `Qwen/Qwen3-8B`
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
```

The script resumes automatically from the last checkpoint if `tinker_state.json` exists. Progress is logged to stdout and checkpoint metadata is saved back to `tinker_state.json` after every 500 steps.

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

| Step | Epoch | Val loss | Bible BLEU | Dict exact match | Sentence BLEU |
|------|-------|----------|------------|-----------------|---------------|
| 2,000 | 1 | 0.719 | 20.49 | 20.0% | 5.62 |
| 4,000 | 1 | 0.442 | — | — | — |
| 6,000 | 1 | 0.325 | 41.94 | 19.5% | 30.67 |
| **8,000** | **1** | **0.185** | **52.38** | **20.3%** | **34.98** |
| 10,500 | 2 | 0.149 | 51.79 | 21.8% | 34.65 |
| 16,000 | 3 | 0.054 | 58.24 | 19.5% | 33.59 |

Training was stopped at step 16,000 (mid-epoch 3). Val loss continued to fall sharply in epoch 3 (0.082 at epoch 2 end → 0.054 by step 16,000), indicating rapid memorisation of the Biblical training set rather than new generalisation.

**Step 8,000 is the default checkpoint** for general use. It achieves the best sentence BLEU (34.98) and produces the most natural output for conversational input. Steps 10,500 and 16,000 show a register drift pattern — BLEU improves on Biblical prose but slightly regresses on conversational sentences as the model overfits to Biblical register. **Step 16,000 is the best checkpoint for Bible text specifically** (Bible BLEU 58.24).

### BLEU context

A BLEU score of 52 for a rare language with ~30k training sentences is strong. Google Translate achieves ~40 BLEU for well-resourced language pairs like French→English with billions of sentence pairs. For comparison, published results on similarly low-resource languages (Swahili, Welsh, Basque at small data scales) typically fall in the 20–35 range with equivalent training set sizes.

The ceiling is partly set by reference translation quality and the domain gap: training data is entirely Biblical, while the evaluation sentence set is conversational.

### Comparison with GPT models

The same val sets were run against general-purpose OpenAI models to establish a baseline for what an off-the-shelf LLM can do with no Lun Bawang-specific training. Scripts: `eval/eval_openai.py`, `eval/eval_checkpoint.py`. Raw outputs saved to `eval/eval_raw/`; combined into `eval/eval_outputs.csv` via `eval/merge_evals.py`.

| Model | Bible BLEU | Dict exact match | Sentence BLEU | Avg ms/call | Notes |
|-------|-----------|-----------------|---------------|-------------|-------|
| **Our model (checkpoint-8000)** | **51.70** | **20.3%** | **30.79** | ~5,400 | Qwen3-8B + LoRA, Tinker inference |
| gpt-4o | 10.44 | 6.0% | 21.46 | ~716 | |
| gpt-5-mini | 8.79 | 3.8% | 10.95 | ~18,300 | Reasoning model; slow despite "mini" label |
| gpt-4o-mini | 7.22 | 2.3% | 11.84 | — | |

Key takeaways:
- Our fine-tuned model scores **5× higher** on Bible BLEU and **3× higher** on Dict exact match than the best general GPT model (gpt-4o), despite being 8B parameters vs. GPT-4o's much larger scale.
- gpt-5-mini is a reasoning model and takes ~18s per API call — 25× slower than gpt-4o and 3× slower than our model on Tinker. Its BLEU scores are also worse, suggesting reasoning capability does not compensate for lack of Lun Bawang training data.
- All GPT models struggle with the dictionary exact-match task (single-word translations), confirming that Lun Bawang vocabulary is largely absent from general pre-training data.
- The checkpoint-8000 timing (~5,400ms) reflects Tinker's cold-start latency; warm-cache requests are faster.

---

## User Feedback Loop

The web UI collects thumbs up/down feedback on every translation. This data feeds back into future fine-tuning runs to improve quality on real user input — particularly conversational Lun Bawang, which the current training data (almost entirely Biblical prose) does not cover well.

### How feedback is collected

After each translation, a 👍 / 👎 widget appears below the output:

- **Thumbs up** — records the translation as a correct example
- **Thumbs down** — optionally prompts for a correction; if provided, the corrected translation is used as the training target instead of the model's output; thumbs-down with no correction is discarded (we know it's wrong but not what's right)

Each feedback entry records: source text, translation direction (LB→EN or EN→LB), checkpoint used, model output, rating, and correction (if provided). The raw IP address is stored locally only for spam/QC purposes; the `/24` prefix (e.g. `1.2.3.x`) is used in any exported data.

### Storage

- **Primary:** `feedback.db` (SQLite, WAL mode) — local, full fidelity including raw IPs
- **Mirror:** `feedback.jsonl` — append-only line-delimited JSON, same data
- **GitHub backup:** if `GITHUB_TOKEN` is set, every submission triggers an async commit of `feedback.csv` (truncated IPs) to the repo — no manual exports needed, data survives redeploys

### Reviewing and preparing training data

```bash
python3.13 eval/review_feedback.py --dry-run     # summary + QC flags, no file written
python3.13 eval/review_feedback.py               # writes feedback_corpus.csv
python3.13 eval/review_feedback.py --csv eval/feedback.csv  # read from GitHub export instead of local DB
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

`TINKER_API_KEY` is set as a Render environment variable. The committed `tinker_state.json` tells the server which checkpoints exist and where to find them on Tinker's infrastructure — no model weights are stored in the repo.

The checkpoint dropdown in the UI lets you compare any saved checkpoint, not just the current default. The step 8,000 checkpoint is pre-selected as the default because it achieves the best overall BLEU.

---

## Project Structure

```
raretranslator/
├── serve.py                  # FastAPI web server + translation API
├── train_translator.py       # Fine-tuning + evaluation script
├── tinker_state.json         # Checkpoint metadata for the current run
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
    ├── feedback.csv          # GitHub-mirrored feedback export
    ├── eval_outputs.csv      # Combined eval results across all runs
    ├── eval_results_openai.json
    └── eval_raw/             # Per-run JSONL output files
        └── *.jsonl
```

---

## Limitations

- **Domain bias:** Training data is entirely Biblical prose. The model handles conversational input reasonably well at step 8,000 but may produce archaic or overly formal Lun Bawang for casual text.
- **Small vocabulary:** The combined corpus covers only a fraction of Lun Bawang vocabulary. Uncommon words are often hallucinated or approximated.
- **En→LB is harder:** The model was evaluated primarily in the LB→EN direction. English→Lun Bawang output is harder to verify without a native speaker.
- **No morphological analysis:** Lun Bawang has productive affixation (nasal prefixes, infixes, reduplication). The model learns these patterns implicitly from examples rather than through explicit linguistic structure.

---

## Acknowledgements

- Lun Bawang Bible translation: Sabah Theological Seminary / Bible Society of Malaysia
- English reference: [World English Bible](https://worldenglish.bible) (public domain)
- Dictionary data: borneodictionary.com
- Phrasebook data: longsemadoh.wordpress.com
- Narrative parallel text: Mortensen, M. (2021). *The Kemaloh Lun Bawang Language of Borneo*. PhD dissertation, University of Hawai'i at Mānoa. Used for non-commercial research purposes.
- Fine-tuning infrastructure: [Tinker](https://thinkingmachines.ai/tinker/) by Thinking Machines
- Base model: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) by Alibaba Cloud
