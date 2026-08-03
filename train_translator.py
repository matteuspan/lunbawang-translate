"""
Fine-tune a bidirectional Lun Bawang ↔ English translator using the Tinker API.

Training data:
  - parallel_corpus.csv   (Bible verse pairs, ~30k rows)      → 90/10 train/val by book
  - aux_corpus.csv        (dictionary words + sentences)       → 80/20 train/val by source
  - feedback_corpus.csv   (user corrections + thumbs-up data) → 80/20 train/val by source
  Aux train examples are repeated 5×, feedback examples 10×, to compensate for small size.

Both directions (LB→EN and EN→LB) are generated for every pair.

Validation metrics:
  - val_loss  (cross-entropy on 200 random Bible val examples)  every SAVE_EVERY steps
  - val_bleu  (sacrebleu on Bible val + sentence val) + exact-match (dict val)
               every VAL_BLEU_EVERY steps (requires sampling checkpoint)

Usage:
  python3.13 train_translator.py --train
  python3.13 train_translator.py --translate --text "Kareb Allah nengimun mudut tana'"
  python3.13 train_translator.py --translate --direction en2lb --text "In the beginning"
  python3.13 train_translator.py --translate   # interactive mode

Requires:
  TINKER_API_KEY env var
  parallel_corpus.csv  (from build_parallel_corpus.py)
  aux_corpus.csv       (from build_aux_corpus.py — optional but recommended)

  For --base-model thinkingmachines/* (Inkling family) only: `pip install
  tml-renderers torch`. These models use a native TML chat format that's
  incompatible with the generic HF chat template — tml-renderers renders
  training/inference prompts the same way Tinker's serving endpoint does
  server-side. Not required for Qwen/other HF-tokenizer models.
"""

import os
import re
import csv
import json
import random
import argparse
import threading
import numpy as np
from pathlib import Path

import tinker
from tinker import (
    ServiceClient, AdamParams,
    Datum, ModelInput, SamplingParams,
)

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY          = os.environ["TINKER_API_KEY"]
BASE_MODEL       = "Qwen/Qwen3-8B"
LORA_RANK        = 16
LEARNING_RATE    = 5e-5
BATCH_SIZE       = 8
MAX_TOKENS       = 384
EPOCHS           = 3
SAVE_EVERY       = 500      # save checkpoint + compute val_loss every N steps
VAL_BLEU_EVERY   = 2000     # additionally compute BLEU + exact-match every N steps
AUX_REPEAT       = 5        # repeat aux training datums N× relative to Bible datums
VAL_LOSS_SAMPLES = 200      # number of Bible val datums for val_loss
VAL_BLEU_BIBLE   = 50       # max Bible val pairs sampled for BLEU (lb→en only)
VAL_BLEU_SENT    = 100      # max sentence val pairs sampled for BLEU — sent_val_pairs
                             # has grown to 1000+ (mostly T4T conversational data that
                             # postdates the v1 Qwen run) so evaluating all of it is both
                             # slow and not a like-for-like comparison population
VAL_BLEU_WORKERS = 10       # concurrent sampling calls — sequential was the dominant cost

STATE_FILE      = Path(__file__).parent / "tinker_state.json"
CORPUS_FILE     = Path(__file__).parent / "corpus" / "parallel_corpus.csv"
AUX_FILE        = Path(__file__).parent / "corpus" / "aux_corpus.csv"
FEEDBACK_FILE   = Path(__file__).parent / "feedback_corpus.csv"
CONV_FILE       = Path(__file__).parent / "corpus" / "conversational_corpus.csv"
FEEDBACK_REPEAT = 10   # more aggressive than AUX_REPEAT — fewer entries, higher signal
CONV_REPEAT     = 3    # T4T conversational dialogue verses; 3× weighting

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate ONLY the exact text provided — output just the translation, nothing else. "
    "Do not add Bible verse titles, context, or anything not present in the input."
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_bible_corpus(path=CORPUS_FILE):
    """
    Load Bible verse pairs with book_code for stratified splitting.
    Returns list of (lun_bawang, english, book_code).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lb   = row["lun_bawang"].strip()
            eng  = row["english"].strip()
            book = row.get("book_code", "UNK").strip()
            if lb and eng:
                rows.append((lb, eng, book))
    return rows


def load_aux_corpus(path=AUX_FILE):
    """
    Load auxiliary corpus (dictionary words + conversational sentences).
    Returns list of (lun_bawang, english, source, type_).
    Returns [] if file does not exist.
    """
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lb     = row["lun_bawang"].strip()
            eng    = row["english"].strip()
            source = row.get("source", "aux").strip()
            type_  = row.get("type", "word").strip()
            if lb and eng:
                rows.append((lb, eng, source, type_))
    return rows


def load_feedback_corpus(path=FEEDBACK_FILE):
    """Load user feedback corpus. Same format as aux_corpus.csv.
    Returns [] if file doesn't exist."""
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lb     = row["lun_bawang"].strip()
            eng    = row["english"].strip()
            source = row.get("source", "user_feedback").strip()
            type_  = row.get("type", "word").strip()
            if lb and eng:
                rows.append((lb, eng, source, type_))
    return rows


# ── Train / val splits ────────────────────────────────────────────────────────

def bible_train_val_split(corpus, val_fraction=0.1, seed=42):
    """
    Stratified 90/10 split of Bible verses by book_code.
    Returns (train_list, val_list), each element (lb, eng, book).
    """
    rng = random.Random(seed)
    by_book: dict[str, list] = {}
    for item in corpus:
        by_book.setdefault(item[2], []).append(item)

    train, val = [], []
    for book, items in sorted(by_book.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def aux_train_val_split(corpus, val_fraction=0.2, seed=42):
    """
    Random 80/20 split per source so each source appears in both sets.
    Returns (train_list, val_list), each element (lb, eng, source, type_).
    """
    rng = random.Random(seed)
    by_source: dict[str, list] = {}
    for item in corpus:
        by_source.setdefault(item[2], []).append(item)

    train, val = [], []
    for source, items in sorted(by_source.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


# ── Tokenisation + Datum construction ────────────────────────────────────────

def _uses_tml_renderer(model_name: str) -> bool:
    """thinkingmachines/* models (Inkling family) use a native TML chat format
    (<|message_system|>, <|content_text|>, ... — see tml_renderers) that is
    completely different from the generic HF chat template. Tinker's serving
    endpoint renders with TML server-side, so training must match it exactly
    or the fine-tuned model won't respond through that endpoint at all."""
    return model_name.split(":")[0].startswith(("thinkingmachines/", "TML/"))


_tml_renderer = None
_tml_lock = threading.Lock()  # guards the renderer singleton + its (uncertain
                               # thread-safety) native calls; the network .sample()
                               # call itself stays outside the lock for concurrency

def _get_tml_renderer():
    global _tml_renderer
    with _tml_lock:
        if _tml_renderer is None:
            from tml_renderers.v0 import Renderer
            from tml_renderers.tokenizers import o200k_base_chat
            _tml_renderer = Renderer(o200k_base_chat())
        return _tml_renderer


def _make_datum_tml(user_content, assistant_content, max_tokens):
    from tml_renderers.chat import MessageList
    from tml_renderers.tinker import training_example_to_tinker_model_input_and_weights

    renderer = _get_tml_renderer()
    messages = MessageList.from_oss_messages([
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ])
    examples = renderer.render_for_sft(messages)
    if not examples:
        return None

    model_input, weights_unshifted = training_example_to_tinker_model_input_and_weights(examples[-1])
    full_tokens = [t for chunk in model_input.chunks for t in chunk.tokens]

    if len(full_tokens) > max_tokens or len(full_tokens) != len(weights_unshifted):
        return None

    # Same next-token shift as the HF path: weights_unshifted[i] says whether
    # predicting token[i] counts toward loss, so after shifting, the weight
    # for target_tokens[i] (= full_tokens[i+1]) is weights_unshifted[i+1].
    input_tokens  = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights       = weights_unshifted[1:]

    return Datum(
        model_input=ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def _translate_one_tml(sampling_client, user_content, params, retries=2):
    """Sample one completion for a thinkingmachines/* model, rendering the
    prompt with the native TML format (matching what Tinker's serving
    endpoint applies server-side) and parsing the response back into
    channel-tagged messages instead of regex-stripping <think> tags (Inkling
    uses an Analysis/Final channel split, not literal <think> tags).

    Empty results are retried: a transient network/server hiccup can return
    a truncated sample without raising an exception, indistinguishable from
    a genuine empty completion unless retried."""
    from tml_renderers.chat import MessageList, MessageChannel, Text
    from tml_renderers.tinker import token_spans_to_tinker_model_input

    renderer = _get_tml_renderer()
    messages = MessageList.from_oss_messages([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ])

    for attempt in range(retries + 1):
        # Native (Rust) renderer/parser calls are of uncertain thread-safety, so
        # they're serialized; the slow network .sample() call stays unlocked so
        # concurrent callers still overlap on the part that actually dominates cost.
        with _tml_lock:
            # effort=0.0 caused frequent single-token empty completions on
            # short phrases in testing (checkpoint-8000, 15/15 reproduced);
            # 0.6 was clean (0/15) on the same native-SDK path.
            spans, parser = renderer.render_for_completion_with_effort(messages, 0.6)
        prompt = token_spans_to_tinker_model_input(spans)
        result = sampling_client.sample(prompt, num_samples=1, sampling_params=params).result()
        with _tml_lock:
            parsed = parser.parse_tokens(list(result.sequences[0].tokens))
        for msg in parsed:
            if msg.channel_enum == MessageChannel.Analysis:
                continue
            if isinstance(msg.content, Text):
                text = msg.content.text.strip()
                if text or attempt == retries:
                    return text
                break  # empty — fall through to retry
    return ""


def get_tokenizer(client, model_name):
    """Tinker's built-in get_tokenizer() needs the private `tml_tokenizers`
    package for thinkingmachines/* models. Fall back to loading the HF
    tokenizer directly — Inkling's tokenizer files are public, and its
    apply_chat_template output shape is already handled by _token_ids()."""
    try:
        return client.get_tokenizer()
    except ModuleNotFoundError:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name.split(":")[0])


def _token_ids(result):
    """apply_chat_template may return a list or a BatchEncoding."""
    if isinstance(result, list):
        return result
    if hasattr(result, 'input_ids'):
        return list(result.input_ids)
    if isinstance(result, dict):
        return list(result['input_ids'])
    return list(result)


def make_datum(tokenizer, source_text, target_text, direction="lb2en", max_tokens=MAX_TOKENS):
    """
    Build a Datum for next-token-prediction with cross-entropy loss.

    The correct Tinker format (from their cookbook):
      input_tokens  = full_tokens[:-1]   (all but last)
      target_tokens = full_tokens[1:]    (all but first — next-token targets)
      weights       = 0 for prompt positions, 1 for completion positions

    direction: "lb2en" or "en2lb"
    """
    if direction == "lb2en":
        user_content = f"Translate to English:\n{source_text}"
        assistant_content = target_text
    else:
        user_content = f"Translate to Lun Bawang:\n{source_text}"
        assistant_content = target_text

    if _uses_tml_renderer(BASE_MODEL):
        return _make_datum_tml(user_content, assistant_content, max_tokens)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    prompt_messages = messages[:-1]

    full_tokens = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    ))
    prompt_tokens = _token_ids(tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True,
    ))

    if len(full_tokens) > max_tokens:
        return None

    n_prompt = len(prompt_tokens)
    n_full   = len(full_tokens)

    # Shift for next-token prediction
    input_tokens  = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    # Weights: 0 on prompt positions, 1 on completion positions (shifted by 1)
    weights = [0.0] * (n_prompt - 1) + [1.0] * (n_full - n_prompt)

    assert len(input_tokens) == len(target_tokens) == len(weights)

    return Datum(
        model_input=ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
        ),
    )


def make_datums_bidirectional(tokenizer, pairs_with_meta, is_bible=True):
    """
    Build bidirectional (lb→en + en→lb) datums from a list of pairs.
    pairs_with_meta: (lb, eng, *extra)
    Returns (datums, n_skipped).
    """
    datums, skipped = [], 0
    for lb, eng, *_ in pairs_with_meta:
        for d in [
            make_datum(tokenizer, lb, eng, "lb2en"),
            make_datum(tokenizer, eng, lb, "en2lb"),
        ]:
            if d is None:
                skipped += 1
            else:
                datums.append(d)
    return datums, skipped


# ── Validation ────────────────────────────────────────────────────────────────

def compute_val_loss(tc, val_datums, n=VAL_LOSS_SAMPLES, seed=42):
    """
    Estimate val loss (cross-entropy) on up to n random val datums.
    Uses forward_backward without an optim_step — gradients are reset
    by the next training forward_backward call.
    Returns average loss per weighted token.
    """
    rng = random.Random(seed)
    sample = rng.sample(val_datums, min(n, len(val_datums)))
    total_loss, total_weighted = 0.0, 0

    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample[i:i + BATCH_SIZE]
        result = tc.forward_backward(batch, "cross_entropy").result()
        total_loss += result.metrics.get("loss:sum", 0.0)
        total_weighted += sum(
            sum(1 for x in out["elementwise_loss"].data if x != 0.0)
            for out in result.loss_fn_outputs
        )

    return total_loss / max(total_weighted, 1)


def compute_val_bleu(service, checkpoint_path, tokenizer,
                     bible_val_pairs, dict_val_pairs, sent_val_pairs):
    """
    Compute validation BLEU (sacrebleu) and dictionary exact-match.

    bible_val_pairs: list of (lb, eng, book)
    dict_val_pairs:  list of (lb, eng, source, type_) where type_=="word"
    sent_val_pairs:  list of (lb, eng, source, type_) where type_=="sentence"

    Returns (bible_bleu, dict_exact_pct, sent_bleu, sent_bleu_short, sent_bleu_long)
      None values if a subset is empty or sacrebleu is unavailable.
    """
    try:
        import sacrebleu as sb
    except ImportError:
        print("  [warn] sacrebleu not installed — skipping BLEU. Run: pip install sacrebleu")
        return None, None, None

    print("  Computing val BLEU (creating sampling client)…")
    sc = service.create_sampling_client(checkpoint_path)
    sc_tokenizer = get_tokenizer(sc, BASE_MODEL)
    params = SamplingParams(max_tokens=128, temperature=0.1, top_p=0.9)

    # ── Crash-safe resume cache ──
    # A container/process restart mid-eval used to lose every translation done
    # so far, forcing a full re-run. Cache each (eval_set, idx) result to CSV
    # as it's produced, flushed immediately, and skip anything already cached.
    cache_dir = Path(__file__).parent / "eval" / "val_bleu_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ck_slug = re.sub(r"[^A-Za-z0-9_\-]", "_", checkpoint_path)
    cache_path = cache_dir / f"{ck_slug}.csv"

    cached: dict[tuple[str, int], str] = {}
    if cache_path.exists():
        with open(cache_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cached[(row["eval_set"], int(row["idx"]))] = row["hyp"]
        if cached:
            print(f"  Resuming from cache: {len(cached)} translations already done → {cache_path.name}")

    cache_is_new = not cache_path.exists()
    cache_file = open(cache_path, "a", newline="", encoding="utf-8")
    cache_writer = csv.writer(cache_file)
    if cache_is_new:
        cache_writer.writerow(["eval_set", "idx", "direction", "input", "hyp", "ref"])
        cache_file.flush()

    def _translate_one(src: str, direction: str, retries: int = 2) -> str:
        user_content = (
            f"Translate to English:\n{src}" if direction == "lb2en"
            else f"Translate to Lun Bawang:\n{src}"
        )
        if _uses_tml_renderer(BASE_MODEL):
            return _translate_one_tml(sc, user_content, params, retries=retries)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        prompt_tokens = _token_ids(sc_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        ))
        for attempt in range(retries + 1):
            result = sc.sample(
                ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=1,
                sampling_params=params,
            ).result()
            raw = sc_tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if text or attempt == retries:
                return text
        return ""

    def _batch_translate(pairs, direction="lb2en", eval_set="eval"):
        """Return (hypotheses, references) lists. Skips pairs already cached
        from a prior (crashed) run of this same checkpoint's eval. Translates
        concurrently — sequential calls were the dominant cost of this eval."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        src_idx = 0 if direction == "lb2en" else 1
        ref_idx = 1 if direction == "lb2en" else 0

        hyps = [None] * len(pairs)
        refs = [None] * len(pairs)
        to_fetch = []
        for idx, pair in enumerate(pairs):
            src, ref = pair[src_idx], pair[ref_idx]
            refs[idx] = ref
            key = (eval_set, idx)
            if key in cached:
                hyps[idx] = cached[key]
            else:
                to_fetch.append((idx, src))

        def _do(idx, src):
            try:
                return idx, src, _translate_one(src, direction)
            except Exception as e:
                print(f"  [warn] translate failed: {e}")
                return idx, src, ""

        if to_fetch:
            with ThreadPoolExecutor(max_workers=VAL_BLEU_WORKERS) as ex:
                futures = [ex.submit(_do, idx, src) for idx, src in to_fetch]
                for fut in as_completed(futures):
                    idx, src, hyp = fut.result()
                    hyps[idx] = hyp
                    cached[(eval_set, idx)] = hyp
                    cache_writer.writerow([eval_set, idx, direction, src, hyp, refs[idx]])
                    cache_file.flush()

        return hyps, refs

    # ── Bible val BLEU (lb→en, up to VAL_BLEU_BIBLE examples) ──
    bible_bleu = None
    if bible_val_pairs:
        rng = random.Random(42)
        sample = rng.sample(bible_val_pairs, min(VAL_BLEU_BIBLE, len(bible_val_pairs)))
        print(f"  Translating {len(sample)} Bible val examples (lb→en)…")
        hyps, refs = _batch_translate(sample, "lb2en", "bible")
        bible_bleu = sb.corpus_bleu(hyps, [refs]).score
        print(f"  Bible val BLEU: {bible_bleu:.2f}")

    # ── Dict val exact match (lb→en) ──
    dict_exact_pct = None
    if dict_val_pairs:
        print(f"  Translating {len(dict_val_pairs)} dictionary val examples (lb→en)…")
        hyps, refs = _batch_translate(dict_val_pairs, "lb2en", "dict")
        n_correct = sum(
            h.lower().strip() == r.lower().strip()
            for h, r in zip(hyps, refs)
        )
        dict_exact_pct = n_correct / len(dict_val_pairs) * 100
        print(f"  Dict val exact match: {dict_exact_pct:.1f}% ({n_correct}/{len(dict_val_pairs)})")

    # ── Sentence val BLEU (lb→en, up to VAL_BLEU_SENT examples) ──
    sent_bleu = sent_bleu_short = sent_bleu_long = None
    if sent_val_pairs:
        rng_sent = random.Random(42)
        sent_sample = rng_sent.sample(sent_val_pairs, min(VAL_BLEU_SENT, len(sent_val_pairs)))
        print(f"  Translating {len(sent_sample)} sentence val examples (lb→en)…")
        hyps, refs = _batch_translate(sent_sample, "lb2en", "sentence")
        sent_bleu = sb.corpus_bleu(hyps, [refs]).score
        short_pairs = [(h, r) for h, r in zip(hyps, refs) if len(r.split()) <= 10]
        long_pairs  = [(h, r) for h, r in zip(hyps, refs) if len(r.split()) >  10]
        sent_bleu_short = sb.corpus_bleu([p[0] for p in short_pairs], [[p[1] for p in short_pairs]]).score if short_pairs else 0.0
        sent_bleu_long  = sb.corpus_bleu([p[0] for p in long_pairs],  [[p[1] for p in long_pairs]]).score  if long_pairs  else 0.0
        print(f"  Sentence val BLEU: {sent_bleu:.2f}  (short ≤10: {sent_bleu_short:.2f} | long >10: {sent_bleu_long:.2f})")

    cache_file.close()
    return bible_bleu, dict_exact_pct, sent_bleu, sent_bleu_short, sent_bleu_long


# ── State persistence ─────────────────────────────────────────────────────────

def save_state(state: dict):
    import fcntl
    if not STATE_FILE.exists():
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    else:
        with open(STATE_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                on_disk = json.load(f)
                disk_by_step = {ck["step"]: ck for ck in on_disk.get("checkpoints", [])}
                for ck in state["checkpoints"]:
                    if ck["step"] in disk_by_step:
                        for k, v in disk_by_step[ck["step"]].items():
                            if k not in ck:
                                ck[k] = v
            except (json.JSONDecodeError, KeyError):
                pass
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()
    print(f"State saved → {STATE_FILE}")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


# ── Training ──────────────────────────────────────────────────────────────────

def train(new_run: bool = False, max_steps: int | None = None):
    print(f"Model:      {BASE_MODEL}")
    print(f"LoRA rank:  {LORA_RANK}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs:     {EPOCHS}")
    print(f"Direction:  bidirectional (LB↔EN)")
    if max_steps:
        print(f"Max steps:  {max_steps} (stops early for step-matched comparison)")
    print()

    # ── Load corpora ──
    print("Loading Bible corpus…")
    bible_corpus = load_bible_corpus()
    print(f"  {len(bible_corpus)} verse pairs")

    print("Loading auxiliary corpus…")
    aux_corpus = load_aux_corpus()
    if aux_corpus:
        n_words = sum(1 for r in aux_corpus if r[3] == "word")
        n_sents = sum(1 for r in aux_corpus if r[3] == "sentence")
        print(f"  {len(aux_corpus)} aux entries ({n_words} words, {n_sents} sentences)")
    else:
        print("  aux_corpus.csv not found — run build_aux_corpus.py to add word/sentence data")

    print("Loading feedback corpus…")
    feedback_corpus = load_feedback_corpus()
    if feedback_corpus:
        n_words = sum(1 for r in feedback_corpus if r[3] == "word")
        n_sents = sum(1 for r in feedback_corpus if r[3] == "sentence")
        print(f"  {len(feedback_corpus)} feedback entries ({n_words} words, {n_sents} sentences)")
    else:
        print("  feedback_corpus.csv not found — skipping (run review_feedback.py to generate)")

    print("Loading conversational corpus (T4T dialogue verses)…")
    conv_corpus = load_aux_corpus(CONV_FILE) if CONV_FILE.exists() else []
    if conv_corpus:
        print(f"  {len(conv_corpus)} conversational sentence pairs")
    else:
        print("  conversational_corpus.csv not found — run corpus/build_conversational_corpus.py")

    # ── Splits ──
    print("\nSplitting data…")
    bible_train, bible_val = bible_train_val_split(bible_corpus, val_fraction=0.1)
    print(f"  Bible:  {len(bible_train)} train / {len(bible_val)} val (stratified by book)")

    aux_train, aux_val = aux_train_val_split(aux_corpus, val_fraction=0.2)
    aux_dict_val  = [r for r in aux_val if r[3] == "word"]
    aux_sent_val  = [r for r in aux_val if r[3] == "sentence"]
    if aux_corpus:
        print(f"  Aux:    {len(aux_train)} train / {len(aux_val)} val ({len(aux_dict_val)} words, {len(aux_sent_val)} sentences in val)")

    feedback_train, feedback_val = (
        aux_train_val_split(feedback_corpus, val_fraction=0.2)
        if feedback_corpus else ([], [])
    )
    aux_dict_val += [r for r in feedback_val if r[3] == "word"]
    aux_sent_val += [r for r in feedback_val if r[3] == "sentence"]
    if feedback_corpus:
        print(f"  Feedback: {len(feedback_train)} train / {len(feedback_val)} val")

    conv_train, conv_val = (
        aux_train_val_split(conv_corpus, val_fraction=0.2)
        if conv_corpus else ([], [])
    )
    aux_sent_val += conv_val
    if conv_corpus:
        print(f"  Conv:     {len(conv_train)} train / {len(conv_val)} val")

    # ── Connect to Tinker ──
    service = ServiceClient()
    state = load_state()

    if state and state.get("training_state_path"):
        # Full resume: restores weights + optimizer state (Adam momentum, LR schedule)
        print(f"\nResuming from training state: {state['training_state_path']}")
        tc = service.create_training_client_from_state_with_optimizer(state["training_state_path"])
    elif state and state.get("warm_start_path"):
        # Warm start: load learned weights from a sampler checkpoint, reset optimizer
        print(f"\nWarm-starting from checkpoint: {state['warm_start_path']}")
        tc = service.create_training_client_from_state(state["warm_start_path"])
        info = tc.get_info()
        state["model_id"] = info.model_id
        del state["warm_start_path"]
        save_state(state)
        print(f"  New experiment ID: {info.model_id}")
    else:
        # Guard: refuse to silently overwrite an existing run
        if not new_run and state and state.get("checkpoints"):
            print("\nERROR: tinker_state.json has existing checkpoints but no resume path.")
            print("  This means the previous training session ended without saving a training state.")
            print("  To start a completely new experiment, run with --new-run.")
            print("  To warm-start from a sampler checkpoint, add 'warm_start_path' to tinker_state.json.")
            raise SystemExit(1)
        if new_run and state and state.get("checkpoints"):
            print(f"\nStarting new experiment (--new-run; previous run had {state['steps']} steps)…")
        else:
            print("\nStarting new training experiment…")
        tc = service.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=LORA_RANK,
            seed=42,
        )
        info = tc.get_info()
        state = {"model_id": info.model_id, "checkpoints": [], "steps": 0}
        save_state(state)
        print(f"  Experiment ID: {info.model_id}")

    print("\nFetching tokenizer…")
    tokenizer = get_tokenizer(tc, BASE_MODEL)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ── Tokenise training data ──
    print("\nTokenising Bible train datums (both directions)…")
    bible_train_datums, skip1 = make_datums_bidirectional(tokenizer, bible_train)
    print(f"  {len(bible_train_datums)} datums ({skip1} skipped as too long)")

    print("Tokenising aux train datums (both directions)…")
    aux_train_datums, skip2 = make_datums_bidirectional(tokenizer, aux_train)
    print(f"  {len(aux_train_datums)} datums ({skip2} skipped)")

    # Repeat aux datums AUX_REPEAT times to compensate for small size
    aux_train_datums_repeated = aux_train_datums * AUX_REPEAT
    if aux_train_datums:
        print(f"  Aux train datums repeated {AUX_REPEAT}× → {len(aux_train_datums_repeated)} datums")

    print("Tokenising feedback train datums (both directions)…")
    feedback_train_datums, skip3 = make_datums_bidirectional(tokenizer, feedback_train)
    feedback_train_datums_repeated = feedback_train_datums * FEEDBACK_REPEAT
    if feedback_train_datums:
        print(f"  {len(feedback_train_datums)} datums ({skip3} skipped)")
        print(f"  Feedback train datums repeated {FEEDBACK_REPEAT}× → {len(feedback_train_datums_repeated)} datums")
    else:
        print("  No feedback datums")

    print("Tokenising conversational train datums (both directions)…")
    conv_train_datums, skip4 = make_datums_bidirectional(tokenizer, conv_train)
    conv_train_datums_repeated = conv_train_datums * CONV_REPEAT
    if conv_train_datums:
        print(f"  {len(conv_train_datums)} datums ({skip4} skipped)")
        print(f"  Conv train datums repeated {CONV_REPEAT}× → {len(conv_train_datums_repeated)} datums")
    else:
        print("  No conversational datums")

    all_train_datums = (bible_train_datums + aux_train_datums_repeated
                        + feedback_train_datums_repeated + conv_train_datums_repeated)
    print(f"\n  Total training datums: {len(all_train_datums)}")

    # ── Tokenise validation data ──
    print("\nTokenising Bible val datums (for val_loss)…")
    bible_val_datums, _ = make_datums_bidirectional(tokenizer, bible_val)
    print(f"  {len(bible_val_datums)} Bible val datums")

    # ── Training loop ──
    global_step = state.get("steps", 0)
    print(f"\nStarting training from step {global_step}…")

    for epoch in range(EPOCHS):
        random.shuffle(all_train_datums)
        batches = [
            all_train_datums[i:i + BATCH_SIZE]
            for i in range(0, len(all_train_datums), BATCH_SIZE)
        ]
        print(f"\n── Epoch {epoch+1}/{EPOCHS}  ({len(batches)} batches) ──")

        for step, batch in enumerate(batches):
            fwdbwd_future = tc.forward_backward(batch, "cross_entropy")
            optim_future  = tc.optim_step(AdamParams(learning_rate=LEARNING_RATE))

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            total_loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
            n_weighted = sum(
                sum(1 for x in out["elementwise_loss"].data if x != 0.0)
                for out in fwdbwd_result.loss_fn_outputs
            )
            train_loss = total_loss / max(n_weighted, 1)

            global_step += 1
            print(
                f"  step {global_step:5d} | train_loss {train_loss:.4f} "
                f"| epoch {epoch+1} batch {step+1}/{len(batches)}"
            )

            if global_step % SAVE_EVERY == 0 or global_step == max_steps:
                # ── Save sampler weights (for inference) ──
                print(f"  → Saving checkpoint at step {global_step}…")
                result = tc.save_weights_for_sampler(f"checkpoint-{global_step}").result()
                ckpt = result.path
                state["checkpoints"].append({"step": global_step, "path": ckpt})
                state["steps"] = global_step

                # ── Save full training state (weights + optimizer) for crash recovery ──
                training_state = tc.save_state(f"checkpoint-{global_step}").result()
                state["training_state_path"] = training_state.path

                # ── Val loss (forward-only, no optim) ──
                if bible_val_datums:
                    val_loss = compute_val_loss(tc, bible_val_datums)
                    print(f"  val_loss: {val_loss:.4f}  (train_loss: {train_loss:.4f})")
                    state["checkpoints"][-1]["val_loss"] = round(val_loss, 6)

                save_state(state)
                print(f"    Checkpoint: {ckpt}")

                # ── Val BLEU every VAL_BLEU_EVERY steps ──
                if global_step % VAL_BLEU_EVERY == 0:
                    bible_bleu, dict_exact, sent_bleu, sent_short, sent_long = compute_val_bleu(
                        service, ckpt, tokenizer,
                        bible_val_pairs=bible_val,
                        dict_val_pairs=aux_dict_val,
                        sent_val_pairs=aux_sent_val,
                    )
                    metrics = {}
                    if bible_bleu is not None:
                        metrics["val_bible_bleu"] = round(bible_bleu, 2)
                    if dict_exact is not None:
                        metrics["val_dict_exact_pct"] = round(dict_exact, 1)
                    if sent_bleu is not None:
                        metrics["val_sent_bleu"]       = round(sent_bleu, 2)
                        metrics["val_sent_bleu_short"] = round(sent_short, 2)
                        metrics["val_sent_bleu_long"]  = round(sent_long, 2)
                    if metrics:
                        state["checkpoints"][-1].update(metrics)
                        save_state(state)

                    # ── Launch full bidirectional eval in background ──
                    import subprocess, sys, os
                    eval_script = Path(__file__).parent / "eval" / "eval_checkpoint.py"
                    api_key = os.environ.get("TINKER_API_KEY", "")
                    if eval_script.exists() and api_key:
                        env = {**os.environ, "TINKER_API_KEY": api_key, "PYTHONUNBUFFERED": "1"}
                        eval_cmd = [sys.executable, str(eval_script), ckpt, "--base-model", BASE_MODEL]
                        if STATE_FILE.name != "tinker_state.json":
                            # Separate experiment (e.g. alternate base model) — keep its
                            # results out of the default v0/v1 buckets and state file.
                            label = STATE_FILE.stem.removeprefix("tinker_state_")
                            eval_cmd += ["--state-file", str(STATE_FILE), "--label", label]
                        subprocess.Popen(
                            eval_cmd,
                            env=env,
                            stdout=open(Path(__file__).parent / f"eval_run_step{global_step}.log", "w"),
                            stderr=subprocess.STDOUT,
                        )
                        print(f"    Full bidirectional eval launched → eval_run_step{global_step}.log")

            if max_steps and global_step >= max_steps:
                print(f"\n✓ Reached max_steps={max_steps} — stopping early for step-matched comparison.")
                print(f"  Final checkpoint: {state['checkpoints'][-1]['path']}")
                return

        # ── End-of-epoch checkpoint ──
        print(f"\nEpoch {epoch+1} done — saving checkpoint…")
        result = tc.save_weights_for_sampler(f"checkpoint-epoch{epoch+1}").result()
        ckpt = result.path
        state["checkpoints"].append({
            "step": global_step,
            "path": ckpt,
            "epoch": epoch + 1,
        })
        state["steps"] = global_step

        # Save full training state at epoch boundary too
        training_state = tc.save_state(f"checkpoint-epoch{epoch+1}").result()
        state["training_state_path"] = training_state.path

        # Val metrics at epoch boundary
        if bible_val_datums:
            val_loss = compute_val_loss(tc, bible_val_datums)
            print(f"  Epoch {epoch+1} val_loss: {val_loss:.4f}")
            state["checkpoints"][-1]["val_loss"] = round(val_loss, 6)

        save_state(state)
        print(f"  Checkpoint: {ckpt}")

    print("\n✓ Training complete.")
    print(f"  Final checkpoint: {state['checkpoints'][-1]['path']}")


# ── Inference ─────────────────────────────────────────────────────────────────

def translate(text: str, direction: str = "lb2en", checkpoint_path: str = None):
    if checkpoint_path is None:
        state = load_state()
        if not state or not state.get("checkpoints"):
            raise RuntimeError("No checkpoints found. Run --train first.")
        checkpoint_path = state["checkpoints"][-1]["path"]
        print(f"Using checkpoint: {checkpoint_path}")

    service = ServiceClient()
    sc = service.create_sampling_client(checkpoint_path)

    if direction == "lb2en":
        user_content = f"Translate to English:\n{text}"
    else:
        user_content = f"Translate to Lun Bawang:\n{text}"

    params = SamplingParams(max_tokens=256, temperature=0.1, top_p=0.9)

    if _uses_tml_renderer(BASE_MODEL):
        return _translate_one_tml(sc, user_content, params)

    tokenizer = get_tokenizer(sc, BASE_MODEL)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt_tokens = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    ))

    prompt = ModelInput.from_ints(tokens=prompt_tokens)
    result = sc.sample(prompt, num_samples=1, sampling_params=params).result()
    output_tokens = result.sequences[0].tokens
    return tokenizer.decode(output_tokens, skip_special_tokens=True).strip()


def interactive_translate(direction="lb2en"):
    state = load_state()
    if not state or not state.get("checkpoints"):
        print("No checkpoints found. Run --train first.")
        return

    latest = state["checkpoints"][-1]
    arrow = "LB→EN" if direction == "lb2en" else "EN→LB"
    print(f"Checkpoint: step {latest['step']} — {latest['path']}")
    print(f"Mode: {arrow}  (type 'quit' to exit, 'flip' to switch direction)\n")

    while True:
        src = "LB" if direction == "lb2en" else "EN"
        tgt = "EN" if direction == "lb2en" else "LB"
        text = input(f"{src}> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "flip":
            direction = "en2lb" if direction == "lb2en" else "lb2en"
            arrow = "LB→EN" if direction == "lb2en" else "EN→LB"
            print(f"Switched to {arrow}\n")
            continue
        if not text:
            continue
        try:
            result = translate(text, direction, latest["path"])
            print(f"{tgt}> {result}\n")
        except Exception as e:
            print(f"Error: {e}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lun Bawang ↔ English translator (Tinker)")
    parser.add_argument("--train",     action="store_true", help="Run fine-tuning")
    parser.add_argument("--new-run",   action="store_true", help="Start a fresh experiment (ignores existing checkpoints)")
    parser.add_argument("--translate", action="store_true", help="Interactive translation")
    parser.add_argument("--direction", choices=["lb2en", "en2lb"], default="lb2en",
                        help="Translation direction (default: lb2en)")
    parser.add_argument("--text",      type=str, help="Single text to translate")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint path")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL,
                        help=f"Tinker base model to fine-tune (default: {BASE_MODEL})")
    parser.add_argument("--state-file", type=str, default=None,
                        help="Override tinker_state.json path — use a separate file per "
                             "base model (e.g. tinker_state_inkling.json) so parallel "
                             "experiments don't clobber each other's checkpoints")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK,
                        help=f"LoRA rank (default: {LORA_RANK})")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop after this many global steps (e.g. to match another "
                             "run's step count for a fair comparison)")
    args = parser.parse_args()

    BASE_MODEL = args.base_model
    LORA_RANK  = args.lora_rank
    if args.state_file:
        STATE_FILE = Path(args.state_file)

    if args.train:
        train(new_run=args.new_run, max_steps=args.max_steps)
    elif args.translate:
        if args.text:
            result = translate(args.text, args.direction, args.checkpoint)
            arrow = "EN" if args.direction == "lb2en" else "LB"
            print(f"{arrow}: {result}")
        else:
            interactive_translate(args.direction)
    else:
        parser.print_help()
