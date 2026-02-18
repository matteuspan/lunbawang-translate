"""
Fine-tune a bidirectional Lun Bawang ↔ English translator using the Tinker API.

Training data includes both directions per verse pair:
  LB → EN:  "Translate to English: <lb>"  → "<english>"
  EN → LB:  "Translate to Lun Bawang: <en>" → "<lb>"

Usage:
  python3.13 train_translator.py --train
  python3.13 train_translator.py --translate --text "Kareb Allah nengimun mudut tana'"
  python3.13 train_translator.py --translate --direction en2lb --text "In the beginning"
  python3.13 train_translator.py --translate   # interactive mode

Requires:
  TINKER_API_KEY env var (or hardcoded below)
  parallel_corpus.csv  (from build_parallel_corpus.py)
"""

import os
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path

import tinker
from tinker import (
    ServiceClient, AdamParams,
    Datum, ModelInput, SamplingParams,
)

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY       = os.environ["TINKER_API_KEY"]
BASE_MODEL    = "Qwen/Qwen3-8B"
LORA_RANK     = 16
LEARNING_RATE = 5e-5
BATCH_SIZE    = 8
MAX_TOKENS    = 384
EPOCHS        = 3
SAVE_EVERY    = 500     # save a sampling checkpoint every N steps
STATE_FILE    = Path(__file__).parent / "tinker_state.json"
CORPUS_FILE   = Path(__file__).parent / "parallel_corpus.csv"

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate accurately and naturally."
)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus(path=CORPUS_FILE):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lb  = row["lun_bawang"].strip()
            eng = row["english"].strip()
            if lb and eng:
                rows.append((lb, eng))
    return rows


# ── Tokenisation + Datum construction ────────────────────────────────────────

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


# ── State persistence ─────────────────────────────────────────────────────────

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    print(f"State saved → {STATE_FILE}")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print(f"Model:      {BASE_MODEL}")
    print(f"LoRA rank:  {LORA_RANK}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs:     {EPOCHS}")
    print(f"Direction:  bidirectional (LB↔EN)")
    print()

    print("Loading corpus...")
    corpus = load_corpus()
    print(f"  {len(corpus)} sentence pairs → {len(corpus)*2} examples (both directions)")

    service = ServiceClient()
    state = load_state()

    if state and state.get("model_id"):
        print(f"\nResuming experiment: {state['model_id']}")
        tc = service.create_training_client_from_state(state["model_id"])
    else:
        print("\nStarting new training experiment...")
        tc = service.create_lora_training_client(
            base_model=BASE_MODEL,
            rank=LORA_RANK,
            seed=42,
        )
        info = tc.get_info()
        state = {"model_id": info.model_id, "checkpoints": [], "steps": 0}
        save_state(state)
        print(f"  Experiment ID: {info.model_id}")

    print("\nFetching tokenizer...")
    tokenizer = tc.get_tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print("\nTokenising corpus (both directions)...")
    all_datums = []
    skipped = 0
    for lb, eng in corpus:
        d_lb2en = make_datum(tokenizer, lb, eng, direction="lb2en")
        d_en2lb = make_datum(tokenizer, eng, lb, direction="en2lb")
        for d in (d_lb2en, d_en2lb):
            if d is None:
                skipped += 1
            else:
                all_datums.append(d)

    print(f"  {len(all_datums)} datums ready ({skipped} skipped as too long)")

    global_step = state.get("steps", 0)
    print(f"\nStarting training from step {global_step}...")

    for epoch in range(EPOCHS):
        random.shuffle(all_datums)
        batches = [all_datums[i:i+BATCH_SIZE] for i in range(0, len(all_datums), BATCH_SIZE)]
        print(f"\n── Epoch {epoch+1}/{EPOCHS}  ({len(batches)} batches) ──")

        for step, batch in enumerate(batches):
            fwdbwd_future = tc.forward_backward(batch, "cross_entropy")
            optim_future  = tc.optim_step(AdamParams(learning_rate=LEARNING_RATE))

            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            # Compute loss for logging using elementwise_loss (already weighted by Tinker)
            total_loss = fwdbwd_result.metrics.get("loss:sum", 0.0)
            n_weighted = sum(
                sum(1 for x in out["elementwise_loss"].data if x != 0.0)
                for out in fwdbwd_result.loss_fn_outputs
            )
            loss = total_loss / max(n_weighted, 1)

            global_step += 1
            print(f"  step {global_step:5d} | loss {loss:.4f} | epoch {epoch+1} batch {step+1}/{len(batches)}")

            if global_step % SAVE_EVERY == 0:
                print(f"  → Saving checkpoint at step {global_step}...")
                sc = tc.save_weights_and_get_sampling_client()
                ckpt = sc.checkpoint.tinker_path
                state["checkpoints"].append({"step": global_step, "path": ckpt})
                state["steps"] = global_step
                save_state(state)
                print(f"    Checkpoint: {ckpt}")

        print(f"\nEpoch {epoch+1} done — saving checkpoint...")
        sc = tc.save_weights_and_get_sampling_client()
        ckpt = sc.checkpoint.tinker_path
        state["checkpoints"].append({"step": global_step, "path": ckpt, "epoch": epoch+1})
        state["steps"] = global_step
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
    tokenizer = sc.get_tokenizer()

    if direction == "lb2en":
        user_content = f"Translate to English:\n{text}"
    else:
        user_content = f"Translate to Lun Bawang:\n{text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt_tokens = _token_ids(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    ))

    prompt = ModelInput.from_ints(tokens=prompt_tokens)
    params = SamplingParams(max_tokens=256, temperature=0.1, top_p=0.9)

    result = sc.sample(prompt, num_samples=1, sampling_params=params).result()
    output_tokens = result.samples[0].tokens
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
    parser = argparse.ArgumentParser(description="Lun Bawang ↔ English translator (Tinker/Qwen3-8B)")
    parser.add_argument("--train",     action="store_true", help="Run fine-tuning")
    parser.add_argument("--translate", action="store_true", help="Interactive translation")
    parser.add_argument("--direction", choices=["lb2en", "en2lb"], default="lb2en",
                        help="Translation direction (default: lb2en)")
    parser.add_argument("--text",      type=str, help="Single text to translate")
    parser.add_argument("--checkpoint",type=str, help="Specific checkpoint path")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.translate:
        if args.text:
            result = translate(args.text, args.direction, args.checkpoint)
            arrow = "EN" if args.direction == "lb2en" else "LB"
            print(f"{arrow}: {result}")
        else:
            interactive_translate(args.direction)
    else:
        parser.print_help()
