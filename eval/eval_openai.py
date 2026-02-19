"""
Eval script for OpenAI models — same val set as eval_checkpoint.py.

Usage:
  python3.13 eval_openai.py                         # defaults to gpt-4o
  python3.13 eval_openai.py --model gpt-4o
  python3.13 eval_openai.py --model gpt-5
  python3.13 eval_openai.py --model gpt-4o-mini

API key is read from OPENAI_API_KEY env var.

Outputs (per run):
  eval_raw/<model>_<timestamp>.jsonl  — one JSON object per line, flushed after
                                        each row (crash-safe, no file contention)
  eval_results_openai.json            — summary scores per model

Run merge_evals.py to combine all JSONL files into eval_outputs.csv.
"""
import argparse, csv, json, os, re, random
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate ONLY the exact text provided — output just the translation, nothing else. "
    "Do not add Bible verse titles, context, or anything not present in the input."
)

ROOT_DIR     = Path(__file__).parent.parent
CORPUS_FILE  = ROOT_DIR / "corpus" / "parallel_corpus.csv"
AUX_FILE     = ROOT_DIR / "corpus" / "aux_corpus.csv"
RESULTS_FILE = Path(__file__).parent / "eval_results_openai.json"
RAW_DIR      = Path(__file__).parent / "eval_raw"


# ── Corpus loading (identical splits to eval_checkpoint.py) ──────────────────

def load_bible(path=CORPUS_FILE):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lb   = r["lun_bawang"].strip()
            eng  = r["english"].strip()
            book = r.get("book_code", "UNK").strip()
            if lb and eng:
                rows.append((lb, eng, book))
    return rows


def load_aux(path=AUX_FILE):
    if not Path(path).exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lb  = r["lun_bawang"].strip()
            eng = r["english"].strip()
            src = r.get("source", "aux").strip()
            typ = r.get("type", "word").strip()
            if lb and eng:
                rows.append((lb, eng, src, typ))
    return rows


def bible_split(corpus, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    by_book = {}
    for item in corpus:
        by_book.setdefault(item[2], []).append(item)
    train, val = [], []
    for book, items in sorted(by_book.items()):
        sh = list(items)
        rng.shuffle(sh)
        n = max(1, int(len(sh) * val_frac))
        val.extend(sh[:n])
        train.extend(sh[n:])
    return train, val


def aux_split(corpus, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    by_src = {}
    for item in corpus:
        by_src.setdefault(item[2], []).append(item)
    train, val = [], []
    for src, items in sorted(by_src.items()):
        sh = list(items)
        rng.shuffle(sh)
        n = max(1, int(len(sh) * val_frac))
        val.extend(sh[:n])
        train.extend(sh[n:])
    return train, val


# ── Translate ─────────────────────────────────────────────────────────────────

def make_translate(client, model):
    # gpt-5* and o-series are reasoning models: no temperature/top_p, need
    # larger token budget (reasoning tokens consume most of max_completion_tokens)
    reasoning_model = model.startswith("o") or model.startswith("gpt-5")
    max_tokens = 4096 if reasoning_model else 256

    def translate(text, direction="lb2en"):
        import time
        hint = "\n(Output only the translation of this word or phrase.)" if len(text.split()) <= 5 else ""
        user_content = (
            f"Translate to English:\n{text}{hint}"
            if direction == "lb2en"
            else f"Translate to Lun Bawang:\n{text}{hint}"
        )
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_completion_tokens=max_tokens,
        )
        if not reasoning_model:
            kwargs["temperature"] = 0.1
            kwargs["top_p"]       = 0.9
        t0  = time.monotonic()
        r   = client.chat.completions.create(**kwargs)
        call_ms = round((time.monotonic() - t0) * 1000)
        raw = r.choices[0].message.content or ""
        processed = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        return raw, processed, call_ms
    return translate


# ── Sanity check ─────────────────────────────────────────────────────────────

def _sanity_check(jsonl_path, model):
    """Read back the first 3 rows from the JSONL and abort if anything looks wrong."""
    print(f"\n[sanity check] Reading back {jsonl_path.name}…")
    try:
        lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as e:
        raise SystemExit(f"[sanity check FAILED] Could not read JSONL: {e}")

    if len(lines) < 3:
        raise SystemExit(f"[sanity check FAILED] Expected ≥3 lines, got {len(lines)}")

    empty_outputs = 0
    for i, line in enumerate(lines[:3], 1):
        try:
            row = json.loads(line)
        except Exception as e:
            raise SystemExit(f"[sanity check FAILED] Line {i} is not valid JSON: {e}")
        for field in ("timestamp", "model", "eval_set", "input_lb", "raw_output", "processed_output", "reference"):
            if field not in row:
                raise SystemExit(f"[sanity check FAILED] Line {i} missing field '{field}'")
        if not row["raw_output"].strip():
            empty_outputs += 1
            print(f"  [!] Row {i}: raw_output is empty for input: {row['input_lb']!r}")

    if empty_outputs == 3:
        raise SystemExit(
            f"[sanity check FAILED] All 3 sampled outputs are empty for model '{model}'.\n"
            "Check token limits, API parameters, or model availability."
        )
    elif empty_outputs > 0:
        print(f"  [warning] {empty_outputs}/3 outputs are empty — monitor closely.")
    else:
        print(f"  [OK] 3/3 rows written correctly, outputs non-empty.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Eval an OpenAI model on the Lun Bawang val set")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    client    = OpenAI(api_key=api_key)
    translate = make_translate(client, args.model)
    ts        = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Per-run JSONL file — safe_model strips characters invalid in filenames
    safe_model = re.sub(r"[^A-Za-z0-9_\-]", "_", args.model)
    RAW_DIR.mkdir(exist_ok=True)
    jsonl_path = RAW_DIR / f"{safe_model}_{ts.replace(':', '')}.jsonl"
    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    rows_written = 0

    def write_row(eval_set, input_lb, raw, processed, reference, call_ms=None):
        nonlocal rows_written
        record = {
            "timestamp":        ts,
            "model":            args.model,
            "eval_set":         eval_set,
            "input_lb":         input_lb,
            "raw_output":       raw,
            "processed_output": processed,
            "reference":        reference,
        }
        if call_ms is not None:
            record["call_ms"] = call_ms
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()  # land on disk immediately — crash-safe
        rows_written += 1
        if rows_written == 3:
            _sanity_check(jsonl_path, args.model)

    print(f"Model: {args.model}\n")

    print("Loading corpora…")
    _, bible_val = bible_split(load_bible())
    _, aux_val   = aux_split(load_aux())
    dict_val = [r for r in aux_val if r[3] == "word"]
    sent_val = [r for r in aux_val if r[3] == "sentence"]
    print(f"  Bible val: {len(bible_val)} | Dict val: {len(dict_val)} | Sent val: {len(sent_val)}\n")

    import sacrebleu as sb

    # Bible BLEU
    rng    = random.Random(42)
    sample = rng.sample(bible_val, min(50, len(bible_val)))
    print(f"Bible BLEU (50 examples, lb→en)…")
    hyps, refs = [], []
    for i, (lb, eng, *_) in enumerate(sample, 1):
        raw, proc, call_ms = translate(lb, "lb2en")
        hyps.append(proc); refs.append(eng)
        write_row("bible", lb, raw, proc, eng, call_ms)
        if i % 10 == 0: print(f"  {i}/50…")
    bible_bleu = sb.corpus_bleu(hyps, [refs]).score
    print(f"  → {bible_bleu:.2f}\n")

    # Dict exact match
    print(f"Dict exact match ({len(dict_val)} examples, lb→en)…")
    d_hyps, d_refs = [], []
    for i, (lb, eng, *_) in enumerate(dict_val, 1):
        raw, proc, call_ms = translate(lb, "lb2en")
        d_hyps.append(proc); d_refs.append(eng)
        write_row("dict", lb, raw, proc, eng, call_ms)
        if i % 10 == 0: print(f"  {i}/{len(dict_val)}…")
    correct  = sum(h.lower().strip() == r.lower().strip() for h, r in zip(d_hyps, d_refs))
    dict_pct = correct / len(d_refs) * 100
    print(f"  → {dict_pct:.1f}% ({correct}/{len(d_refs)})\n")

    # Sentence BLEU
    print(f"Sentence BLEU ({len(sent_val)} examples, lb→en)…")
    s_hyps, s_refs = [], []
    for i, (lb, eng, *_) in enumerate(sent_val, 1):
        raw, proc, call_ms = translate(lb, "lb2en")
        s_hyps.append(proc); s_refs.append(eng)
        write_row("sentence", lb, raw, proc, eng, call_ms)
        if i % 10 == 0: print(f"  {i}/{len(sent_val)}…")
    sent_bleu = sb.corpus_bleu(s_hyps, [s_refs]).score
    print(f"  → {sent_bleu:.2f}\n")

    jsonl_file.close()

    print("Sentence samples:")
    for (lb, *_), hyp, ref in zip(sent_val, s_hyps, s_refs):
        print(f"  LB:  {lb}\n  Got: {hyp}\n  Ref: {ref}\n")

    print("Dict samples:")
    for lb, hyp, ref in zip([r[0] for r in dict_val], d_hyps, d_refs):
        mark = "✓" if hyp.lower().strip() == ref.lower().strip() else "✗"
        print(f"  {mark} '{lb}' → '{hyp}' (ref: '{ref}')")

    # Save summary results
    results = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
    results[args.model] = {
        "timestamp":      ts,
        "bible_bleu":     round(bible_bleu, 2),
        "dict_exact_pct": round(dict_pct, 1),
        "sentence_bleu":  round(sent_bleu, 2),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nSummary  → {RESULTS_FILE}")
    print(f"Raw JSONL → {jsonl_path}")
    print(f"Run 'python3.13 merge_evals.py' to rebuild eval_outputs.csv")


if __name__ == "__main__":
    main()
