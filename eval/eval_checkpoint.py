"""
Quick BLEU/exact-match eval for a single checkpoint.
Usage: python3.13 eval_checkpoint.py [checkpoint_path]
Defaults to the latest checkpoint in tinker_state.json.

For checkpoints from a separate experiment (different base model, own state
file), pass --state-file so results are labeled and written back correctly
instead of being bucketed into the default v1/v0 labels:
  python3.13 eval_checkpoint.py <checkpoint> --state-file tinker_state_inkling.json --label inkling-small

Outputs (per run):
  eval_raw/<checkpoint>_<timestamp>.jsonl  — one JSON object per line, flushed
                                             after each row (crash-safe)
  tinker_state.json                        — updated with BLEU scores

Run merge_evals.py to combine all JSONL files into eval_outputs.csv.
"""
import os, csv, json, re, random, sys, argparse
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

ROOT_DIR     = Path(__file__).parent.parent
STATE_FILE   = ROOT_DIR / "tinker_state.json"
CORPUS_FILE  = ROOT_DIR / "corpus" / "parallel_corpus.csv"
AUX_FILE     = ROOT_DIR / "corpus" / "aux_corpus.csv"
TINKER_BASE  = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
API_KEY      = os.environ["TINKER_API_KEY"]
RAW_DIR      = Path(__file__).parent / "eval_raw"

SYSTEM_PROMPT = (
    "You are a translator specializing in the Lun Bawang language of Borneo. "
    "Translate ONLY the exact text provided — output just the translation, nothing else. "
    "Use everyday conversational language, not religious or scriptural register. "
    "Proper names (e.g. Bethel, Joyce, Sarah) are names of ordinary people — do not treat them as biblical references. "
    "Preserve proper nouns (personal names, place names) exactly as they appear in the input unless you are certain of the standard equivalent in the target language. "
    "Do not expand, paraphrase, or add any meaning not present in the input. "
    "Do not produce Bible verse language."
)

# ── Checkpoint selection ──────────────────────────────────────────────────────
RUN1_FILE  = ROOT_DIR / "tinker_state_run1.json"
state      = json.loads(STATE_FILE.read_text())
run1_state = json.loads(RUN1_FILE.read_text()) if RUN1_FILE.exists() else {"model_id": "", "checkpoints": []}

_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument("checkpoint", nargs="?", default=None)
_arg_parser.add_argument("--state-file", type=str, default=None,
                          help="Explicit state file for a separate experiment "
                               "(e.g. tinker_state_inkling.json) — results are "
                               "labeled and written back here instead of v0/v1")
_arg_parser.add_argument("--label", type=str, default=None,
                          help="Override the run label used in output filenames/CSVs")
cli_args = _arg_parser.parse_args()

if cli_args.state_file:
    target_state_file = Path(cli_args.state_file)
    if not target_state_file.is_absolute():
        target_state_file = ROOT_DIR / target_state_file
    ext_state = json.loads(target_state_file.read_text())
    CHECKPOINT = cli_args.checkpoint or ext_state["checkpoints"][-1]["path"]
    step = next((c["step"] for c in ext_state["checkpoints"] if c["path"] == CHECKPOINT), None)
    run  = cli_args.label or target_state_file.stem.removeprefix("tinker_state_") or "custom"
    STATE_FILE = target_state_file   # so the write-back block below updates the right file
elif cli_args.checkpoint:
    CHECKPOINT = cli_args.checkpoint
    step = next((c["step"] for c in state["checkpoints"] if c["path"] == CHECKPOINT), None)
    run  = "v1"
    if step is None:
        step = next((c["step"] for c in run1_state["checkpoints"] if c["path"] == CHECKPOINT), None)
        if step is not None:
            run = "v0"
else:
    ck = state["checkpoints"][-1]
    CHECKPOINT = ck["path"]
    step = ck["step"]
    run  = "v1"

# Short label for the JSONL filename (e.g. "v1·checkpoint-8000")
ck_label    = CHECKPOINT.split("/")[-1] if "/" in CHECKPOINT else CHECKPOINT
model_label = f"{run}·{ck_label}"

print(f"Checkpoint: {CHECKPOINT}\n")

# ── Load + split corpora ──────────────────────────────────────────────────────
def load_bible(path=CORPUS_FILE):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lb, eng, book = r["lun_bawang"].strip(), r["english"].strip(), r.get("book_code","UNK").strip()
            if lb and eng:
                rows.append((lb, eng, book))
    return rows

def load_aux(path=AUX_FILE):
    if not Path(path).exists(): return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lb = r["lun_bawang"].strip(); eng = r["english"].strip()
            src = r.get("source","aux").strip(); typ = r.get("type","word").strip()
            if lb and eng: rows.append((lb, eng, src, typ))
    return rows

def bible_split(corpus, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    by_book = {}
    for item in corpus: by_book.setdefault(item[2],[]).append(item)
    train, val = [], []
    for book, items in sorted(by_book.items()):
        sh = list(items); rng.shuffle(sh)
        n = max(1, int(len(sh)*val_frac))
        val.extend(sh[:n]); train.extend(sh[n:])
    return train, val

def aux_split(corpus, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    by_src = {}
    for item in corpus: by_src.setdefault(item[2],[]).append(item)
    train, val = [], []
    for src, items in sorted(by_src.items()):
        sh = list(items); rng.shuffle(sh)
        n = max(1, int(len(sh)*val_frac))
        val.extend(sh[:n]); train.extend(sh[n:])
    return train, val

print("Loading corpora…")
_, bible_val = bible_split(load_bible())
_, aux_val   = aux_split(load_aux())
dict_val = [r for r in aux_val if r[3]=="word"]
sent_val = [r for r in aux_val if r[3]=="sentence"]
print(f"  Bible val: {len(bible_val)} | Dict val: {len(dict_val)} | Sent val: {len(sent_val)}\n")

# ── Translate helper ──────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=TINKER_BASE)

def translate(text, direction="lb2en"):
    hint = "\n(Output only the translation of this word or phrase.)" if len(text.split()) <= 5 else ""
    user_content = (
        f"Translate to English:\n{text}{hint}"
        if direction == "lb2en"
        else f"Translate to Lun Bawang:\n{text}{hint}"
    )
    import time
    t0 = time.monotonic()
    r = client.chat.completions.create(
        model=CHECKPOINT,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_content}],
        max_tokens=256, temperature=0.1, top_p=0.9,
    )
    call_ms = round((time.monotonic() - t0) * 1000)
    raw = r.choices[0].message.content or ""
    processed = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw, processed, call_ms

# ── Per-run JSONL output ──────────────────────────────────────────────────────
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
safe_label = re.sub(r"[^A-Za-z0-9_\-]", "_", model_label)
RAW_DIR.mkdir(exist_ok=True)
jsonl_path = RAW_DIR / f"{safe_label}_{ts.replace(':', '')}.jsonl"
jsonl_file = open(jsonl_path, "w", encoding="utf-8")

def write_row(eval_set, input_lb, raw, processed, reference, call_ms=None):
    record = {
        "timestamp":        ts,
        "model":            model_label,
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

import sacrebleu as sb

# ── Bible BLEU (lb→en) ────────────────────────────────────────────────────────
rng = random.Random(42)
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

# ── Bible BLEU (en→lb) ────────────────────────────────────────────────────────
print(f"Bible BLEU (50 examples, en→lb)…")
hyps_el, refs_el = [], []
for i, (lb, eng, *_) in enumerate(sample, 1):
    raw, proc, call_ms = translate(eng, "en2lb")
    hyps_el.append(proc); refs_el.append(lb)
    write_row("bible_en2lb", eng, raw, proc, lb, call_ms)
    if i % 10 == 0: print(f"  {i}/50…")
bible_bleu_el = sb.corpus_bleu(hyps_el, [refs_el]).score
print(f"  → {bible_bleu_el:.2f}\n")

# ── Dict exact match (lb→en) ──────────────────────────────────────────────────
print(f"Dict exact match ({len(dict_val)} examples, lb→en)…")
d_hyps, d_refs = [], []
for i, (lb, eng, *_) in enumerate(dict_val, 1):
    raw, proc, call_ms = translate(lb, "lb2en")
    d_hyps.append(proc); d_refs.append(eng)
    write_row("dict", lb, raw, proc, eng, call_ms)
    if i % 10 == 0: print(f"  {i}/{len(dict_val)}…")
correct  = sum(h.lower().strip()==r.lower().strip() for h,r in zip(d_hyps, d_refs))
dict_pct = correct / len(d_refs) * 100
print(f"  → {dict_pct:.1f}% ({correct}/{len(d_refs)})\n")

print("Dict samples (lb→en):")
for lb, hyp, ref in zip([r[0] for r in dict_val], d_hyps, d_refs):
    mark = "✓" if hyp.lower().strip()==ref.lower().strip() else "✗"
    print(f"  {mark} '{lb}' → '{hyp}' (ref: '{ref}')")

# ── Dict exact match (en→lb) ──────────────────────────────────────────────────
print(f"\nDict exact match ({len(dict_val)} examples, en→lb)…")
d_hyps_el, d_refs_el = [], []
for i, (lb, eng, *_) in enumerate(dict_val, 1):
    raw, proc, call_ms = translate(eng, "en2lb")
    d_hyps_el.append(proc); d_refs_el.append(lb)
    write_row("dict_en2lb", eng, raw, proc, lb, call_ms)
    if i % 10 == 0: print(f"  {i}/{len(dict_val)}…")
correct_el  = sum(h.lower().strip()==r.lower().strip() for h,r in zip(d_hyps_el, d_refs_el))
dict_pct_el = correct_el / len(d_refs_el) * 100
print(f"  → {dict_pct_el:.1f}% ({correct_el}/{len(d_refs_el)})\n")

print("Dict samples (en→lb):")
for lb, hyp, ref in zip([r[0] for r in dict_val], d_hyps_el, d_refs_el):
    mark = "✓" if hyp.lower().strip()==ref.lower().strip() else "✗"
    print(f"  {mark} '{ref}' → '{hyp}' (ref: '{lb}')")

# ── Sentence BLEU (lb→en) ─────────────────────────────────────────────────────
print(f"\nSentence BLEU ({len(sent_val)} examples, lb→en)…")
s_hyps, s_refs = [], []
for i, (lb, eng, *_) in enumerate(sent_val, 1):
    raw, proc, call_ms = translate(lb, "lb2en")
    s_hyps.append(proc); s_refs.append(eng)
    write_row("sentence", lb, raw, proc, eng, call_ms)
    if i % 10 == 0: print(f"  {i}/{len(sent_val)}…")
sent_bleu = sb.corpus_bleu(s_hyps, [s_refs]).score
print(f"  → {sent_bleu:.2f} (combined)\n")

short_pairs = [(h, r) for h, r in zip(s_hyps, s_refs) if len(r.split()) <= 10]
long_pairs  = [(h, r) for h, r in zip(s_hyps, s_refs) if len(r.split()) >  10]
short_bleu = sb.corpus_bleu([p[0] for p in short_pairs], [[p[1] for p in short_pairs]]).score if short_pairs else 0.0
long_bleu  = sb.corpus_bleu([p[0] for p in long_pairs],  [[p[1] for p in long_pairs]]).score  if long_pairs  else 0.0
print(f"  Short (ref ≤10 words): {len(short_pairs)} examples → {short_bleu:.2f}")
print(f"  Long  (ref >10 words): {len(long_pairs)}  examples → {long_bleu:.2f}\n")

print("Sentence samples (lb→en):")
for (lb,*_), hyp, ref in zip(sent_val, s_hyps, s_refs):
    print(f"  LB:  {lb}\n  Got: {hyp}\n  Ref: {ref}\n")

# ── Sentence BLEU (en→lb) ─────────────────────────────────────────────────────
print(f"Sentence BLEU ({len(sent_val)} examples, en→lb)…")
s_hyps_el, s_refs_el = [], []
for i, (lb, eng, *_) in enumerate(sent_val, 1):
    raw, proc, call_ms = translate(eng, "en2lb")
    s_hyps_el.append(proc); s_refs_el.append(lb)
    write_row("sentence_en2lb", eng, raw, proc, lb, call_ms)
    if i % 10 == 0: print(f"  {i}/{len(sent_val)}…")
sent_bleu_el = sb.corpus_bleu(s_hyps_el, [s_refs_el]).score
print(f"  → {sent_bleu_el:.2f} (combined)\n")

short_pairs_el = [(h, r) for h, r in zip(s_hyps_el, s_refs_el) if len(r.split()) <= 10]
long_pairs_el  = [(h, r) for h, r in zip(s_hyps_el, s_refs_el) if len(r.split()) >  10]
short_bleu_el = sb.corpus_bleu([p[0] for p in short_pairs_el], [[p[1] for p in short_pairs_el]]).score if short_pairs_el else 0.0
long_bleu_el  = sb.corpus_bleu([p[0] for p in long_pairs_el],  [[p[1] for p in long_pairs_el]]).score  if long_pairs_el  else 0.0
print(f"  Short (ref ≤10 words): {len(short_pairs_el)} examples → {short_bleu_el:.2f}")
print(f"  Long  (ref >10 words): {len(long_pairs_el)}  examples → {long_bleu_el:.2f}\n")

print("Sentence samples (en→lb):")
for (lb, eng, *_), hyp in zip(sent_val, s_hyps_el):
    print(f"  EN:  {eng}\n  Got: {hyp}\n  Ref: {lb}\n")

jsonl_file.close()

# ── Update tinker_state.json / tinker_state_run1.json ────────────────────────
if step is not None:
    import fcntl
    target_file = RUN1_FILE if run == "v0" else STATE_FILE
    new_fields = {
        "val_bleu_bible":            round(bible_bleu, 3),
        "val_bleu_bible_en2lb":      round(bible_bleu_el, 3),
        "val_exact_dict":            round(dict_pct, 1),
        "val_exact_dict_en2lb":      round(dict_pct_el, 1),
        "val_bleu_sentence":         round(sent_bleu, 3),
        "val_bleu_sent_short":       round(short_bleu, 3),
        "val_bleu_sent_long":        round(long_bleu, 3),
        "val_bleu_sentence_en2lb":   round(sent_bleu_el, 3),
        "val_bleu_sent_short_en2lb": round(short_bleu_el, 3),
        "val_bleu_sent_long_en2lb":  round(long_bleu_el, 3),
    }
    with open(target_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        current = json.load(f)
        for ck in current["checkpoints"]:
            if ck["step"] == step:
                ck.update(new_fields)
                break
        f.seek(0)
        json.dump(current, f, indent=2)
        f.truncate()
    print(f"Saved to {target_file.name}\n")

print(f"Raw JSONL → {jsonl_path}")

# ── Auto-merge into eval_outputs.csv ─────────────────────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location("merge_evals", Path(__file__).parent / "merge_evals.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
rows, files = _mod.load_all_jsonl()
rows.sort(key=lambda r: (r["model"], _mod.EVAL_SET_ORDER.get(r["eval_set"], 99)))

import csv as _csv
from collections import defaultdict as _dd

def _write_csv(path, subset):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_mod.COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(subset)

_write_csv(_mod.OUTPUT_FILE, rows)
by_gen = _dd(list)
for r in rows:
    gen = r["model"].split("·")[0] if "·" in r["model"] else r["model"]
    by_gen[gen].append(r)
for gen, gen_rows in sorted(by_gen.items()):
    safe = gen.replace("/", "-").replace(" ", "_")
    _write_csv(_mod.OUTPUT_FILE.parent / f"eval_outputs_{safe}.csv", gen_rows)

print(f"eval_outputs.csv updated → {len(rows)} rows across {len(files)} run(s) | {len(by_gen)} generation(s)")
