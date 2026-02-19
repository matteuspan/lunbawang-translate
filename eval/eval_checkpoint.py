"""
Quick BLEU/exact-match eval for a single checkpoint.
Usage: python3.13 eval_checkpoint.py [checkpoint_path]
Defaults to the latest checkpoint in tinker_state.json.

Outputs (per run):
  eval_raw/<checkpoint>_<timestamp>.jsonl  — one JSON object per line, flushed
                                             after each row (crash-safe)
  tinker_state.json                        — updated with BLEU scores

Run merge_evals.py to combine all JSONL files into eval_outputs.csv.
"""
import os, csv, json, re, random, sys
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
    "Do not add Bible verse titles, context, or anything not present in the input."
)

# ── Checkpoint selection ──────────────────────────────────────────────────────
state = json.loads(STATE_FILE.read_text())
if len(sys.argv) > 1:
    CHECKPOINT = sys.argv[1]
    step = next((c["step"] for c in state["checkpoints"] if c["path"] == CHECKPOINT), None)
else:
    ck = state["checkpoints"][-1]
    CHECKPOINT = ck["path"]
    step = ck["step"]

# Short label for the JSONL filename (e.g. "checkpoint-8000")
model_label = CHECKPOINT.split("/")[-1] if "/" in CHECKPOINT else CHECKPOINT

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

# ── Bible BLEU ────────────────────────────────────────────────────────────────
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

# ── Dict exact match ──────────────────────────────────────────────────────────
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

print("Dict samples:")
for lb, hyp, ref in zip([r[0] for r in dict_val], d_hyps, d_refs):
    mark = "✓" if hyp.lower().strip()==ref.lower().strip() else "✗"
    print(f"  {mark} '{lb}' → '{hyp}' (ref: '{ref}')")

# ── Sentence BLEU ─────────────────────────────────────────────────────────────
print(f"\nSentence BLEU ({len(sent_val)} examples, lb→en)…")
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
for (lb,*_), hyp, ref in zip(sent_val, s_hyps, s_refs):
    print(f"  LB:  {lb}\n  Got: {hyp}\n  Ref: {ref}\n")

# ── Update tinker_state.json ──────────────────────────────────────────────────
if step is not None:
    for ck in state["checkpoints"]:
        if ck["step"] == step:
            ck["val_bleu_bible"]    = round(bible_bleu, 3)
            ck["val_exact_dict"]    = round(dict_pct, 1)
            ck["val_bleu_sentence"] = round(sent_bleu, 3)
            break
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print("Saved to tinker_state.json\n")

print(f"Raw JSONL → {jsonl_path}")
print(f"Run 'python3.13 merge_evals.py' to rebuild eval_outputs.csv")
