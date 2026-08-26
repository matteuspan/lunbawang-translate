"""
Quick BLEU/exact-match eval for a single checkpoint.
Usage: python3.13 eval_checkpoint.py [checkpoint_path]
Defaults to the latest checkpoint in tinker_state.json.

For checkpoints from a separate experiment (different base model, own state
file), pass --state-file so results are labeled and written back correctly
instead of being bucketed into the default v1/v0 labels — the label defaults
to the state file's name (e.g. tinker_state_inkling.json -> "inkling"), or
pass --label to override it:
  python3.13 eval_checkpoint.py <checkpoint> --state-file tinker_state_inkling.json

Outputs (per run):
  eval_raw/<checkpoint>_<timestamp>.jsonl  — one JSON object per line, flushed
                                             after each row (crash-safe)
  tinker_state.json                        — updated with BLEU scores

Run merge_evals.py to combine all JSONL files into eval_outputs.csv.
"""
import os, csv, json, re, random, sys, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

MAX_WORKERS = 24  # concurrent translation calls — sequential was the dominant eval
                  # cost; the calls are I/O-bound so raising this scales wall-time
                  # down almost linearly (Bible verses are the slowest, ~3s each)
VAL_DICTWORD = 150  # cap on new-dictionary word probes (of ~779) — seeded sample,
VAL_DICTSENT = 60   # same probes every eval so the trajectory stays comparable

ROOT_DIR     = Path(__file__).parent.parent
STATE_FILE   = ROOT_DIR / "tinker_state.json"
CORPUS_FILE  = ROOT_DIR / "corpus" / "parallel_corpus.csv"
AUX_FILE     = ROOT_DIR / "corpus" / "aux_corpus.csv"
DICT_FILE    = ROOT_DIR / "corpus" / "dictionary_corpus.csv"
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
_arg_parser.add_argument("--base-model", type=str, default=None,
                          help="Base model the checkpoint was fine-tuned from. For "
                               "thinkingmachines/* (Inkling family), passes "
                               "reasoning_effort='low' on every completion call — "
                               "reasoning_effort='none' (the effective default when "
                               "unset) causes frequent single-token empty completions "
                               "on short phrases; 'low' fixed this in testing while "
                               "staying fast.")
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
print(f"  Bible val: {len(bible_val)} | Dict val: {len(dict_val)} | Sent val: {len(sent_val)}")

# Dictionary corpus held-out slices. aux_split here is byte-identical to the
# trainer's aux_train_val_split (same seed=42, same 0.1 fraction, same full
# corpus before any definition trim), so these ARE exactly the trainer's
# held-out dictionary rows — evaluated but never trained on.
_dict_corpus = load_aux(DICT_FILE)
_, _dict_val = aux_split(_dict_corpus, val_frac=0.1) if _dict_corpus else ([], [])
dictword_val = [r for r in _dict_val if r[3] == "word"]
dictsent_val = [r for r in _dict_val if r[3] == "sentence"]
# Sub-sample to keep each eval cheap (~420 calls vs ~2000). Seeded, so the same
# probes are drawn every checkpoint — the trajectory stays comparable.
_rng_dict = random.Random(42)
if len(dictword_val) > VAL_DICTWORD:
    dictword_val = _rng_dict.sample(dictword_val, VAL_DICTWORD)
if len(dictsent_val) > VAL_DICTSENT:
    dictsent_val = _rng_dict.sample(dictsent_val, VAL_DICTSENT)
print(f"  Dict(word) val: {len(dictword_val)} | Dict(sent) val: {len(dictsent_val)}\n")

# ── Translate helper ──────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY, base_url=TINKER_BASE)

_uses_tml = bool(cli_args.base_model) and cli_args.base_model.split(":")[0].startswith(
    ("thinkingmachines/", "TML/")
)
_messages_base = [{"role": "system", "content": SYSTEM_PROMPT}]
# reasoning_effort="none" (the effective default when unset) reproduced a 100%
# empty-completion rate on short phrases in testing; "low" cut it to ~19%
# residual, but on the dictionary-retrained v2 weights ~36% of single-word
# lb->en probes still come back empty at "low" (they score zero, understating
# dict-word metrics). "medium" drops the empty rate to ~0% across checkpoints,
# so it's the eval floor — matching serve.py, which also starts every input at
# "medium". A "Thinking effort level: N" system message was tried first but
# turned out to be inert noise over this REST endpoint — reasoning_effort is
# the real control.
_completion_kwargs = {"reasoning_effort": "medium"} if _uses_tml else {}


def translate(text, direction="lb2en", retries=2):
    """A completion with empty content and finish_reason == "stop" is
    ambiguous: either the model genuinely produced nothing, or a transient
    network/server hiccup returned a truncated response without raising an
    exception (observed: this correlated with connection instability that
    preceded a container restart). Retry a couple of times before accepting
    empty as the real answer, so eval numbers reflect the model, not noise."""
    # Match the training prompt (train_translator.py) and serve.py verbatim: a
    # bare "Translate to English:" / "Translate to Lun Bawang:" with no
    # short-input hint. The "(Output only …)" hint is out-of-distribution and
    # reliably empties hard single words at every reasoning_effort (e.g. "water"
    # en->lb: 5/5 empty with the hint, 0/5 without) — those blanks scored zero
    # and were quietly depressing the dictionary single-word metrics. Dropping
    # it makes eval measure exactly what production serves.
    user_content = (
        f"Translate to English:\n{text}"
        if direction == "lb2en"
        else f"Translate to Lun Bawang:\n{text}"
    )
    t0 = time.monotonic()
    for attempt in range(retries + 1):
        r = client.chat.completions.create(
            model=CHECKPOINT,
            messages=_messages_base + [{"role": "user", "content": user_content}],
            max_tokens=256, temperature=0.1, top_p=0.9,
            **_completion_kwargs,
        )
        raw = r.choices[0].message.content or ""
        if raw or attempt == retries:
            break
    call_ms = round((time.monotonic() - t0) * 1000)
    processed = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw, processed, call_ms


def translate_batch(pairs, direction, eval_set, src_idx, ref_idx):
    """Translate a list of pairs concurrently (order-preserving), writing each
    row to the JSONL as it completes. Returns (hyps, refs)."""
    results = [None] * len(pairs)

    def _do(i, pair):
        src, ref = pair[src_idx], pair[ref_idx]
        raw, proc, call_ms = translate(src, direction)
        return i, src, ref, raw, proc, call_ms

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_do, i, pair) for i, pair in enumerate(pairs)]
        for fut in as_completed(futures):
            i, src, ref, raw, proc, call_ms = fut.result()
            results[i] = (proc, ref)
            write_row(eval_set, src, raw, proc, ref, call_ms, idx=i)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(pairs)}…")

    return [r[0] for r in results], [r[1] for r in results]

# ── Per-run JSONL output ──────────────────────────────────────────────────────
import threading
ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
safe_label = re.sub(r"[^A-Za-z0-9_\-]", "_", model_label)
RAW_DIR.mkdir(exist_ok=True)
jsonl_path = RAW_DIR / f"{safe_label}_{ts.replace(':', '')}.jsonl"
jsonl_file = open(jsonl_path, "w", encoding="utf-8")
_jsonl_lock = threading.Lock()  # translate_batch writes from multiple threads

def write_row(eval_set, input_lb, raw, processed, reference, call_ms=None, idx=None):
    record = {
        "timestamp":        ts,
        "model":            model_label,
        "eval_set":         eval_set,
        "input_lb":         input_lb,
        "raw_output":       raw,
        "processed_output": processed,
        "reference":        reference,
    }
    if idx is not None:
        record["idx"] = idx  # position within this eval_set's pairs list — rows land in
                              # completion order under concurrency, not submission order,
                              # so this is the only reliable way to correlate the same
                              # logical example across different checkpoints' JSONL files
    if call_ms is not None:
        record["call_ms"] = call_ms
    with _jsonl_lock:
        jsonl_file.write(json.dumps(record) + "\n")
        jsonl_file.flush()  # land on disk immediately — crash-safe

import sacrebleu as sb

# ── Bible BLEU (lb→en) ────────────────────────────────────────────────────────
rng = random.Random(42)
sample = rng.sample(bible_val, min(50, len(bible_val)))
print(f"Bible BLEU (50 examples, lb→en)…")
hyps, refs = translate_batch(sample, "lb2en", "bible", src_idx=0, ref_idx=1)
bible_bleu = sb.corpus_bleu(hyps, [refs]).score
print(f"  → {bible_bleu:.2f}\n")

# ── Bible BLEU (en→lb) ────────────────────────────────────────────────────────
print(f"Bible BLEU (50 examples, en→lb)…")
hyps_el, refs_el = translate_batch(sample, "en2lb", "bible_en2lb", src_idx=1, ref_idx=0)
bible_bleu_el = sb.corpus_bleu(hyps_el, [refs_el]).score
print(f"  → {bible_bleu_el:.2f}\n")

# ── Dict exact match (lb→en) ──────────────────────────────────────────────────
print(f"Dict exact match ({len(dict_val)} examples, lb→en)…")
d_hyps, d_refs = translate_batch(dict_val, "lb2en", "dict", src_idx=0, ref_idx=1)
correct  = sum(h.lower().strip()==r.lower().strip() for h,r in zip(d_hyps, d_refs))
dict_pct = correct / len(d_refs) * 100
print(f"  → {dict_pct:.1f}% ({correct}/{len(d_refs)})\n")

print("Dict samples (lb→en):")
for lb, hyp, ref in zip([r[0] for r in dict_val], d_hyps, d_refs):
    mark = "✓" if hyp.lower().strip()==ref.lower().strip() else "✗"
    print(f"  {mark} '{lb}' → '{hyp}' (ref: '{ref}')")

# ── Dict exact match (en→lb) ──────────────────────────────────────────────────
print(f"\nDict exact match ({len(dict_val)} examples, en→lb)…")
d_hyps_el, d_refs_el = translate_batch(dict_val, "en2lb", "dict_en2lb", src_idx=1, ref_idx=0)
correct_el  = sum(h.lower().strip()==r.lower().strip() for h,r in zip(d_hyps_el, d_refs_el))
dict_pct_el = correct_el / len(d_refs_el) * 100
print(f"  → {dict_pct_el:.1f}% ({correct_el}/{len(d_refs_el)})\n")

print("Dict samples (en→lb):")
for lb, hyp, ref in zip([r[0] for r in dict_val], d_hyps_el, d_refs_el):
    mark = "✓" if hyp.lower().strip()==ref.lower().strip() else "✗"
    print(f"  {mark} '{ref}' → '{hyp}' (ref: '{lb}')")

# ── Sentence BLEU (lb→en) ─────────────────────────────────────────────────────
print(f"\nSentence BLEU ({len(sent_val)} examples, lb→en)…")
s_hyps, s_refs = translate_batch(sent_val, "lb2en", "sentence", src_idx=0, ref_idx=1)
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
s_hyps_el, s_refs_el = translate_batch(sent_val, "en2lb", "sentence_en2lb", src_idx=1, ref_idx=0)
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

# ── Dictionary word exact-match (both directions) — the new lexical slice ──────
dictword_pct = dictword_pct_el = 0.0
dictword_chrf = dictword_chrf_el = 0.0
if dictword_val:
    print(f"\nDict-word ({len(dictword_val)} examples)…")
    dw_hyps, dw_refs = translate_batch(dictword_val, "lb2en", "dictword", src_idx=0, ref_idx=1)
    dictword_pct = sum(h.lower().strip()==r.lower().strip() for h,r in zip(dw_hyps, dw_refs)) / len(dw_refs) * 100
    dw_hyps_el, dw_refs_el = translate_batch(dictword_val, "en2lb", "dictword_en2lb", src_idx=1, ref_idx=0)
    dictword_pct_el = sum(h.lower().strip()==r.lower().strip() for h,r in zip(dw_hyps_el, dw_refs_el)) / len(dw_refs_el) * 100
    # chrF is far more informative than exact-match for dictionary words: it
    # credits synonyms/extra words ("lift" vs "lift something up") and one-letter
    # spelling variants ("melau" vs "meleu") that exact-match scores as total misses.
    dictword_chrf = sb.corpus_chrf(dw_hyps, [dw_refs]).score
    dictword_chrf_el = sb.corpus_chrf(dw_hyps_el, [dw_refs_el]).score
    print(f"  → exact lb→en {dictword_pct:.1f}% en→lb {dictword_pct_el:.1f}%  |  chrF lb→en {dictword_chrf:.1f} en→lb {dictword_chrf_el:.1f}\n")

# ── Dictionary sentence BLEU (both directions) — the new parallel-sentence slice ─
dictsent_bleu = dictsent_bleu_el = 0.0
if dictsent_val:
    print(f"Dict-sentence BLEU ({len(dictsent_val)} examples)…")
    ds_hyps, ds_refs = translate_batch(dictsent_val, "lb2en", "dictsent", src_idx=0, ref_idx=1)
    dictsent_bleu = sb.corpus_bleu(ds_hyps, [ds_refs]).score
    ds_hyps_el, ds_refs_el = translate_batch(dictsent_val, "en2lb", "dictsent_en2lb", src_idx=1, ref_idx=0)
    dictsent_bleu_el = sb.corpus_bleu(ds_hyps_el, [ds_refs_el]).score
    print(f"  → lb→en {dictsent_bleu:.2f} | en→lb {dictsent_bleu_el:.2f}\n")

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
        "val_exact_dictword":        round(dictword_pct, 1),
        "val_exact_dictword_en2lb":  round(dictword_pct_el, 1),
        "val_chrf_dictword":         round(dictword_chrf, 2),
        "val_chrf_dictword_en2lb":   round(dictword_chrf_el, 2),
        "val_bleu_dictsent":         round(dictsent_bleu, 3),
        "val_bleu_dictsent_en2lb":   round(dictsent_bleu_el, 3),
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
