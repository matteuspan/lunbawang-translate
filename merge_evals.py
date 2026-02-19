"""
Merge all per-run JSONL files in eval_raw/ into a single eval_outputs.csv.

Usage:
  python3.13 merge_evals.py              # writes eval_outputs.csv
  python3.13 merge_evals.py --dry-run    # prints summary, no file written

The CSV has one row per translation with columns:
  timestamp         — ISO-8601 UTC time the eval run started
  model             — model name or checkpoint label (e.g. gpt-4o, checkpoint-8000)
  eval_set          — bible | dict | sentence
  input_lb          — Lun Bawang input text
  raw_output        — exactly what the model returned (before any post-processing)
  processed_output  — after stripping <think>…</think> reasoning blocks
  reference         — ground-truth English translation

Rows are sorted by: model, eval_set, then original file order.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

RAW_DIR     = Path(__file__).parent / "eval_raw"
OUTPUT_FILE = Path(__file__).parent / "eval_outputs.csv"
COLS        = ["timestamp", "model", "eval_set", "input_lb",
               "raw_output", "processed_output", "reference", "call_ms"]

EVAL_SET_ORDER = {"bible": 0, "dict": 1, "sentence": 2}


def load_all_jsonl():
    rows = []
    files = sorted(RAW_DIR.glob("*.jsonl")) if RAW_DIR.exists() else []
    if not files:
        return rows, []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows, files


def main():
    parser = argparse.ArgumentParser(description="Merge eval_raw/*.jsonl → eval_outputs.csv")
    parser.add_argument("--dry-run", action="store_true", help="Print summary only, no file written")
    args = parser.parse_args()

    rows, files = load_all_jsonl()

    if not rows:
        print("No JSONL files found in eval_raw/ — nothing to merge.")
        return

    # Summary
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    print(f"Found {len(files)} JSONL file(s), {len(rows)} total rows\n")
    print(f"{'Model':<30} {'bible':>6} {'dict':>6} {'sentence':>9} {'total':>6} {'avg ms/call':>12}")
    print("─" * 76)
    for model, model_rows in sorted(by_model.items()):
        counts = defaultdict(int)
        for r in model_rows:
            counts[r["eval_set"]] += 1
        times = [r["call_ms"] for r in model_rows if r.get("call_ms") is not None]
        avg_ms = f"{sum(times)/len(times):.0f}" if times else "n/a"
        print(f"  {model:<28} {counts['bible']:>6} {counts['dict']:>6} {counts['sentence']:>9} {len(model_rows):>6} {avg_ms:>12}")
    print()

    if args.dry_run:
        print("(dry-run: no file written)")
        return

    # Sort: model alphabetically, then eval_set in bible/dict/sentence order,
    # then preserve original order within each group
    rows.sort(key=lambda r: (r["model"], EVAL_SET_ORDER.get(r["eval_set"], 99)))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Written {len(rows)} rows → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
