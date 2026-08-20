"""
Build dictionary_corpus.csv from the OCR'd dictionary entries.

Stage C of the dictionary pipeline: flattens the structured per-page entries
produced by ocr_dictionary.py into the same four-column shape as the rest of
the corpus, so it can be concatenated with aux_corpus.csv etc.

  Columns: source, lun_bawang, english, type

Flattening rules
----------------
* One row per (head-word, sense) — exactly how aux_corpus already lists a
  head-word with several senses as several rows.
* `type` carries the quality flag (no extra columns, stays "as is"):
    - "word"        clean translation equivalents (kind == "equivalent"),
                    matching the existing lexical rows in aux_corpus.
    - "definition"  descriptive glosses (kind == "definition", e.g. "can be
                    channeled or drained"). Kept so nothing is lost, but tagged
                    so training can down-weight or drop them from en->lb.
    - "sentence"    an entry's example sentences — a full parallel Lun Bawang /
                    English pair, the highest-value data in the book — matching
                    the existing "sentence" rows in the corpus.
    - "redirect"    a 'see X' pointer with no meaning of its own (e.g. "aa" ->
                    "see naa"). Kept for completeness but not a training pair;
                    --drop-redirects omits them.
* "also X" variant forms become extra rows with the same gloss (--no-variants
  to skip). Cross-references stay as provenance only, never as pairs — the root
  they point to has its own entry elsewhere in the dictionary.
* Apostrophe variants are normalised to a plain ASCII apostrophe, matching
  parse_lun_bawang.normalize(), so joins against the rest of the corpus line up.
* Rows are de-duplicated on (lun_bawang, english); --dedupe-against skips pairs
  already present in an existing corpus CSV.

Usage
-----
  python3 build_dictionary_corpus.py                       # entries -> csv
  python3 build_dictionary_corpus.py --dry-run             # stats only
  python3 build_dictionary_corpus.py --dedupe-against aux_corpus.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent


def normalize(text: str) -> str:
    """Plain ASCII apostrophes + collapsed whitespace (cf. parse_lun_bawang)."""
    text = text.replace("’", "'").replace("‘", "'").replace("\x0c", "")
    return re.sub(r"\s+", " ", text).strip()


def clean_gloss(gloss: str) -> str:
    """Tidy an English gloss: normalise, drop a single trailing period."""
    g = normalize(gloss)
    return g[:-1].rstrip() if g.endswith(".") and not g.endswith("..") else g


# A gloss that only points elsewhere ("see naa", "cf. X") — caught even if the
# model failed to tag kind="redirect".
_REDIRECT_RE = re.compile(r"^(see|cf\.?|same as|compare)\b", re.IGNORECASE)


def sense_type(kind: str, gloss: str) -> str:
    """Map the model's sense kind + gloss text onto the corpus `type`:
      - "redirect"   'see X' pointer, no meaning of its own (not a training pair)
      - "definition" descriptive gloss (lb->en-leaning)
      - "word"       clean bidirectional equivalent

    Redirect is decided by the gloss TEXT, not the model's `kind`: the model
    sometimes over-tags redirect on a real gloss that merely carries a
    cross-reference (e.g. 'will be fenced.'), and we must keep those. So a stray
    kind=='redirect' on non-pointer text falls back to "definition".
    """
    if _REDIRECT_RE.match(gloss):
        return "redirect"
    return "word" if kind == "equivalent" else "definition"


def iter_entries(jsonl_path: Path):
    """Yield entry dicts from the per-page JSONL, tolerating blank lines."""
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        page = json.loads(line)
        for entry in page.get("entries", []):
            yield entry


def build_rows(jsonl_path: Path, source: str, include_variants: bool,
               drop_redirects: bool = False) -> list:
    rows = []
    for entry in iter_entries(jsonl_path):
        headword = normalize(entry.get("headword", ""))
        if not headword:
            continue
        surface_forms = [headword]
        if include_variants:
            surface_forms += [normalize(v) for v in entry.get("variants", []) if normalize(v)]

        for sense in entry.get("senses", []):
            gloss = clean_gloss(sense.get("gloss", ""))
            if not gloss:
                continue
            typ = sense_type(sense.get("kind", "equivalent"), gloss)
            if typ == "redirect" and drop_redirects:
                continue
            for lb in surface_forms:
                rows.append({"source": source, "lun_bawang": lb,
                             "english": gloss, "type": typ})

        # Example sentences are full parallel pairs -> type "sentence" (the
        # existing corpus convention). They are not multiplied over variants.
        for ex in entry.get("examples", []):
            lb = normalize(ex.get("lun_bawang", ""))
            en = normalize(ex.get("english", ""))
            if lb and en:
                rows.append({"source": source, "lun_bawang": lb,
                             "english": en, "type": "sentence"})
    return rows


def load_existing_pairs(csv_path: Path) -> set:
    """(lun_bawang, english) pairs already present in a corpus CSV, normalised."""
    pairs = set()
    if not csv_path.exists():
        return pairs
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lb = normalize(r.get("lun_bawang", "")).lower()
            en = normalize(r.get("english", "")).lower()
            if lb and en:
                pairs.add((lb, en))
    return pairs


def dedupe(rows: list, seen: set) -> list:
    """Drop rows whose (lun_bawang, english) key is already in `seen`; mutates seen."""
    out = []
    for row in rows:
        key = (row["lun_bawang"].lower(), row["english"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entries", type=Path, default=BASE_DIR / "dictionary_entries.jsonl",
                    help="Input JSONL from ocr_dictionary.py.")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "dictionary_corpus.csv",
                    help="Output CSV.")
    ap.add_argument("--source", default="lb_en_dictionary",
                    help="Value for the `source` column (rename to the dictionary's citation).")
    ap.add_argument("--no-variants", action="store_true",
                    help="Do not emit rows for 'also X' variant surface forms.")
    ap.add_argument("--drop-redirects", action="store_true",
                    help="Omit 'see X' redirect rows entirely (default: keep them tagged type=redirect).")
    ap.add_argument("--dedupe-against", type=Path, action="append", default=[],
                    metavar="CSV", help="Skip pairs already in this corpus CSV (repeatable).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats without writing the CSV.")
    args = ap.parse_args()

    if not args.entries.exists():
        raise SystemExit(f"No entries file: {args.entries} (run ocr_dictionary.py first)")

    rows = build_rows(args.entries, args.source, include_variants=not args.no_variants,
                      drop_redirects=args.drop_redirects)

    seen = set()
    for existing in args.dedupe_against:
        seen |= load_existing_pairs(existing)
    before = len(rows)
    rows = dedupe(rows, seen)

    from collections import Counter
    by_type = Counter(r["type"] for r in rows)
    print(f"Entries flattened -> {before} rows; {before - len(rows)} dropped as duplicates.")
    print(f"Kept {len(rows)} rows: " + ", ".join(f"{t}={n}" for t, n in sorted(by_type.items())))
    if rows:
        print("Sample:")
        for r in rows[:6]:
            print(f"  {r['lun_bawang']:<18} {r['english']:<40} [{r['type']}]")

    if args.dry_run:
        print("(dry run — nothing written)")
        return

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "lun_bawang", "english", "type"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
