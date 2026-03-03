"""
Build a conversational Bible subcorpus using the Translation for Translators (T4T).

Filter: only verses that contain direct speech (quotation marks in WEB text) and
are 5–40 words long. This aggressively selects dialogue content and drops genealogies,
legal codes, poetry, and narrative description.

The English side uses T4T (modern, translator-oriented paraphrase) rather than WEB,
so the new subcorpus adds stylistic diversity without duplicating the main corpus.

Inputs:
  corpus/parallel_corpus.csv      (LB + WEB, already aligned — used for filtering)
  corpus/t4t_english/             (extracted eng-t4t_readaloud.zip from ebible.org)

Output:
  corpus/conversational_corpus.csv  (source, lun_bawang, english, type)

Usage:
  python3.13 corpus/build_conversational_corpus.py [--dry-run]
"""

import csv
import glob
import os
import re
import argparse

BASE = os.path.dirname(os.path.abspath(__file__))
PARALLEL_CSV = os.path.join(BASE, "parallel_corpus.csv")
T4T_DIR      = os.path.join(BASE, "t4t_english")
OUT_CSV      = os.path.join(BASE, "conversational_corpus.csv")

MIN_WORDS = 5
MAX_WORDS = 40


def load_t4t(t4t_dir):
    """
    Parse all T4T chapter files and return a dict:
      (book_code, chapter, verse_num) -> t4t_text
    Same format as WEB readaloud files.
    """
    verses = {}
    pattern = os.path.join(t4t_dir, "eng-t4t_*_read.txt")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        fname = os.path.basename(fpath)
        m = re.match(r"eng-t4t_\d+_([A-Z0-9]+)_(\d+)_read\.txt", fname)
        if not m:
            continue
        book_code = m.group(1)
        chapter = int(m.group(2))

        with open(fpath, encoding="utf-8-sig") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]

        # Skip book name / chapter heading lines and empty lines (same as WEB parser)
        verse_lines = [l for l in lines[2:] if l.strip()]

        for i, text in enumerate(verse_lines, start=1):
            verses[(book_code, chapter, i)] = text.strip()

    return verses


def is_conversational(english_web: str) -> bool:
    """Return True if the WEB verse passes the conversational filter."""
    # WEB uses curly double quotes (U+201C / U+201D) for direct speech
    if '\u201c' not in english_web:
        return False
    word_count = len(english_web.split())
    return MIN_WORDS <= word_count <= MAX_WORDS


def build(dry_run: bool = False):
    if not os.path.exists(PARALLEL_CSV):
        print(f"ERROR: {PARALLEL_CSV} not found. Run build_parallel_corpus.py first.")
        return

    if not os.path.exists(T4T_DIR):
        print(f"ERROR: {T4T_DIR} not found.")
        print("  Download: curl -L https://ebible.org/Scriptures/eng-t4t_readaloud.zip -o corpus/t4t_english.zip")
        print("  Extract:  unzip corpus/t4t_english.zip -d corpus/t4t_english/")
        return

    print("Loading T4T translation…")
    t4t = load_t4t(T4T_DIR)
    print(f"  {len(t4t)} T4T verses loaded")

    print("Loading parallel corpus (WEB + LB)…")
    with open(PARALLEL_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} verse pairs")

    print(f"\nApplying conversational filter (has quotes, {MIN_WORDS}–{MAX_WORDS} words)…")
    passed = 0
    no_t4t = 0
    results = []

    for row in rows:
        if not is_conversational(row["english"]):
            continue
        passed += 1

        key = (row["book_code"], int(row["chapter"]), int(row["verse"]))
        t4t_text = t4t.get(key)
        if t4t_text is None:
            no_t4t += 1
            continue

        results.append({
            "source":     "t4t",
            "lun_bawang": row["lun_bawang"],
            "english":    t4t_text,
            "type":       "sentence",
        })

    print(f"  {passed} verses passed filter")
    print(f"  {no_t4t} skipped (no T4T match)")
    print(f"  {len(results)} rows in output")

    # Show a few samples
    print("\nSample pairs:")
    for r in results[:5]:
        print(f"  LB: {r['lun_bawang'][:90]}")
        print(f"  T4T: {r['english'][:90]}")
        print()

    # Book distribution
    from collections import Counter
    book_counts = Counter(
        rows[i]["book_code"]
        for i, row in enumerate(rows)
        if is_conversational(row["english"]) and t4t.get((row["book_code"], int(row["chapter"]), int(row["verse"])))
    )
    top_books = book_counts.most_common(10)
    print("Top books by verse count:")
    for book, count in top_books:
        print(f"  {book}: {count}")

    if dry_run:
        print("\n(dry run — not writing output)")
        return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "lun_bawang", "english", "type"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {OUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing output")
    args = parser.parse_args()
    build(dry_run=args.dry_run)
