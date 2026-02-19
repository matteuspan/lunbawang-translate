"""
Align the parsed Lun Bawang verses with the World English Bible (WEB)
to produce a parallel translation corpus.

Inputs:
  lun_bawang_verses.csv   (from parse_lun_bawang.py)
  web_english/            (unzipped eng-web_readaloud.zip)

Output:
  parallel_corpus.csv     (book_code, chapter, verse, lun_bawang, english)
"""

import csv
import os
import re
import glob
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))


def load_web_english(web_dir):
    """
    Parse all WEB chapter files and return a dict:
      (book_code, chapter, verse_num) -> english_text
    """
    verses = {}
    pattern = os.path.join(web_dir, 'eng-web_*_read.txt')
    files = sorted(glob.glob(pattern))

    for fpath in files:
        fname = os.path.basename(fpath)
        # Extract book code and chapter from filename, e.g. eng-web_002_GEN_01_read.txt
        m = re.match(r'eng-web_\d+_([A-Z0-9]+)_(\d+)_read\.txt', fname)
        if not m:
            continue
        book_code = m.group(1)
        chapter = int(m.group(2))

        with open(fpath, encoding='utf-8-sig') as f:
            lines = [l.rstrip('\n') for l in f.readlines()]

        # Skip header lines (book name, chapter heading) and empty lines
        verse_lines = [l for l in lines[2:] if l.strip()]

        for i, text in enumerate(verse_lines, start=1):
            verses[(book_code, chapter, i)] = text.strip()

    return verses


def build_corpus(lb_csv_path, web_dir, out_path):
    # Load English
    print("Loading World English Bible...")
    english = load_web_english(web_dir)
    print(f"  Loaded {len(english)} English verses")

    # Load Lun Bawang
    print("Loading Lun Bawang verses...")
    lb_verses = []
    with open(lb_csv_path, encoding='utf-8') as f:
        lb_verses = list(csv.DictReader(f))
    print(f"  Loaded {len(lb_verses)} Lun Bawang verses")

    # Align
    matched = []
    unmatched_lb = []

    for row in lb_verses:
        key = (row['book_code'], int(row['chapter']), int(row['verse']))
        eng_text = english.get(key)
        if eng_text:
            matched.append({
                'book_code':  row['book_code'],
                'book_lb':    row['book_name_lb'],
                'chapter':    row['chapter'],
                'verse':      row['verse'],
                'lun_bawang': row['lun_bawang'],
                'english':    eng_text,
            })
        else:
            unmatched_lb.append(row)

    print(f"\nMatched:   {len(matched)} verse pairs")
    print(f"Unmatched (LB has no English): {len(unmatched_lb)}")

    # Write output
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'book_code', 'book_lb', 'chapter', 'verse', 'lun_bawang', 'english'
        ])
        writer.writeheader()
        writer.writerows(matched)

    print(f"\nSaved parallel corpus to {out_path}")

    # Summary
    from collections import Counter
    book_counts = Counter(r['book_code'] for r in matched)
    print(f"\nBooks covered: {len(book_counts)}/66")

    # Show a few samples
    print("\nSample pairs:")
    for row in matched[:5]:
        print(f"  [{row['book_code']} {row['chapter']}:{row['verse']}]")
        print(f"    LB:  {row['lun_bawang'][:90]}")
        print(f"    EN:  {row['english'][:90]}")

    # Show any unmatched LB verses (for debugging)
    if unmatched_lb:
        print(f"\nSample unmatched LB verses:")
        for row in unmatched_lb[:5]:
            print(f"  {row['book_code']} {row['chapter']}:{row['verse']} — {row['lun_bawang'][:60]}")


if __name__ == '__main__':
    build_corpus(
        lb_csv_path=os.path.join(BASE, 'lun_bawang_verses.csv'),
        web_dir=os.path.join(BASE, 'web_english'),
        out_path=os.path.join(BASE, 'parallel_corpus.csv'),
    )
