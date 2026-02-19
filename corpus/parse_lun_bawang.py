"""
Parse the Lun Bawang Bible PDF text into structured (book, chapter, verse, text) rows.

Input:  bible_raw.txt  (extracted from LunBawang-Bible.pdf via pdfminer)
Output: lun_bawang_verses.csv
"""

import re
import csv

# Mapping from Lun Bawang book names to standard USFM book codes
BOOK_MAP = {
    # Old Testament
    "Pudut":                  "GEN",
    "Pura'":                  "EXO",
    "Ukum":                   "LEV",
    "Lap":                    "NUM",
    "Bada' Ratnan Ukum":      "DEU",
    "Yusak":                  "JOS",
    "Lun Luk Nguyut":         "JDG",
    "Rut":                    "RUT",
    "Semuil Luk Pun-Pun":     "1SA",
    "Semuil Luk Kedueh":      "2SA",
    "Raca'-Raca' Luk Pun-Pun":"1KI",
    "Raca'-Raca' Luk Kedueh": "2KI",
    "Inul Luk Pun-Pun":       "1CH",
    "Inul Luk Kedueh":        "2CH",
    "Esra":                   "EZR",
    "Nehemia":                "NEH",
    "Ester":                  "EST",
    "Ayub":                   "JOB",
    "Nani Ubur":              "PSA",
    "Bale'":                  "PRO",
    "Lun Luk Mada'":          "ECC",
    "Nani Selimun":           "SNG",
    "Yesaya":                 "ISA",
    "Yeremia":                "JER",
    "Tido Yeremia":           "LAM",
    "Yehiskiel":              "EZK",
    "Daniel":                 "DAN",
    "Hosia":                  "HOS",
    "Yoel":                   "JOL",
    "Amos":                   "AMO",
    "Obadia":                 "OBA",
    "Yunus":                  "JON",
    "Mika":                   "MIC",
    "Nahm":                   "NAM",
    "Habakuk":                "HAB",
    "Sepania":                "ZEP",
    "Hagai":                  "HAG",
    "Sekaria":                "ZEC",
    "Malaki":                 "MAL",
    # New Testament
    "Matius":                 "MAT",
    "Markus":                 "MRK",
    "Lukas":                  "LUK",
    "Yahya":                  "JHN",
    "Kekamen Rasul-Rasul":    "ACT",
    "Rum":                    "ROM",
    "1 Korintus":             "1CO",
    "2 Korintus":             "2CO",
    "Galatia":                "GAL",
    "Epesus":                 "EPH",
    "Pilipi":                 "PHP",
    "Kolose":                 "COL",
    "1 Tesalonika":           "1TH",
    "2 Tesalonika":           "2TH",
    "1 Timotius":             "1TI",
    "2 Timotius":             "2TI",
    "Titus":                  "TIT",
    "Pilimon":                "PHM",
    "Iberani":                "HEB",
    "Yakub":                  "JAS",
    "1 Petrus":               "1PE",
    "2 Petrus":               "2PE",
    "1 Yahya":                "1JN",
    "2 Yahya":                "2JN",
    "3 Yahya":                "3JN",
    "Yahuda":                 "JUD",
    "Bala Luk Linio":         "REV",
}

# Lines to filter out entirely (checked after normalization, so plain apostrophes)
NOISE_PATTERNS = [
    re.compile(r"^BALA LUK DO'$"),
    re.compile(r"^Lun Bawang Bible$"),
    re.compile(r"^©\s*1982"),
    re.compile(r"^Lun Bawang - All Bible$"),
    re.compile(r"^Old Testament$"),
    re.compile(r"^New Testament$"),
    re.compile(r"^\d{1,4}$"),            # standalone page numbers
    re.compile(r"^\d{1,2}\.\d{3}"),     # TOC page numbers like "1.029"
]

def is_noise(line):
    return any(p.match(line) for p in NOISE_PATTERNS)

def parse_chapter_heading(line, book_names_sorted):
    """
    Returns (book_name, chapter_num) if line is a chapter heading, else None.
    Book names are tried longest-first to avoid prefix collisions.
    """
    for norm_name in book_names_sorted:
        if line.startswith(norm_name):
            rest = line[len(norm_name):].strip()
            if re.match(r'^\d+$', rest):
                return (norm_name, int(rest))
    return None

def parse_verse_start(line):
    """
    If line starts with a verse number, return (verse_num, text_remainder).
    Verse numbers go up to ~176 (Psalm 119).
    """
    m = re.match(r'^(\d{1,3})(.*)', line)
    if m:
        num = int(m.group(1))
        text = m.group(2).strip()
        if 1 <= num <= 200:
            return (num, text)
    return None

def clean_text(text):
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()

def normalize(text):
    """Normalize apostrophe variants to a plain ASCII apostrophe."""
    return text.replace('\u2019', "'").replace('\u2018', "'").replace('\x0c', '')


def parse(raw_path):
    with open(raw_path, encoding='utf-8') as f:
        raw_lines = [normalize(l).strip() for l in f.readlines()]

    # Normalize BOOK_MAP keys too so lookups work after normalization
    normalized_book_map = {normalize(k): v for k, v in BOOK_MAP.items()}
    normalized_book_names = {normalize(k): k for k in BOOK_MAP}

    # Sort book names longest-first to avoid prefix collisions
    book_names_sorted = sorted(normalized_book_map.keys(), key=len, reverse=True)

    verses = []
    current_book_name = None
    current_book_code = None
    current_chapter = None
    current_verse_num = None
    current_verse_text = []

    def flush_verse():
        if current_book_code and current_chapter and current_verse_num is not None:
            text = clean_text(' '.join(current_verse_text))
            if text:
                verses.append({
                    'book_code': current_book_code,
                    'book_name_lb': current_book_name,
                    'chapter': current_chapter,
                    'verse': current_verse_num,
                    'lun_bawang': text,
                })

    for line in raw_lines:
        if not line:
            continue
        if is_noise(line):
            continue

        # Check for chapter heading
        heading = parse_chapter_heading(line, book_names_sorted)
        if heading:
            flush_verse()
            current_verse_num = None
            current_verse_text = []
            norm_name, chapter_num = heading
            current_book_name = normalized_book_names[norm_name]
            current_book_code = normalized_book_map[norm_name]
            current_chapter = chapter_num
            continue

        # Check for standalone book name (running header, ignore)
        if line in normalized_book_map:
            continue

        # Check for verse start
        verse_start = parse_verse_start(line)
        if verse_start and current_chapter is not None:
            flush_verse()
            current_verse_num, text = verse_start
            current_verse_text = [text] if text else []
            continue

        # Continuation of current verse
        if current_verse_num is not None:
            current_verse_text.append(line)

    flush_verse()
    return verses


def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base, 'bible_raw.txt')
    out_path = os.path.join(base, 'lun_bawang_verses.csv')

    print(f"Parsing {raw_path}...")
    verses = parse(raw_path)
    print(f"Parsed {len(verses)} verses")

    # Count by book
    from collections import Counter
    book_counts = Counter(v['book_code'] for v in verses)
    print("\nVerses per book (sample):")
    for code, count in sorted(book_counts.items())[:10]:
        print(f"  {code}: {count}")
    print(f"  ... ({len(book_counts)} books total)")

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['book_code', 'book_name_lb', 'chapter', 'verse', 'lun_bawang'])
        writer.writeheader()
        writer.writerows(verses)

    print(f"\nSaved to {out_path}")

    # Show a sample
    print("\nSample verses:")
    for v in verses[:3]:
        print(f"  {v['book_code']} {v['chapter']}:{v['verse']} — {v['lun_bawang'][:80]}")


if __name__ == '__main__':
    main()
