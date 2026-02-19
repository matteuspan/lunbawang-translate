"""
Parse Appendix A.1 (Laba' fairy tale) from Mortensen (2021) PhD dissertation:
"The Kemaloh Lun Bawang Language of Borneo", University of Hawai'i.

Extracts parallel Lun Bawang ↔ English sentence pairs from the two-column
layout and writes mortensen_corpus.csv.

Usage:
  python3.13 parse_mortensen.py
  python3.13 parse_mortensen.py --dry-run    # print samples, no file written

Source: Mortensen (2021), UMI #10969. Used for non-commercial research purposes.
"""
import argparse, csv, re, unicodedata
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox

PDF_PATH    = Path(__file__).parent / "Mortensen_hawii_0085A_10969.pdf"
OUTPUT_FILE = Path(__file__).parent / "mortensen_corpus.csv"

# Physical pages to scan (0-indexed). Covers Appendix A.1 (fairy tale).
# Doc pages ~261-272 = physical pages 281-293. Widen by a few to be safe.
SCAN_PAGES = list(range(280, 297))

COL_SPLIT = 250   # x < this → left/LB column; x ≥ this → right/EN column
PARA_GAP  = 38    # y-distance larger than this separates paragraphs

# Patterns that identify noise (footnotes, cross-refs, page numbers)
_FOOTNOTE = re.compile(r'^\d+[A-Za-z*]|^\*+Sentence|^\*+See')
_BARE_NUM = re.compile(r'^\d+$')
# Inline footnote reference numbers embedded in LB text: "em,1 uten", "nepernah13 mala", "lek?3\""
_INLINE_FN = re.compile(r',(\d+)(\s)|(?<=[a-zA-Z\'\u00e0-\u00ff])\d+(?=[\s"\',.!?]|$)')
# Ligature normalization map (pdfminer often returns ligatures as single chars)
_LIGATURES = str.maketrans({'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl'})


def _norm(text: str) -> str:
    """Normalize ligatures and whitespace."""
    return text.translate(_LIGATURES).strip()


def _is_noise(text: str, y0: float, page_height: float = 792) -> bool:
    t = text.strip()
    if not t: return True
    if y0 < 50: return True                    # footer area (page number, etc.)
    if y0 > page_height - 55: return True      # header area
    if _BARE_NUM.match(t): return True         # standalone page number
    if _FOOTNOTE.match(t): return True         # numbered footnote or *Sentence ref
    return False


def _group_paragraphs(boxes: list) -> list:
    """Group (y0, text) list into paragraphs. Input must be sorted by y0 desc."""
    if not boxes: return []
    groups, cur = [], [boxes[0]]
    for b in boxes[1:]:
        if cur[-1][0] - b[0] > PARA_GAP:
            groups.append(cur)
            cur = [b]
        else:
            cur.append(b)
    groups.append(cur)
    return groups


def _join(boxes: list) -> str:
    """Join text boxes into a paragraph, de-hyphenating line-break hyphens."""
    parts = []
    for _, txt in boxes:
        txt = _norm(txt)
        if parts and parts[-1].endswith('-') and txt and txt[0].islower():
            # Line-break hyphen: join without space
            parts[-1] = parts[-1][:-1] + txt
        else:
            parts.append(txt)
    return ' '.join(parts)


def _extract_pairs(page) -> list:
    """Extract (lb_text, en_text) pairs from one two-column page."""
    left, right = [], []
    ph = page.height

    for elem in page:
        if not isinstance(elem, LTTextBox):
            continue
        text = elem.get_text().strip()
        if _is_noise(text, elem.y0, ph):
            continue
        if elem.x0 < COL_SPLIT:
            left.append((elem.y0, text))
        else:
            right.append((elem.y0, text))

    if not left or not right:
        return []

    left.sort(key=lambda b: -b[0])
    right.sort(key=lambda b: -b[0])

    lg = _group_paragraphs(left)
    rg = _group_paragraphs(right)

    pairs = []
    if len(lg) == len(rg):
        # Happy path: equal paragraph counts → zip by index
        for l, r in zip(lg, rg):
            pairs.append((_join(l), _join(r)))
    else:
        # Fallback: match each left group to nearest right group by y-centre
        used = set()
        for l in lg:
            lc = sum(y for y, _ in l) / len(l)
            best_j, best_d = -1, float('inf')
            for j, r in enumerate(rg):
                if j in used: continue
                rc = sum(y for y, _ in r) / len(r)
                d = abs(lc - rc)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d < 120:
                used.add(best_j)
                pairs.append((_join(l), _join(rg[best_j])))

    return pairs


def _looks_english(text: str) -> bool:
    """True if text is more likely English than Lun Bawang."""
    EN = {'the', 'and', 'said', 'he', 'she', 'it', 'was', 'were', 'is', 'are',
          'to', 'of', 'in', 'that', 'his', 'her', 'they', 'mouse', 'deer',
          'crocodile', 'cow', 'i', 'you', 'we', 'not', 'but', 'so', 'had'}
    LB = {'lek', 'peh', 'meh', 'ieh', 'neh', 'em', 'ki', 'dih', 'ineh',
          'tuk', 'pelanuk', 'buayeh', 'sapi', 'eca', 'keneh', 'kedawa', 'lek.'}
    words = set(re.findall(r"[a-z']+", text.lower()))
    return len(words & EN) >= len(words & LB)


def main():
    parser = argparse.ArgumentParser(description="Parse Mortensen (2021) appendix A.1")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print samples only, do not write file")
    args = parser.parse_args()

    print(f"Scanning pages {SCAN_PAGES[0]+1}–{SCAN_PAGES[-1]+1} of {PDF_PATH.name}…\n")

    all_pairs = []
    seen: set[tuple] = set()
    page_counts = {}

    for page in extract_pages(PDF_PATH, page_numbers=SCAN_PAGES):
        raw_pairs = _extract_pairs(page)
        page_new = 0
        for lb, en in raw_pairs:
            # Basic quality gates
            if not lb or not en or lb == en:
                continue
            if len(lb.split()) < 3 or len(en.split()) < 3:
                continue
            # Strip inline footnote reference numbers from LB text
            lb = _INLINE_FN.sub(lambda m: (',' + m.group(2)) if m.group(2) else '', lb).strip()
            # Verify orientation: left column should be LB, right should be EN
            # If they appear swapped (shouldn't happen with our column split), fix it
            if _looks_english(lb) and not _looks_english(en):
                lb, en = en, lb
            # If LB still looks English, it's footnote commentary continuation → discard
            if _looks_english(lb):
                continue
            # Dedup
            key = (lb[:60], en[:60])
            if key in seen:
                continue
            seen.add(key)
            all_pairs.append({
                'source':     'mortensen2021',
                'lun_bawang': lb,
                'english':    en,
                'type':       'sentence',
            })
            page_new += 1
        if page_new:
            page_counts[page_new] = page_counts.get(page_new, 0) + 1

    print(f"Extracted {len(all_pairs)} sentence pairs\n")

    print("Sample pairs:")
    step = max(1, len(all_pairs) // 10)
    for p in all_pairs[::step][:10]:
        print(f"  LB: {p['lun_bawang'][:90]}")
        print(f"  EN: {p['english'][:90]}")
        print()

    if args.dry_run:
        print("(dry-run: no file written)")
        return

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'lun_bawang', 'english', 'type'])
        w.writeheader()
        w.writerows(all_pairs)
    print(f"Wrote {len(all_pairs)} pairs → {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
