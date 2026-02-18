"""
Build aux_corpus.csv from two local text files copied from the web.

Sources:
  borneodict.txt    — copied from borneodictionary.com/lun-bawang/
  longsemadoh.txt   — copied from longsemadoh.wordpress.com/2011/05/19/learn-lun-bawang-language/

Output: aux_corpus.csv
  Columns: source, lun_bawang, english, type   (type = "word" | "sentence")

Usage:
  python3.13 build_aux_corpus.py
  python3.13 build_aux_corpus.py --dry-run
"""

import csv
import re
import sys
import random
import argparse
from pathlib import Path

BASE_DIR    = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "aux_corpus.csv"
BORNEO_FILE = BASE_DIR / "borneodict.txt"
LONGS_FILE  = BASE_DIR / "longsemadoh.txt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("–—-:;,.'\"").strip()


# Broad English vocabulary for LB-vs-EN side detection on ambiguous lines.
_EN_VOCAB = {
    # Function words
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "did", "will", "would", "can", "could",
    "should", "may", "might", "shall", "must", "and", "or", "but", "if",
    "when", "where", "what", "who", "why", "how", "which", "that", "this",
    "not", "no", "yes", "i", "you", "he", "she", "we", "they", "it",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "in", "on", "at", "to", "of", "for", "with", "from", "by", "up",
    "out", "into", "there", "here", "then", "now", "just", "so", "very",
    "also", "some", "any", "all", "many", "few", "none", "one", "two",
    # Common English nouns
    "head", "hair", "nose", "ear", "eye", "eyes", "neck", "lip", "lips",
    "mouth", "tooth", "tongue", "chin", "hand", "elbow", "palm", "arm",
    "finger", "nail", "armpit", "back", "shoulder", "chest", "stomach",
    "knee", "leg", "toe", "foot", "ankle", "heart", "forehead",
    "mountain", "hill", "valley", "plain", "slope", "cliff", "waterfall",
    "river", "lake", "pond", "sea", "rock", "grass", "tree", "trees",
    "cloud", "clouds", "moon", "sun", "sky", "rain", "spring", "stream",
    "brook", "road", "path",
    "bird", "dog", "cat", "snake", "goat", "fish", "chicken", "pig",
    "buffalo", "cow", "eagle", "tiger", "lion", "rat", "monkey", "deer",
    "leech", "mosquito", "fly", "food", "water",
    "love", "joy", "praise", "pain", "life", "sound", "conversation",
    "beginning", "end", "cost", "twins", "orphan", "junction", "footprints",
    # Common English adjectives / verbs
    "eat", "drink", "sit", "stand", "walk", "run", "sleep", "speak", "tell",
    "see", "hear", "think", "sing", "call", "open", "play", "pray", "wash",
    "hide", "change", "meet",
    "beautiful", "ugly", "long", "short", "tall", "big", "small", "wide",
    "sharp", "blunt", "red", "yellow", "black", "white", "green", "blue",
    "hot", "cold", "angry", "happy", "sad", "tired", "hungry", "thirsty",
    "alone", "fierce", "kind", "patient", "shy", "brave", "wild", "tame",
    "proud", "humble", "deep", "shallow", "high", "good", "bad", "new",
    "lazy", "male", "female", "young", "old", "slow", "fast", "cheap",
    "sick", "dead", "empty", "full", "heavy", "light", "right", "left",
    "front", "behind", "above", "below", "inside", "outside",
}


def _en_score(text: str) -> int:
    """Count how many tokens in text match known English words."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return sum(1 for t in tokens if t in _EN_VOCAB)


def _classify(lb: str) -> str:
    """word or sentence based on LB word count."""
    return "sentence" if len(lb.split()) >= 3 else "word"


def _make_row(lb: str, eng: str, source: str) -> dict | None:
    lb  = _clean(lb)
    eng = _clean(eng)
    if not lb or not eng or lb == eng:
        return None
    return {"source": source, "lun_bawang": lb, "english": eng, "type": _classify(lb)}


# ── Parser 1: borneodict.txt ─────────────────────────────────────────────────
#
# Format (repeating blocks separated by blank lines):
#
#   LB_WORD
#   English: DEFINITION
#   Bahasa Malaysia: BM_DEFINITION
#
# The LB word is the non-blank, non-header line that precedes "English: ...".

def parse_borneodict(path: Path) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    seen: set[tuple[str, str]] = set()
    pending_lb = ""

    for raw in lines:
        line = raw.strip()

        if not line:
            pending_lb = ""
            continue

        if line.lower().startswith("english:"):
            eng = _clean(line[len("english:"):])
            if pending_lb and eng:
                row = _make_row(pending_lb, eng, "borneodictionary.com")
                if row and (row["lun_bawang"], row["english"]) not in seen:
                    seen.add((row["lun_bawang"], row["english"]))
                    rows.append(row)
            pending_lb = ""
            continue

        if line.lower().startswith("bahasa malaysia:") or line.lower() == "word list":
            continue

        # Any other non-empty line is a candidate LB headword
        pending_lb = line

    print(f"  borneodict.txt:   {len(rows)} entries")
    return rows


# ── Parser 2: longsemadoh.txt ─────────────────────────────────────────────────
#
# Multiple formats coexist in this file:
#
# A. Dialogue with LB(...):
#      JOHN: Hello!
#      LB( Ngadanku John)
#
# B. Alternating EN / -LB lines:
#      If you come, bring your dog.
#      -kudeng iko ame tunge, nguit uko' midih.
#
# C. Numbered sentences:  1. You are beautiful-Iko metaga
#
# D. Parenthetical LB:    you are very proud (Iko mesido')
#
# E. Separator formats:   EN – LB  |  EN-LB  |  LB-EN  |  EN=LB  |  LB=EN
#    Direction detected by counting English word matches on each side.
#
# Headers (ALL CAPS lines), section dividers (—–), and long compound lines
# are skipped.

def parse_longsemadoh(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    SOURCE = "longsemadoh.wordpress.com"

    def add(lb: str, eng: str):
        row = _make_row(lb, eng, SOURCE)
        if row and (row["lun_bawang"], row["english"]) not in seen:
            seen.add((row["lun_bawang"], row["english"]))
            rows.append(row)

    def oriented(left: str, right: str):
        """Return (lb, en) by scoring which side has more English words."""
        if _en_score(left) >= _en_score(right):
            # left is English side
            return right, left
        else:
            # right is English side
            return left, right

    prev_line = ""
    for raw in lines:
        line = raw.strip()

        # Skip blanks
        if not line:
            prev_line = ""
            continue

        # Skip pure section dividers (lines of — or – or - or =)
        if re.fullmatch(r"[—–\-=\s]+", line):
            prev_line = ""
            continue

        # Skip ALL-CAPS section headers (≥3 uppercase letters, no lowercase)
        if re.fullmatch(r"[A-Z][A-Z\s\(\)/]+", line) and len(line) > 3:
            prev_line = ""
            continue

        # ── Format A: LB(...) dialogue ──────────────────────────────────────
        m = re.match(r"^LB\s*\((.+)\)\s*$", line, re.IGNORECASE)
        if m:
            lb_text = _clean(m.group(1))
            if lb_text and prev_line:
                # Strip speaker label "NAME: " from the English line
                en_text = re.sub(r"^[A-Z]+\s*:\s*", "", prev_line)
                en_text = _clean(en_text.rstrip("."))
                add(lb_text, en_text)
            prev_line = ""
            continue

        # ── Format B: alternating EN / -LB ──────────────────────────────────
        if line.startswith("-") and len(line) > 2:
            lb_text = _clean(line[1:])
            if lb_text and prev_line and not prev_line.startswith("-"):
                en_text = _clean(prev_line.rstrip("."))
                add(lb_text, en_text)
            prev_line = ""
            continue

        # ── Format C: numbered sentences  "1. English-LB" ───────────────────
        m = re.match(r"^\d+\.?\s*(.+)", line)
        if m:
            rest = m.group(1).strip()
            # The separator is a hyphen; find split by scoring both sides
            if "-" in rest:
                # Try every hyphen as potential split point, pick best
                best = None
                best_score = -1
                for idx, ch in enumerate(rest):
                    if ch == "-" and idx > 0:
                        left  = _clean(rest[:idx])
                        right = _clean(rest[idx+1:])
                        if left and right:
                            score = abs(_en_score(left) - _en_score(right))
                            if score > best_score:
                                best_score = score
                                best = (left, right)
                if best:
                    lb, en = oriented(*best)
                    add(lb, en)
            prev_line = line
            continue

        # ── Format D: parenthetical LB  "English phrase (LB phrase)" ────────
        m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*\.?$", line)
        if m:
            candidate_en = _clean(m.group(1).rstrip("."))
            candidate_lb = _clean(m.group(2))
            if candidate_en and candidate_lb:
                # Use score to orient (the LB in parens usually has lower EN score)
                if _en_score(candidate_lb) < _en_score(candidate_en):
                    add(candidate_lb, candidate_en)
                else:
                    add(candidate_en, candidate_lb)
            prev_line = line
            continue

        # ── Format E: separator-based  (–, —, -, =) ─────────────────────────
        sep = None
        for candidate in [" – ", " — ", "–", "—"]:
            if candidate in line:
                sep = candidate
                break

        if sep:
            parts = line.split(sep, 1)
            if len(parts) == 2:
                left, right = _clean(parts[0]), _clean(parts[1])
                if left and right:
                    lb, en = oriented(left, right)
                    add(lb, en)
            prev_line = line
            continue

        if "=" in line:
            parts = line.split("=", 1)
            if len(parts) == 2:
                left, right = _clean(parts[0]), _clean(parts[1])
                if left and right:
                    lb, en = oriented(left, right)
                    add(lb, en)
            prev_line = line
            continue

        if "-" in line:
            # Skip multi-entry lines like "kick-nupak kicked-sinupak kicked-tinipak"
            if line.count("-") >= 3:
                prev_line = line
                continue
            parts = line.split("-", 1)
            if len(parts) == 2:
                left, right = _clean(parts[0]), _clean(parts[1])
                if left and right:
                    lb, en = oriented(left, right)
                    add(lb, en)
            prev_line = line
            continue

        prev_line = line

    n_words = sum(1 for r in rows if r["type"] == "word")
    n_sents = sum(1 for r in rows if r["type"] == "sentence")
    print(f"  longsemadoh.txt:  {len(rows)} entries ({n_words} words, {n_sents} sentences)")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def build(dry_run: bool = False):
    print("Building auxiliary corpus from local files…\n")

    for path, label in [(BORNEO_FILE, "borneodict.txt"), (LONGS_FILE, "longsemadoh.txt")]:
        if not path.exists():
            sys.exit(f"Missing file: {path}\nPlease add {label} to the project directory.")

    borneo_rows = parse_borneodict(BORNEO_FILE)
    longs_rows  = parse_longsemadoh(LONGS_FILE)
    all_rows    = borneo_rows + longs_rows

    n_words = sum(1 for r in all_rows if r["type"] == "word")
    n_sents = sum(1 for r in all_rows if r["type"] == "sentence")
    print(f"\nTotal: {len(all_rows)} entries  ({n_words} words, {n_sents} sentences)")

    if dry_run:
        sample = random.sample(all_rows, min(12, len(all_rows)))
        print("\nSample entries:")
        for r in sample:
            print(f"  [{r['type']:8s}] {r['lun_bawang']!r:35s} → {r['english']!r}")
        return

    fieldnames = ["source", "lun_bawang", "english", "type"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} entries → {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build auxiliary Lun Bawang corpus from local text files")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing file")
    args = parser.parse_args()
    build(dry_run=args.dry_run)
