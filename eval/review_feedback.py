"""
QC review script for user feedback collected via the web UI.

Reads feedback.db (or a CSV export) and outputs feedback_corpus.csv
ready for use as training data in train_translator.py.

Usage:
  python3.13 review_feedback.py                       # reads feedback.db
  python3.13 review_feedback.py --csv feedback.csv    # from exported CSV
  python3.13 review_feedback.py --dry-run             # stats only, no output file
"""

import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

ROOT_DIR        = Path(__file__).parent.parent
FEEDBACK_DB     = ROOT_DIR / "feedback.db"
OUTPUT_FILE     = ROOT_DIR / "feedback_corpus.csv"
IP_FLOOD_THRESH = 10   # flag IPs with more than this many entries


def load_from_db(path=FEEDBACK_DB):
    con = sqlite3.connect(path)
    rows = con.execute("SELECT * FROM feedback ORDER BY id").fetchall()
    con.close()
    cols = ["id", "created_at", "ip_address", "user_agent", "source_text",
            "direction", "checkpoint", "model_output", "rating", "correction"]
    return [dict(zip(cols, r)) for r in rows]


def load_from_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["rating"] = int(r["rating"])
            rows.append(r)
    return rows


def qc_and_summarise(rows, dry_run=False):
    total = len(rows)
    thumbs_up       = [r for r in rows if r["rating"] == 1]
    thumbs_down     = [r for r in rows if r["rating"] == -1]
    with_correction = [r for r in thumbs_down if (r.get("correction") or "").strip()]

    # IP flood detection
    ip_counts = Counter(r["ip_address"] for r in rows if r["ip_address"])
    flood_ips = {ip: c for ip, c in ip_counts.items() if c > IP_FLOOD_THRESH}

    # Suspicious user-agent detection
    suspect_ua = [r for r in rows if not (r.get("user_agent") or "").strip()]

    pct = lambda n: f"({n/total*100:.0f}%)" if total else ""
    print("Feedback summary")
    print("════════════════")
    print(f"Total entries:         {total}")
    print(f"  Thumbs up:           {len(thumbs_up)}  {pct(len(thumbs_up))}")
    print(f"  Thumbs down:         {len(thumbs_down)}  {pct(len(thumbs_down))}")
    print(f"    With correction:   {len(with_correction)}")
    print(f"    Without:           {len(thumbs_down) - len(with_correction)}")
    print()
    print("QC flags (review manually):")
    if flood_ips:
        for ip, count in flood_ips.items():
            print(f"  IP flood (>{IP_FLOOD_THRESH} entries): {count} entries from {ip}")
    else:
        print(f"  IP flood (>{IP_FLOOD_THRESH} from same IP): none")
    print(f"  Suspicious user-agent (empty): {len(suspect_ua)}")
    print()

    # Build usable training entries
    usable = []
    removed = 0

    # Thumbs-up: (source_text, model_output) oriented by direction
    for r in thumbs_up:
        src = r["source_text"].strip()
        out = (r["model_output"] or "").strip()
        if not src or not out:
            continue
        lb, eng = (src, out) if r["direction"] == "lb2en" else (out, src)
        type_ = "word" if len(src.split()) <= 2 else "sentence"
        usable.append({"source": "user_feedback", "lun_bawang": lb, "english": eng, "type": type_})

    # Thumbs-down with correction: (source_text, correction) oriented by direction
    for r in with_correction:
        src        = r["source_text"].strip()
        correction = r["correction"].strip()
        model_out  = (r["model_output"] or "").strip()
        if not correction:
            removed += 1; continue
        if correction == model_out:       # user didn't change anything
            removed += 1; continue
        if correction == src:             # user copied input
            removed += 1; continue
        lb, eng = (src, correction) if r["direction"] == "lb2en" else (correction, src)
        type_ = "word" if len(src.split()) <= 2 else "sentence"
        usable.append({"source": "user_feedback", "lun_bawang": lb, "english": eng, "type": type_})

    from_up   = len([r for r in thumbs_up if (r["source_text"].strip() and (r["model_output"] or "").strip())])
    from_corr = len(with_correction) - removed

    print("After QC filters:")
    print(f"  Usable for training: {len(usable)}  (removed {removed} no-ops/copies)")
    print(f"    From thumbs-up:    {from_up}")
    print(f"    From corrections:  {from_corr}")
    print()

    if not dry_run:
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["source", "lun_bawang", "english", "type"])
            w.writeheader()
            w.writerows(usable)
        print(f"Output → {OUTPUT_FILE} ({len(usable)} rows)")
    else:
        print("(dry-run: no file written)")


def main():
    parser = argparse.ArgumentParser(description="QC and prepare user feedback for training")
    parser.add_argument("--csv", metavar="FILE",
                        help="Read from exported CSV instead of feedback.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only, no output file written")
    args = parser.parse_args()

    if args.csv:
        rows = load_from_csv(args.csv)
    else:
        if not FEEDBACK_DB.exists():
            print(f"No feedback.db found at {FEEDBACK_DB}")
            print("Tip: run the server, collect some feedback, or use --csv to load an export.")
            return
        rows = load_from_db()

    if not rows:
        print("No feedback entries found.")
        return

    qc_and_summarise(rows, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
