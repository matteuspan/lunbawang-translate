#!/usr/bin/env python3.13
"""
Delete old checkpoints from Tinker and prune state files.

Keep rule: every 2000th step + any epoch-end checkpoint (has "epoch" key).
All other intermediate checkpoints are deleted from Tinker and removed from the state file.

Usage:
    python3.13 cleanup_checkpoints.py [--dry-run] [state_file ...]

    With no state_file args, cleans up all three standard state files:
        tinker_state.json, tinker_state_v1.json, tinker_state_run1.json

    With explicit paths, cleans only those files.

Examples:
    python3.13 cleanup_checkpoints.py --dry-run
    python3.13 cleanup_checkpoints.py tinker_state_v1.json
    TINKER_API_KEY=tml-... python3.13 cleanup_checkpoints.py
"""

import json
import sys
from pathlib import Path
import tinker

ROOT = Path(__file__).parent

DEFAULT_STATE_FILES = [
    ROOT / "tinker_state.json",
    ROOT / "tinker_state_v1.json",
    ROOT / "tinker_state_run1.json",
]


def should_keep(checkpoint: dict, final_step: int) -> bool:
    """Keep every-2000th step, epoch-end checkpoints, and the final step."""
    return checkpoint["step"] % 2000 == 0 or "epoch" in checkpoint or checkpoint["step"] == final_step


def cleanup(state_file: Path, rc, dry_run: bool = False):
    if not state_file.exists():
        print(f"\n{state_file.name}: not found, skipping")
        return

    data = json.loads(state_file.read_text())
    checkpoints = data.get("checkpoints", [])
    if not checkpoints:
        print(f"\n{state_file.name}: no checkpoints, skipping")
        return

    final_step = max(c["step"] for c in checkpoints)
    to_delete = [c for c in checkpoints if not should_keep(c, final_step)]
    to_keep   = [c for c in checkpoints if should_keep(c, final_step)]

    print(f"\n{state_file.name}: keeping {len(to_keep)}, deleting {len(to_delete)}")
    print(f"  Keep steps:  {[c['step'] for c in to_keep]}")
    if to_delete:
        print(f"  Delete steps: {[c['step'] for c in to_delete]}")

    for c in to_delete:
        path = c["path"]
        print(f"  DELETE step={c['step']}  {path}")
        if not dry_run:
            try:
                rc.delete_checkpoint_from_tinker_path(path).result()
                print(f"         ✓ deleted")
            except Exception as e:
                print(f"         ✗ ERROR: {e}")

    if not dry_run:
        data["checkpoints"] = to_keep
        state_file.write_text(json.dumps(data, indent=2) + "\n")
        print(f"  State file updated → {len(to_keep)} checkpoints remain")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    parser.add_argument("state_files", nargs="*", type=Path, help="State files to clean (default: all three)")
    args = parser.parse_args()

    state_files = args.state_files if args.state_files else DEFAULT_STATE_FILES

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()

    for sf in state_files:
        cleanup(sf, rc, dry_run=args.dry_run)

    if args.dry_run:
        print("\n(dry run — nothing deleted)")
    else:
        print("\nDone. Don't forget to commit the updated state files.")


if __name__ == "__main__":
    main()
