#!/usr/bin/env python3
"""Recycle-proof driver for the dictionary retrain on an ephemeral host.

The training loop runs client-side, but this container can be reclaimed at any
time. Tinker checkpoints are server-side and durable; the only local record of
where to resume is tinker_state_inkling_v2.json. So this driver:

  1. Ensures a clean start point. If no DURABLE checkpoint exists yet (no
     checkpoints and no training_state_path in the state file), it (re)seeds a
     warm-start from checkpoint-11500 — so a recycle before the first checkpoint
     just cleanly restarts the warm-start instead of spawning a fresh-from-base
     experiment. Once a checkpoint exists, it leaves the state alone so
     train_translator.py full-resumes from the last training_state_path.
  2. Runs train_translator.py with a short --save-every so a resumable
     checkpoint lands quickly (before the container can recycle).
  3. Every COMMIT_EVERY seconds, validates + commits + force-pushes the state
     file to the branch (amending one "Retrain v2 run state" commit on top of
     the recipe), so progress survives reclamation.

Idempotent: exits immediately if a training subprocess is already running, so a
resume tick that lands on a still-alive container never double-launches.

Run:  python3 orchestrate_retrain.py   (TINKER_API_KEY must be set)
"""
import json
import subprocess
import time
from pathlib import Path

ROOT   = Path(__file__).parent
STATE  = ROOT / "tinker_state_inkling_v2.json"
LOG    = ROOT / "trainv2.log"          # gitignored
BRANCH = "claude/newline-handling-logic-lrjl58"
WEIGHTS_11500 = "tinker://3a051457-f00d-5e81-9f89-5f9cba6b6820:train:0/weights/checkpoint-11500"
BASE_MODEL   = "thinkingmachines/Inkling-Small"
MAX_STEPS     = 16000
SAVE_EVERY    = 100
COMMIT_EVERY  = 180     # seconds between state pushes
STALL_TIMEOUT = 480     # kill+relaunch if the step counter hasn't advanced in this long
POLL          = 20      # watchdog poll interval (seconds)
MAX_RELAUNCH  = 40      # give up after this many consecutive relaunches (safety)
STATE_MSG     = "Retrain v2 run state (warm-start from checkpoint-11500)"


def git(*args, check=False):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=check)


def training_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "train_translator.py --train"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def seed_if_needed():
    s = load_state()
    if not s.get("checkpoints") and not s.get("training_state_path"):
        STATE.write_text(json.dumps(
            {"warm_start_path": WEIGHTS_11500, "checkpoints": [], "steps": 0}, indent=2))
        print("[orch] no durable checkpoint — seeded warm-start from checkpoint-11500", flush=True)
    else:
        print(f"[orch] resuming: steps={s.get('steps')} ckpts={len(s.get('checkpoints', []))}", flush=True)


def commit_state():
    if not STATE.exists():
        return
    try:
        json.loads(STATE.read_text())         # skip a torn mid-write read
    except Exception:
        print("[orch] state not valid JSON right now — skipping this commit", flush=True)
        return
    git("add", STATE.name)
    if not git("diff", "--cached", "--quiet").returncode:
        return                                 # nothing staged / no change
    subj = git("log", "-1", "--format=%s").stdout.strip()
    if subj.startswith("Retrain v2 run state"):
        git("commit", "--amend", "--no-edit")
    else:
        git("commit", "-m", STATE_MSG + "\n\nLive resume file, amended as checkpoints land.")
    rc = git("push", "--force-with-lease", "origin", BRANCH).returncode
    s = load_state()
    print(f"[orch] pushed state (rc={rc}) steps={s.get('steps')} ckpts={len(s.get('checkpoints', []))}", flush=True)


import re as _re

def latest_step():
    """Highest 'step N |' seen in the training log, or None."""
    try:
        nums = _re.findall(r"step\s+(\d+)\s*\|", LOG.read_text()[-4000:])
        return int(nums[-1]) if nums else None
    except Exception:
        return None


def run_once():
    """Launch training and babysit it until it exits or stalls. Returns
    'done' (reached max-steps), 'exited' (crashed/ended — relaunch), or
    'stalled' (watchdog killed a hung process — relaunch)."""
    seed_if_needed()
    with open(LOG, "a") as logf:
        proc = subprocess.Popen(
            ["python3", "train_translator.py", "--train",
             "--base-model", BASE_MODEL, "--state-file", STATE.name,
             "--save-every", str(SAVE_EVERY), "--max-steps", str(MAX_STEPS)],
            cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[orch] launched training pid {proc.pid}", flush=True)
        last_step, last_advance, last_commit = latest_step(), time.time(), time.time()
        while proc.poll() is None:
            time.sleep(POLL)
            now = time.time()
            st = latest_step()
            if st is not None and st != last_step:
                last_step, last_advance = st, now
            if now - last_commit >= COMMIT_EVERY:
                commit_state(); last_commit = now
            if now - last_advance > STALL_TIMEOUT:
                print(f"[orch] STALL: no progress past step {last_step} for "
                      f"{STALL_TIMEOUT}s — killing to resume from last checkpoint", flush=True)
                proc.kill(); proc.wait()
                commit_state()
                return "stalled"
    commit_state()
    print(f"[orch] training exited rc={proc.returncode} at step {latest_step()}", flush=True)
    return "done" if load_state().get("steps", 0) >= MAX_STEPS else "exited"


def main():
    if training_running():
        print("[orch] a training subprocess is already running — exiting", flush=True)
        return
    for attempt in range(1, MAX_RELAUNCH + 1):
        if load_state().get("steps", 0) >= MAX_STEPS:
            print("[orch] reached max-steps — done", flush=True)
            return
        print(f"[orch] run attempt {attempt}/{MAX_RELAUNCH}", flush=True)
        outcome = run_once()
        if outcome == "done":
            print("[orch] DONE — reached max-steps", flush=True)
            return
        time.sleep(10)   # brief pause before relaunch (resumes from last checkpoint)
    print(f"[orch] gave up after {MAX_RELAUNCH} relaunches", flush=True)


if __name__ == "__main__":
    main()
