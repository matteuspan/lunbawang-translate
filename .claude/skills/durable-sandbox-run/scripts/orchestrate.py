#!/usr/bin/env python3
"""Recycle-proof driver for a long job on an ephemeral sandbox.

Generalized template — see the durable-sandbox-run skill. The job runs as a
child process; this driver makes it survive container recycles and silent hangs:

  1. Ensures a clean resume point. If no DURABLE checkpoint exists yet, it
     (re)seeds a warm-start/resume pointer so a recycle before the first
     checkpoint cleanly restarts the short pre-checkpoint stretch instead of
     starting a brand-new run.
  2. Runs the job with a SHORT checkpoint interval so a resumable checkpoint
     lands quickly (before the container can recycle).
  3. Every COMMIT_EVERY seconds, validates + commits + force-pushes the resume
     STATE file to the branch, so progress survives reclamation. (Server-side
     checkpoints — e.g. model weights — are assumed durable already; this only
     protects the small local "where am I" pointer.)
  4. Watchdogs the job: if the step counter hasn't advanced in STALL_TIMEOUT,
     kill + relaunch (resumes from the last durable checkpoint).

Idempotent: exits immediately if a job process is already running, so a
wake-up tick landing on a still-alive container never double-launches.

======================================================================
FILL IN THESE FOUR PROJECT HOOKS, then the rest is generic:
======================================================================
"""
import json, subprocess, time, re
from pathlib import Path

ROOT   = Path(__file__).resolve().parent
BRANCH = "REPLACE_ME_work_branch"                 # branch the state file is pushed to
STATE  = ROOT / "REPLACE_ME_resume_state.json"    # small resume pointer (committed to git)
LOG    = ROOT / "REPLACE_ME_job.log"              # job's stdout/stderr (GITIGNORED)
TARGET_STEP  = 16000                              # stop when the job reaches this step
CHECKPOINT_EVERY = 100                            # how often the job saves a resumable checkpoint

# HOOK 1: the command that launches the job. Point stdout/stderr at LOG.
def job_command() -> list[str]:
    return ["python3", "train_translator.py", "--train",
            "--state-file", STATE.name,
            "--save-every", str(CHECKPOINT_EVERY), "--max-steps", str(TARGET_STEP)]

# HOOK 2: how to tell if the job is already running (used for idempotency + health).
JOB_PGREP = "train_translator.py --train"

# HOOK 3: parse the latest completed step from the tail of LOG (None if unknown).
def latest_step():
    try:
        nums = re.findall(r"step\s+(\d+)\s*\|", LOG.read_text()[-4000:])
        return int(nums[-1]) if nums else None
    except Exception:
        return None

# HOOK 4: re-seed a resume pointer when there's no durable checkpoint yet.
#         Write whatever your job's --state-file needs to warm-start cleanly.
WARM_START_POINTER = "REPLACE_ME_tinker://.../weights/checkpoint-N"   # training-weights path, NOT sampler_weights
def seed_state():
    STATE.write_text(json.dumps(
        {"warm_start_path": WARM_START_POINTER, "checkpoints": [], "steps": 0}, indent=2))

# ======================================================================
# Generic machinery below — usually no need to edit.
# ======================================================================
COMMIT_EVERY  = 180     # seconds between state pushes
STALL_TIMEOUT = 480     # kill+relaunch if the step counter hasn't advanced in this long
POLL          = 20      # watchdog poll interval (seconds)
MAX_RELAUNCH  = 40      # give up after this many consecutive relaunches (safety)
STATE_MSG     = "Long-run resume state"


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def job_running() -> bool:
    r = subprocess.run(["pgrep", "-f", JOB_PGREP], capture_output=True, text=True)
    return bool(r.stdout.strip())


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def seed_if_needed():
    s = load_state()
    if not s.get("checkpoints") and not s.get("training_state_path"):
        seed_state()
        print("[orch] no durable checkpoint — re-seeded warm-start", flush=True)
    else:
        print(f"[orch] resuming: steps={s.get('steps')} ckpts={len(s.get('checkpoints', []))}", flush=True)


def commit_state():
    if not STATE.exists():
        return
    try:
        json.loads(STATE.read_text())          # skip a torn mid-write read
    except Exception:
        print("[orch] state not valid JSON right now — skipping this commit", flush=True)
        return
    git("add", STATE.name)
    if not git("diff", "--cached", "--quiet").returncode:
        return                                  # nothing staged / no change
    subj = git("log", "-1", "--format=%s").stdout.strip()
    if subj.startswith(STATE_MSG):
        git("commit", "--amend", "--no-edit")   # keep one rolling state commit
    else:
        git("commit", "-m", STATE_MSG)
    rc = git("push", "--force-with-lease", "origin", BRANCH).returncode
    print(f"[orch] pushed state (rc={rc}) steps={load_state().get('steps')}", flush=True)


def run_once():
    """Launch the job and babysit until it exits or stalls. Returns
    'done' (reached target), 'exited' (crashed/ended), or 'stalled'."""
    seed_if_needed()
    with open(LOG, "a") as logf:
        proc = subprocess.Popen(job_command(), cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[orch] launched job pid {proc.pid}", flush=True)
        last_step, last_advance, last_commit = latest_step(), time.time(), time.time()
        while proc.poll() is None:
            time.sleep(POLL)
            now, st = time.time(), latest_step()
            if st is not None and st != last_step:
                last_step, last_advance = st, now
            if now - last_commit >= COMMIT_EVERY:
                commit_state(); last_commit = now
            if now - last_advance > STALL_TIMEOUT:
                print(f"[orch] STALL: no progress past step {last_step} for "
                      f"{STALL_TIMEOUT}s — killing to resume from last checkpoint", flush=True)
                proc.kill(); proc.wait(); commit_state()
                return "stalled"
    commit_state()
    print(f"[orch] job exited rc={proc.returncode} at step {latest_step()}", flush=True)
    return "done" if load_state().get("steps", 0) >= TARGET_STEP else "exited"


def main():
    if job_running():
        print("[orch] a job process is already running — exiting", flush=True)
        return
    for attempt in range(1, MAX_RELAUNCH + 1):
        if load_state().get("steps", 0) >= TARGET_STEP:
            print("[orch] reached target — done", flush=True)
            return
        print(f"[orch] run attempt {attempt}/{MAX_RELAUNCH}", flush=True)
        if run_once() == "done":
            print("[orch] DONE — reached target", flush=True)
            return
        time.sleep(10)   # brief pause before relaunch (resumes from last checkpoint)
    print(f"[orch] gave up after {MAX_RELAUNCH} relaunches", flush=True)


if __name__ == "__main__":
    main()
