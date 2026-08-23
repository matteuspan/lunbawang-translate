---
name: durable-sandbox-run
description: >-
  Keep a long-running job (model fine-tune, big eval, batch OCR, any multi-hour
  compute) alive and resumable on an ephemeral, suspend-on-idle remote sandbox
  that can freeze or fully recycle underneath you. Use this whenever you're
  about to start — or are already babysitting — a training run / long job on a
  Claude Code web/remote container, or when a job "keeps dying", "loses
  progress on restart", "the container recycled", or you need to "drive a
  training run in chunks" over hours or days. Covers committing resume state to
  git for durability, a recycle-proof orchestrator with a hang watchdog, the
  two-layer keep-alive + auto-resume scheduling, the wake-up health/recovery
  procedure, and guarding against billing/API blocks. Reach for it before
  writing your own babysitting loop — the failure modes here are non-obvious
  and this captures them.
---

# Keeping a long job alive on an ephemeral sandbox

## The situation this is for

Remote/web Claude Code sandboxes are **ephemeral**. A long job (a fine-tune, a
big eval sweep, a batch pipeline) that runs for hours or days will hit two
distinct failure modes, and they need different defenses:

1. **Suspend-on-idle.** When no one is interacting with the session, the
   container is frozen. A client-side training loop that's making API calls is
   *not* enough to count as activity — the idle timer keys on session
   interaction, so the whole thing pauses. Left alone, throughput collapses to
   a trickle (it only runs during the brief windows something wakes it).
2. **Full recycle / reclamation.** After enough inactivity (or at the
   platform's discretion) the container is torn down and a fresh one is later
   provisioned. **Its disk reverts to the git snapshot from session start** —
   every uncommitted file is gone, and any local record of "where was I" is
   lost.

The core insight that makes long runs survivable: **the only durable things are
(a) what lives on a server you don't control the lifecycle of, and (b) what you
commit and push to git.** Everything else on the sandbox disk is disposable.

For a fine-tune specifically: the model checkpoints usually live server-side
(e.g. Tinker keeps them and they're addressable by URI), so *weights* are safe.
What is **not** safe is the little local state file that says "resume from
checkpoint N, training-state path X". If that only lives on disk, a recycle
loses your place and you restart from zero. So: **commit the resume-state file
to git, frequently.**

## The architecture (three parts)

Build these three things. They compose: the orchestrator does the work and the
durability; the schedulers wake the sandbox; the recovery procedure is what a
wake-up actually runs.

### 1. A recycle-proof orchestrator

A driver process that wraps the actual job and makes it durable and self-healing.
See `scripts/orchestrate.py` in this skill for a working, generalized template —
copy it into the repo and fill in the four project-specific hooks at the top
(how to launch the job, how to read the current progress step, where the resume
state lives, the max/target step). Its responsibilities:

- **Commit + push the resume state on a timer** (every ~180s). This is the
  whole ballgame for surviving recycles. Validate the file parses (skip a torn
  mid-write read), stage it, and if it changed, commit and **force-with-lease
  push** it to the work branch. Amend a single dedicated "run state" commit
  each time rather than piling up thousands of commits.
- **Watch for hangs, not just crashes.** Long jobs hang silently — a
  server-side checkpoint save can wedge with no error and no exit. Poll the
  job's log; if the progress step hasn't advanced in `STALL_TIMEOUT` (~8 min),
  **kill and relaunch** (it resumes from the last durable checkpoint). Do *not*
  key "healthy" on the last-seen step alone — a hang keeps the last step
  forever. Key on *advancement over time* (log mtime + step delta).
- **Re-seed a clean start point if there's no durable checkpoint yet.** If a
  recycle happens *before* the first checkpoint lands, don't spawn a
  fresh-from-scratch run — re-seed the same warm-start/resume pointer so the
  relaunch cleanly redoes the short pre-first-checkpoint stretch.
- **Be idempotent.** Exit immediately if a job process is already running, so a
  wake-up that lands on a still-alive container never double-launches.
- **Relaunch loop with a cap.** Wrap the run in a loop that relaunches on
  crash/stall up to some `MAX_RELAUNCH`, exiting when the target step is
  reached.

### 2. Two-layer scheduling (keep-alive + auto-resume)

You need *two* schedulers because the two failure modes need different wake
mechanisms. This is the part people get wrong.

| | keep-alive tick | auto-resume tick |
|---|---|---|
| **Tool** | in-session cron (`CronCreate`) | durable server-side routine (`create_trigger`) |
| **Interval** | sub-hourly (e.g. every 10 min) | hourly (server routines are often floored at hourly) |
| **Can wake a fully-recycled/cold container?** | **No** — it's in-session state and dies with the container | **Yes** — it's server-side and fires into the session even after a recycle |
| **Job** | keep an *alive* container awake (each tick is session activity that resets the idle timer) and do a fast health check | be the recovery net: on a cold container, re-fetch state from git and relaunch |
| **Survives recycle?** | No — **must be re-created after every recycle** | Yes |

The keep-alive is what actually buys you throughput (frequent activity ⇒ the
container stays awake ⇒ the job runs continuously instead of in stutters). But
because it dies on a full recycle, **every recovery must also re-create the
keep-alive cron.** If you notice throughput has quietly collapsed, check
`CronList` first — a silently-dead keep-alive is the usual culprit.

Both ticks run the same recovery logic (below) and reply with a one-line status
so the human driving "in chunks" can see progress without noise. See
`references/tick-prompts.md` for ready-to-use tick prompt bodies.

### 3. The wake-up health / recovery procedure

Every tick (from either scheduler) runs this. The detection logic is the
subtle part — see `references/tick-prompts.md` for the exact commands.

1. **Classify the container.** Compute: is a job process alive (`pgrep`)? how
   stale is the log (`now - mtime`)? what's the latest step? **HEALTHY** =
   process alive AND log advanced recently (idle below threshold). **DOWN** =
   no process, OR log stale beyond the watchdog window, OR a fresh container
   (log missing / reverted to a stale snapshot step).
2. **If HEALTHY:** reply one line (`step N/target (idle Xs)`) and stop. Don't
   touch git, don't preempt the orchestrator's own watchdog.
3. **If DOWN:** re-fetch durability, then relaunch:
   `git fetch origin <branch> && git reset --hard origin/<branch>` (this pulls
   the committed resume state back onto the fresh disk), remove the stale log,
   read the resume step, and relaunch the orchestrator. Re-create the
   keep-alive cron if it's gone.
4. **If the target step is reached:** the run is done — do the final
   selection/handoff and delete the schedulers.

## Guard against external blockers (billing / API 402)

An out-of-credit or billing-blocked API turns "relaunch on failure" into a
**crash loop** that burns through relaunch attempts and spams confusing status.
Before a DOWN-path relaunch, **probe the API cheaply** (one tiny request against
a known-good model/endpoint). If it comes back with a hard block (e.g. HTTP
402 / "billing"), **do not relaunch** — report `PAUSED: <reason>` and stop.
When the block clears, the next tick's probe passes and it resumes on its own.
Bake this probe into the DOWN branch of both tick prompts.

## Gotchas learned the hard way

- **"Still on the same step" ≠ healthy.** A hung save sits on the last step
  indefinitely. Always measure *advancement*, via log mtime and step delta —
  never `pgrep` alone.
- **The keep-alive silently dies on recycle.** In-session crons don't survive a
  full container reset. Throughput quietly drops from (say) ~600 steps/hr to
  ~150. Re-create it on every recovery; check `CronList` when throughput looks
  wrong.
- **Warm-start wants the *training* weights path, not the sampler path.** If
  you warm-start/resume from a checkpoint URI, use the training-weights variant
  (`.../weights/checkpoint-N`), not the serving/sampler variant
  (`.../sampler_weights/checkpoint-N`) — the latter is rejected by the loader.
- **Commit the state file, not the log.** Gitignore the actively-written log
  (it races the per-turn git hook and is huge); commit only the small resume
  state. Push with `--force-with-lease` and amend one "run state" commit.
- **Server routines are hourly-floored; in-session crons aren't.** That's
  exactly why you need both — hourly is too coarse to keep a container awake,
  but only the server routine can wake a cold one.
- **Disk is a fixed per-session allowance.** "No space left" with low `df`
  usage means the allowance is spent, not that the machine is broken. Deletes
  still succeed while writes fail — clear build artifacts/caches/stale clones
  and the freed space is immediately writable.
- **Managing server routines may need a permission grant.** Deleting/updating a
  durable routine can hit a tool-approval wall in a background session; if the
  inline approve doesn't register, switch the session to a bypass/don't-ask
  permission mode (or delete the routine from the web Routines UI), then remove
  it in one call. Always delete the schedulers when the run finishes so they
  don't keep firing.

## Worked example (what this was distilled from)

A dictionary fine-tune warm-started from a prior checkpoint, run to 16,000
steps over ~1.5 days on a suspend-on-idle web sandbox that recycled roughly
hourly. `orchestrate.py` committed the resume state every 3 min and force-pushed
it; an 8-min watchdog killed/relaunched on the one 9-hour silent save-hang; an
hourly server routine re-fetched state and relaunched after each recycle; a
10-min in-session cron kept live containers awake (~4× throughput vs. hourly
alone) and was re-created on each recovery; a 402 probe paused the loop cleanly
through a mid-run billing lapse and auto-resumed when it cleared. No training
progress was ever lost across many recycles, because the resume pointer was
always one `git reset --hard` away.
