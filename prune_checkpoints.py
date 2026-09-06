#!/usr/bin/env python3.13
"""Prune Tinker checkpoints down to a hand-picked keep set, keeping JSON metadata.

Deletes weights from Tinker for every checkpoint NOT in KEEP, tagging the JSON
entry weights_deleted=true (metrics/history preserved). Deletes BOTH the stored
sampler_weights path and the paired weights/ path so training-weight artifacts
are freed too. Never touches a KEEP checkpoint's paths.

  python3.13 prune_checkpoints.py            # dry run (no deletion)
  python3.13 prune_checkpoints.py --execute  # actually delete + rewrite JSON
"""
import json, sys
from pathlib import Path

ROOT = Path("/home/user/lunbawang-translate")
KEEP = {   # state file -> set of steps to KEEP (weights stay on Tinker)
    "tinker_state_run1.json":       {7560, 15120},
    "tinker_state_v1.json":         {11500},
    "tinker_state_inkling.json":    {11500},
    "tinker_state_inkling_v2.json": {16000},
}
EXECUTE = "--execute" in sys.argv

# Collect keep + delete
keep_paths, delete_specs = set(), []   # delete_specs: (state_file, step, [paths])
for sf, keep_steps in KEEP.items():
    d = json.loads((ROOT/sf).read_text())
    for c in d["checkpoints"]:
        p = c["path"]
        both = {p, p.replace("/sampler_weights/", "/weights/")}
        if c["step"] in keep_steps:
            keep_paths |= both
        else:
            delete_specs.append((sf, c["step"], sorted(both)))

# ── SAFETY GUARDS ──
# 1. Nothing in the delete list may touch a keep path.
for sf, step, paths in delete_specs:
    for pth in paths:
        assert pth not in keep_paths, f"SAFETY ABORT: delete would hit keep path {pth}"
# 2. The live-served checkpoint must be a keep.
SERVED = "tinker://cfb19279-f23f-56d0-a789-548f94070088:train:0/sampler_weights/checkpoint-16000"
assert SERVED in keep_paths, "SAFETY ABORT: served checkpoint not in keep set!"

print(f"KEEP paths ({len(keep_paths)}):")
for p in sorted(keep_paths): print("   ", p)
print(f"\nDELETE: {len(delete_specs)} checkpoints × up to 2 variants each "
      f"= {sum(len(x[2]) for x in delete_specs)} delete calls")
from collections import Counter
by_file = Counter(sf for sf,_,_ in delete_specs)
for sf,n in by_file.items(): print(f"   {sf}: {n} checkpoints")

if not EXECUTE:
    print("\n(dry run — nothing deleted. Re-run with --execute.)")
    sys.exit(0)

import tinker
rc = tinker.ServiceClient().create_rest_client()
ok = err = 0
for sf, step, paths in delete_specs:
    for pth in paths:
        try:
            rc.delete_checkpoint_from_tinker_path(pth).result()
            ok += 1
        except Exception as e:
            err += 1
            print(f"   note step {step} {pth.split('/')[-2]}/{pth.split('/')[-1]}: {str(e)[:70]}")
print(f"\nTinker deletes: {ok} ok, {err} errors/absent")

# Rewrite JSON: tag deleted checkpoints, keep entries + metrics.
for sf, keep_steps in KEEP.items():
    d = json.loads((ROOT/sf).read_text())
    for c in d["checkpoints"]:
        if c["step"] not in keep_steps:
            c["weights_deleted"] = True
    (ROOT/sf).write_text(json.dumps(d, indent=2) + "\n")
    print(f"tagged {sf}: {sum(1 for c in d['checkpoints'] if c.get('weights_deleted'))} marked weights_deleted")
print("Done.")
