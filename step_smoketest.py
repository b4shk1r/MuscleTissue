#!/usr/bin/env python3
"""
step_smoketest.py — prove CC3D can be stepped one MCS at a time for the RL loop.

CC3D owns the MCS loop; rl_driver.CC3DDriver pauses the CML main loop and steps it
by hand (simservice, the "official" way, is broken for this model — see rl_driver).
This checks that path on your build before the RL code is trusted.

    python step_smoketest.py --quarter               # plain quarter model
    python step_smoketest.py --rl --quarter          # RLSteppable model + obs/reward
    python step_smoketest.py --rl --quarter --steps 10

PASS  = it steps N times and prints plausible / rising cell counts.
FAIL  = import error, hang, crash, or 0 cells -> see docs/upgrade-plan.md.
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", action="store_true", help="use the RLSteppable model")
    ap.add_argument("--quarter", action="store_true", help="quarter lattice")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    if args.rl:
        cc3d = "MuscleRegenRLQuarter.cc3d" if args.quarter else "MuscleRegenRL.cc3d"
    else:
        cc3d = "MuscleRegenQuarter.cc3d" if args.quarter else "MuscleRegen.cc3d"

    try:
        from rl_driver import CC3DDriver
    except Exception as e:
        print(f"[smoke] FAIL: cannot import rl_driver ({e!r})")
        sys.exit(2)

    import os
    seed = int(os.environ.get("MUSCLEREGEN_SEED", "0"))
    print(f"[smoke] model: {cc3d}  seed: {seed}")
    t0 = time.time()
    d = CC3DDriver(cc3d, seed=seed)
    out = d.start()
    print(f"[smoke] started + MCS 0 in {time.time() - t0:.1f}s", flush=True)
    if args.rl:
        raw = out.get("raw", {})
        print(f"[smoke]   mcs=0 agents={len(out.get('obs', {}))} "
              f"fiber_vol={raw.get('fiber_volume')} n_ssc={raw.get('n_ssc')}", flush=True)

    for i in range(1, args.steps + 1):
        t = time.time()
        out = d.step({"actions": {}} if args.rl else {})
        line = f"[smoke] step {i:3d}/{args.steps}  {time.time() - t:6.2f}s"
        if args.rl:
            raw = out.get("raw", {})
            line += (f"  mcs={out.get('mcs')}  agents={len(out.get('obs', {}))}"
                     f"  fiber_vol={raw.get('fiber_volume')}  n_ssc={raw.get('n_ssc')}"
                     f"  done={out.get('done')}")
        print(line, flush=True)

    d.close()
    print(f"[smoke] PASS — stepped {args.steps} MCS in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
