#!/usr/bin/env python3
"""
simservice smoke test — the GO / NO-GO gate for the RL approach.

CompuCell3D owns the MCS loop. The only supported way to drive it one step at a
time from outside (which every RL rollout needs) is CC3D's simservice API. This
script proves that works on THIS build before any RL code is trusted.

Run inside the cc3d environment / container:
    python simservice_smoketest.py                 # plain full model, 20 MCS
    python simservice_smoketest.py --rl --quarter  # RL model, quarter lattice
    xvfb-run -a python simservice_smoketest.py      # headless Linux

PASS  = it steps N times and prints rising/plausible cell counts.
FAIL  = import error, hang, or crash  -> the external-stepping approach needs
        rethinking; see docs/upgrade-plan.md §4.0.
"""
import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", action="store_true",
                    help="use the RLSteppable model + exchange actions/obs via pg")
    ap.add_argument("--quarter", action="store_true", help="quarter lattice")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    if args.rl:
        cc3d = "MuscleRegenRLQuarter.cc3d" if args.quarter else "MuscleRegenRL.cc3d"
    else:
        cc3d = "MuscleRegenQuarter.cc3d" if args.quarter else "MuscleRegen.cc3d"

    try:
        from cc3d.core.simservice.PyServiceCC3D import service_cc3d
    except Exception as e:
        print(f"[smoke] FAIL: cannot import simservice ({e!r})")
        print("[smoke] this CC3D build may not ship simservice — RL approach blocked.")
        sys.exit(2)

    # Run from a staged, .git-free copy (see rl_stage.py for why).
    from rl_stage import sim_paths
    sim_file, out_dir = sim_paths(cc3d)
    print(f"[smoke] staged sim file: {sim_file}")

    t0 = time.time()
    sim = service_cc3d(
        cc3d_sim_fname=sim_file,
        output_frequency=0,
        screenshot_output_frequency=0,
        output_dir=out_dir,
    )
    sim.run()
    sim.init()
    sim.start()
    print(f"[smoke] sim started in {time.time() - t0:.1f}s")

    for i in range(args.steps):
        if args.rl:
            sim.set_sim_input({"actions": {}})
        t = time.time()
        sim.step()
        line = f"[smoke] step {i + 1:3d}/{args.steps}  {time.time() - t:5.2f}s"
        if args.rl:
            out = sim.get_sim_output() or {}
            raw = out.get("raw", {})
            line += (f"  mcs={out.get('mcs', '?')}  agents={len(out.get('obs', {}))}"
                     f"  fiber_vol={raw.get('fiber_volume', '?')}"
                     f"  n_ssc={raw.get('n_ssc', '?')}  done={out.get('done', '?')}")
        print(line, flush=True)

    try:
        sim.stop()
    except Exception:
        pass
    print(f"[smoke] PASS — stepped {args.steps} MCS in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
