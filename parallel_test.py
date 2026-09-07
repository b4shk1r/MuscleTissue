#!/usr/bin/env python3
"""
parallel_test.py — confirm N MuscleRegenEnv rollouts can run concurrently.

Each rollout is a separate OS process (rl_driver runs CC3D in-process, and CC3D
core + the model's module globals are singletons -> one sim per process). Each
process gets its own staged model copy (rl_stage.mkdtemp) and its own CC3D
settings dir, so there is nothing to clobber. This just proves that in practice.

    python parallel_test.py --workers 2 --steps 8 --quarter
"""
import argparse
import multiprocessing as mp
import time


def _rollout(worker_id: int, quarter: bool, steps: int, seed: int, q):
    try:
        from rl_env import MuscleRegenEnv, policy
        env = MuscleRegenEnv(quarter=quarter, seed=seed)
        obs, info = env.reset()
        n0 = len(obs)
        last = info
        for _ in range(steps):
            actions = {cid: policy(o, env.rng) for cid, o in obs.items()}
            obs, reward, done, trunc, last = env.step(actions)
            if done:
                break
        env.close()
        q.put((worker_id, "ok", n0, last["mcs"], last["raw"].get("n_ssc"),
               last["raw"].get("fiber_volume")))
    except Exception as e:
        import traceback
        q.put((worker_id, "err", repr(e), traceback.format_exc(), None, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--quarter", action="store_true")
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)   # macOS default; also what a cluster job would use
    q = mp.Queue()
    procs = [mp.Process(target=_rollout, args=(i, args.quarter, args.steps, 100 + i, q))
             for i in range(args.workers)]

    t0 = time.time()
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    print(f"\n{args.workers} workers, {args.steps} steps each, {time.time()-t0:.1f}s total\n")
    ok = 0
    for r in sorted(results):
        if r[1] == "ok":
            ok += 1
            print(f"  worker {r[0]}: OK  start_agents={r[2]} end_mcs={r[3]} "
                  f"n_ssc={r[4]} fiber_vol={r[5]}")
        else:
            print(f"  worker {r[0]}: ERROR {r[2]}\n{r[3]}")
    print(f"\n{'PASS' if ok == args.workers else 'FAIL'} — {ok}/{args.workers} workers finished clean")


if __name__ == "__main__":
    main()
