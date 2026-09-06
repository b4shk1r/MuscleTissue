"""
rl_env.py — THE single file to interface with for RL on the muscle-regen model.

Everything CC3D-facing (`MuscleRegenEnv`, the driver plumbing in rl_driver.py) is
meant to stay as-is. The parts you own are between the `EDIT BELOW` / `EDIT ABOVE`
markers near the bottom:

    policy(obs, rng)         -> your action selection
    reward_fn(raw, prev_raw) -> your scalar reward from raw sim quantities

Quick start
-----------
    conda activate cc3d           # env with compucell3d + rtree (+ gymnasium)
    python rl_env.py --quarter --steps 50      # demo loop, random policy

Programmatic use
----------------
    from rl_env import MuscleRegenEnv
    env = MuscleRegenEnv(quarter=True, seed=0)
    obs, info = env.reset()
    for _ in range(env.max_mcs):
        actions = {cid: policy(o, env.rng) for cid, o in obs.items()}
        obs, reward, done, truncated, info = env.step(actions)
        if done:
            break
    env.close()

How it drives CC3D
------------------
Not via simservice (broken for this model on CC3D 4.9/4.10 — steppables see an
empty cell list). rl_driver.CC3DDriver runs the normal CML main loop but pauses
it before its `while` loop and steps it one MCS at a time in-process. So:
  * one sim per process (CC3D core + model globals are singletons); run parallel
    rollouts as separate processes.
  * no multiprocessing here -> import and use freely, no __main__ guard needed.
  * `seed=` seeds numpy + stdlib random in the model (via MUSCLEREGEN_SEED); the
    Potts core RNG needs <RandomSeed> in Simulation/MuscleRegen*.xml (see README).
  * no mid-episode snapshot/restore: reset() starts a fresh sim from the fixed
    initial injury.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    _Base = gym.Env
except ImportError:                     # gymnasium optional — env still works
    gym = None
    _Base = object

OBS_DIM   = 22
N_ACTIONS = 7
ACTION_NAMES = ["noop/migrate", "activate", "divide", "differentiate",
                "fuse_to_fiber", "fuse_to_myotube", "apoptose"]

# Observation layout (all roughly normalised to [0, 1]); see RLSteppable._build_obs
OBS_LABELS = [
    "x", "y", "activationState", "cellType",
    "time2activate", "time2divide", "time2diff", "numDiv",
    "HGF", "MMP", "TGF", "VEGF", "TNF", "IL10", "necrosisRemain",
    "nb_fiber", "nb_ecm", "nb_necrotic", "nb_capillary", "nb_ssc", "nb_macrophage",
    "mcs_frac",
]


class MuscleRegenEnv(_Base):
    """CC3D muscle-regeneration environment, one shared policy across all SSC agents.

    obs   : dict {cell_id(int) -> np.ndarray(22,) float32}
    action: dict {cell_id(int) -> int in [0, 7)}   (missing id => 0 / noop)
    """

    metadata = {"render_modes": []}

    def __init__(self, quarter: bool = False, seed: int | None = None):
        self._cc3d_name = "MuscleRegenRLQuarter.cc3d" if quarter else "MuscleRegenRL.cc3d"
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        self._drv = None
        self._prev_raw = {}
        self.max_mcs = 2688

        if gym is not None:
            self.observation_space = gym.spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32)
            self.action_space = gym.spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        if _Base is not object:
            super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

        self.close()
        from rl_driver import CC3DDriver
        self._drv = CC3DDriver(self._cc3d_name, seed=self.seed_value)
        out = self._drv.start()
        self.max_mcs = self._drv.num_steps or self.max_mcs
        self._prev_raw = out.get("raw", {})
        return _parse_obs(out), {"mcs": out.get("mcs", 0), "raw": self._prev_raw}

    def step(self, actions: dict):
        out = self._drv.step({"actions": {int(k): int(v) for k, v in actions.items()}})
        raw = out.get("raw", {})
        reward = float(reward_fn(raw, self._prev_raw))
        self._prev_raw = raw
        done = bool(out.get("done", False))
        info = {"mcs": out.get("mcs", self._drv.mcs), "raw": raw}
        return _parse_obs(out), reward, done, False, info

    def close(self):
        if self._drv is not None:
            self._drv.close()
            self._drv = None


def _parse_obs(out: dict) -> dict:
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in out.get("obs", {}).items()}


# ============================================================================
#  EDIT BELOW — this is the part you own
# ============================================================================

def reward_fn(raw: dict, prev_raw: dict) -> float:
    """Scalar reward from the raw quantities RLSteppable exports each step.

    `raw` keys: fiber_volume, initial_fiber_volume, newMyotube, n_ssc,
                n_myotube_immature, necrosisRemain
    `prev_raw` is the same dict from the previous step ({} on the first step).

    Default: per-step gain in fiber volume (fraction of initial) + a myotube bonus.
    """
    if not raw or not prev_raw:
        return 0.0
    init = raw.get("initial_fiber_volume", 1.0) or 1.0
    d_fiber = (raw.get("fiber_volume", 0.0) - prev_raw.get("fiber_volume", 0.0)) / init
    d_myotube = raw.get("newMyotube", 0) - prev_raw.get("newMyotube", 0)
    return d_fiber + 0.1 * d_myotube


def policy(obs_vec: np.ndarray, rng: np.random.Generator) -> int:
    """Pick one action (0..6) for a single agent from its 22-float observation.

    Replace this with your learned policy. Default: uniform random.
    """
    return int(rng.integers(N_ACTIONS))

# ============================================================================
#  EDIT ABOVE
# ============================================================================


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Demo rollout of MuscleRegenEnv")
    ap.add_argument("--quarter", action="store_true", help="use the quarter lattice")
    ap.add_argument("--steps", type=int, default=50, help="max env steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = MuscleRegenEnv(quarter=args.quarter, seed=args.seed)
    t0 = time.time()
    obs, info = env.reset()
    print(f"reset: {len(obs)} SSC agents, mcs={info['mcs']}, {time.time()-t0:.1f}s", flush=True)

    total = 0.0
    for t in range(args.steps):
        ts = time.time()
        actions = {cid: policy(o, env.rng) for cid, o in obs.items()}
        obs, reward, done, trunc, info = env.step(actions)
        total += reward
        print(f"  step {t:4d}  mcs={info['mcs']:5d}  agents={len(obs):4d}  "
              f"reward={reward:+.4f}  return={total:+.4f}  {time.time()-ts:.1f}s", flush=True)
        if done:
            print("  episode done")
            break
    env.close()
    print(f"total {time.time()-t0:.1f}s")
