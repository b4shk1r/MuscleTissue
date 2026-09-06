"""
rl_env.py — THE single file to interface with for RL on the muscle-regen model.

Everything CC3D-facing (the MuscleRegenEnv class, the simservice plumbing) is
meant to stay as-is. The parts you own are near the bottom, between the
`EDIT BELOW` / `EDIT ABOVE` markers:

    policy(obs, rng)         -> your action selection
    reward_fn(raw, prev_raw) -> your scalar reward from raw sim quantities

Quick start
-----------
    conda activate cc3d           # env with compucell3d + rtree + gymnasium
    python rl_env.py              # runs the demo loop below (random policy)

Programmatic use
----------------
    from rl_env import MuscleRegenEnv
    env = MuscleRegenEnv(quarter=True, output_dir="/tmp/cc3d_run_0", seed=0)
    obs, info = env.reset()
    for _ in range(env.max_mcs):
        actions = {cid: policy(o, env.rng) for cid, o in obs.items()}
        obs, reward, done, truncated, info = env.step(actions)
        if done:
            break
    env.close()

Parallel rollouts
-----------------
    Each MuscleRegenEnv spins up its own CC3D via simservice. Module-level state
    in the model means one sim per process — run parallel envs as separate
    processes (multiprocessing / a job array), each with its own output_dir.

Notes / not yet verified end-to-end
-----------------------------------
  * The simservice call shape below (`service_cc3d(...).run/.init/.start`,
    `set_sim_input` / `step` / `get_sim_output`) needs confirming against the
    target CC3D build — run `python simservice_smoketest.py` first.
  * `seed` seeds numpy + stdlib random inside the model (via MUSCLEREGEN_SEED).
    The Potts core RNG needs <RandomSeed> in MuscleRegen*.xml — see README.
  * No mid-episode state snapshot/restore: reset() only starts a fresh sim.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    _Base = gym.Env
except ImportError:                     # gymnasium optional — env still works
    gym = None
    _Base = object

_HERE = Path(__file__).parent

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

    def __init__(self, quarter: bool = False, output_dir: str | None = None,
                 seed: int | None = None):
        self._cc3d_name = "MuscleRegenRLQuarter.cc3d" if quarter else "MuscleRegenRL.cc3d"
        # CC3D's simservice unconditionally copies the folder holding the .cc3d
        # file into its output dir on start, so we run from a staged .git-free
        # copy (rl_stage.py). `output_dir` here is ignored unless you need VTK
        # snapshots — pass one OUTSIDE the repo if so.
        self.output_dir = output_dir
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        self._sim = None
        self._prev_raw = None
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

        self._close_sim()
        if self.seed_value is not None:
            os.environ["MUSCLEREGEN_SEED"] = str(self.seed_value)

        from cc3d.core.simservice.PyServiceCC3D import service_cc3d
        from rl_stage import sim_paths
        sim_file, staged_out = sim_paths(self._cc3d_name)
        out_dir = self.output_dir or staged_out
        os.makedirs(out_dir, exist_ok=True)
        self._sim = service_cc3d(
            cc3d_sim_fname=sim_file,
            output_frequency=0,
            screenshot_output_frequency=0,
            output_dir=out_dir,
        )
        self._sim.run()
        self._sim.init()
        self._sim.start()

        self._sim.set_sim_input({"actions": {}})
        self._sim.step()
        out = self._sim.get_sim_output() or {}
        self._prev_raw = out.get("raw", {})
        return _parse_obs(out), {"mcs": out.get("mcs", 0), "raw": self._prev_raw}

    def step(self, actions: dict):
        self._sim.set_sim_input({"actions": {int(k): int(v) for k, v in actions.items()}})
        self._sim.step()
        out = self._sim.get_sim_output() or {}

        raw = out.get("raw", {})
        reward = float(reward_fn(raw, self._prev_raw))
        self._prev_raw = raw
        done = bool(out.get("done", False))
        info = {"mcs": out.get("mcs", -1), "raw": raw}
        return _parse_obs(out), reward, done, False, info

    def close(self):
        self._close_sim()

    def _close_sim(self):
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:
                pass
            self._sim = None


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

    ap = argparse.ArgumentParser(description="Demo rollout of MuscleRegenEnv")
    ap.add_argument("--quarter", action="store_true", help="use the quarter lattice")
    ap.add_argument("--steps", type=int, default=200, help="max env steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=None,
                    help="optional VTK output dir; must be OUTSIDE this repo")
    args = ap.parse_args()

    env = MuscleRegenEnv(quarter=args.quarter, output_dir=args.output_dir, seed=args.seed)
    obs, info = env.reset()
    print(f"reset: {len(obs)} SSC agents, mcs={info['mcs']}")

    total = 0.0
    for t in range(args.steps):
        actions = {cid: policy(o, env.rng) for cid, o in obs.items()}
        obs, reward, done, trunc, info = env.step(actions)
        total += reward
        if t % 20 == 0 or done:
            print(f"  step {t:4d}  mcs={info['mcs']:5d}  agents={len(obs):4d}  "
                  f"reward={reward:+.4f}  return={total:+.4f}")
        if done:
            print("  episode done")
            break
    env.close()
