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
    """Scalar reward from the whole-tissue quantities RLSteppable exports each step.

    `raw` keys (all plain scalars unless noted):
        mcs, fiber_volume, initial_fiber_volume, fiber_count, n_myotube_immature,
        newMyotube, n_ssc, n_ssc_active, n_myoblast, n_myocyte, n_macrophage,
        n_neutrophil, n_fibroblast, n_necrotic, necrosisRemain, collagen_mean,
        collagen_fibrotic_frac, cyto_mean {HGF,MMP,TGF,VEGF,TNF,IL10}
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


# --- Reference baseline: the model's hand-coded SSC decision, as a policy -----
# Run this through the env (--policy fixed) to get the trajectory the published
# model produces, so "can RL match / beat the hand-coded policy?" is a direct
# comparison. APPROXIMATE: the real rule also gates on the ECM cell directly
# below (collagen level, "repairable" flag) and on being on a *mature*-fiber
# edge -- neither is in the 22-float obs, so this uses the fiber-neighbour count
# as a proxy. Thresholds are from Simulation/MuscleRegenSteppables.py.
_CYTO_MAX = 500.0
_ACT_THRESH   = 44.7091   # sscActivationThreshold  (HGF to activate)
_DIV_THRESH   = 126.6268  # SSCdivisionThreshold    (TNF+VEGF-TGF)
_DIFF_THRESH  = 57.9870   # SSCdiffThreshold        (3*IL10-HGF-TNF-TGF)
_DIV_PROB     = 0.0972    # sscDivideProb           (divide w/o signal)
_DIFF_PROB    = 0.6136    # sscDiffProb             (diff w/o signal)
_APOP_TGF     = 79.1563   # sscTGFApoptosisThreshold
_APOP_PROB    = 0.7651    # sscApoptosisProb
_VEGF_BLOCK   = 80.6055   # VEGFblockApop
_QUIESC_HGF   = 36.3109   # quiescentThreshold
_DIV_CHANCE   = [0.9, 0.85, 0.65, 0.2]   # sscDivisionChanceSubseq, by numDiv


def fixed_policy(obs_vec: np.ndarray, rng: np.random.Generator) -> int:
    o = obs_vec
    activated = o[2] > 0.5
    cell_type = round(float(o[3]) * 2)          # 0 SSC, 1 myoblast, 2 myocyte
    num_div   = round(float(o[7]) * 4)
    HGF, MMP, TGF  = o[8] * _CYTO_MAX, o[9] * _CYTO_MAX, o[10] * _CYTO_MAX
    VEGF, TNF, IL10 = o[11] * _CYTO_MAX, o[12] * _CYTO_MAX, o[13] * _CYTO_MAX
    nb_fiber   = o[15] * 10.0
    nb_ecm     = o[16] * 10.0
    nb_ssc     = o[19] * 10.0
    nb_macro   = o[20] * 10.0
    on_fiber_edge = nb_fiber > 0

    if not activated:
        return 1 if HGF >= _ACT_THRESH else 0     # activate / wait

    # apoptosis check (unconditional in the original, before the decision body)
    if TGF > _APOP_TGF and rng.random() < _APOP_PROB and VEGF < _VEGF_BLOCK and nb_macro == 0:
        return 6

    if cell_type == 2:                            # myocyte: try to fuse
        if nb_ecm > 0 and nb_ssc > 0:
            return 5                              # fuse with adjacent myocyte -> myotube
        if nb_ecm > 0:
            return 4                              # fuse to fiber
        return 0

    if on_fiber_edge:                             # SSC / myoblast on a fiber edge
        if (TNF + VEGF - TGF > _DIV_THRESH or rng.random() < _DIV_PROB):
            if rng.random() < _DIV_CHANCE[min(num_div, 3)]:
                return 2                          # divide
        if (3 * IL10 - HGF - TNF - TGF > _DIFF_THRESH or rng.random() < _DIFF_PROB):
            return 3                              # differentiate

    if HGF < _QUIESC_HGF:
        return 0                                  # drifts back to quiescence in-model
    return 0                                      # migrate

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
    ap.add_argument("--policy", choices=["random", "fixed"], default="random",
                    help="'fixed' = the model's hand-coded SSC rule (reference baseline)")
    args = ap.parse_args()

    act_fn = fixed_policy if args.policy == "fixed" else policy

    env = MuscleRegenEnv(quarter=args.quarter, seed=args.seed)
    t0 = time.time()
    obs, info = env.reset()
    print(f"reset: {len(obs)} SSC agents, mcs={info['mcs']}, {time.time()-t0:.1f}s "
          f"[{args.policy} policy]", flush=True)

    total = 0.0
    for t in range(args.steps):
        ts = time.time()
        actions = {cid: act_fn(o, env.rng) for cid, o in obs.items()}
        obs, reward, done, trunc, info = env.step(actions)
        total += reward
        print(f"  step {t:4d}  mcs={info['mcs']:5d}  agents={len(obs):4d}  "
              f"reward={reward:+.4f}  return={total:+.4f}  {time.time()-ts:.1f}s", flush=True)
        if done:
            print("  episode done")
            break
    env.close()
    print(f"total {time.time()-t0:.1f}s")
