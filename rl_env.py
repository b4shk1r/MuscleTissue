"""
MuscleRegenEnv — Gymnasium wrapper around the CC3D muscle regeneration sim.

Usage
-----
    from rl_env import MuscleRegenEnv

    env = MuscleRegenEnv(output_dir="/tmp/cc3d_run_0")
    obs, _ = env.reset()
    # obs = {cell_id: np.array(shape=(22,), dtype=float32)}

    for step in range(2688):
        # Your policy here — one action per live SSC cell
        actions = {cell_id: env.action_space.sample() for cell_id in obs}
        obs, reward, done, truncated, info = env.step(actions)
        if done:
            break
    env.close()

Running multiple envs in parallel
----------------------------------
    envs = [MuscleRegenEnv(output_dir=f"/tmp/cc3d_{i}") for i in range(4)]
    # Each env spins up its own CC3D subprocess automatically.

Requires the cc3d conda env:
    /opt/anaconda3/envs/cc3d/bin/python3.12
"""

import os
import gymnasium as gym
import numpy as np
from pathlib import Path

_SIM_FILE = str(Path(__file__).parent / "MuscleRegenRL.cc3d")

OBS_DIM   = 22
N_ACTIONS = 7


class MuscleRegenEnv(gym.Env):
    """
    Multi-agent CC3D environment where each SSC cell is an independent agent
    sharing a single policy network.

    obs  : dict {cell_id (int) -> np.ndarray shape (22,) float32}
    act  : dict {cell_id (int) -> int in [0, N_ACTIONS)}

    Actions
    -------
    0  do nothing
    1  activate
    2  divide
    3  differentiate
    4  fuse to adjacent fiber  (myocyte only)
    5  fuse with adjacent myocyte → myotube
    6  apoptose

    Observation vector (22 floats, all normalised to [0, 1])
    ----------------------------------------------------------
    0  x / dim_x
    1  y / dim_y
    2  activationState
    3  cellType / 2
    4  time2activate / sscActivationTime
    5  time2divide / sscDivisionTime
    6  time2diff / sscDifferentiationTime
    7  numDiv / 4
    8  HGF
    9  MMP
    10 TGF
    11 VEGF
    12 TNF
    13 IL10
    14 necrosisRemain (global)
    15 fiber neighbor count / 10
    16 ECM neighbor count / 10
    17 necrotic neighbor count / 10
    18 capillary neighbor count / 10
    19 SSC neighbor count / 10
    20 macrophage neighbor count / 10
    21 mcs / total_mcs

    Reward
    ------
    Δ(fiber_volume) / initial_fiber_volume  per step
    + 0.1 * new_myotube_count_this_step
    """

    metadata = {"render_modes": []}

    def __init__(self, output_dir: str = "/tmp/cc3d_rl_default"):
        super().__init__()
        self.output_dir = output_dir
        self._sim       = None

        # Per-cell spaces (the multi-agent loop is handled by the caller)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sim()

        from cc3d.core.simservice.PyServiceCC3D import service_cc3d

        os.makedirs(self.output_dir, exist_ok=True)
        self._sim = service_cc3d(
            cc3d_sim_fname=_SIM_FILE,
            output_frequency=0,
            screenshot_output_frequency=0,
            output_dir=self.output_dir,
        )
        self._sim.run()
        self._sim.init()
        self._sim.start()

        # Send empty actions to prime the steppable and get initial observations
        self._sim.set_sim_input({"actions": {}})
        self._sim.step()
        output = self._sim.get_sim_output() or {}

        obs = _parse_obs(output)
        return obs, {}

    def step(self, actions: dict):
        """
        Parameters
        ----------
        actions : dict {cell_id (int) -> action (int)}
            Pass an empty dict to advance the sim with no agent interventions.

        Returns
        -------
        obs      : dict {cell_id -> np.ndarray(22,)}
        reward   : float
        done     : bool
        truncated: bool  (always False — no time limit separate from sim end)
        info     : dict  {"mcs": int}
        """
        self._sim.set_sim_input({"actions": actions})
        self._sim.step()
        output = self._sim.get_sim_output() or {}

        obs      = _parse_obs(output)
        reward   = float(output.get("reward", 0.0))
        done     = bool(output.get("done", False))
        info     = {"mcs": output.get("mcs", -1)}

        return obs, reward, done, False, info

    def close(self):
        self._close_sim()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _close_sim(self):
        if self._sim is not None:
            try:
                self._sim.stop()
            except Exception:
                pass
            self._sim = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_obs(output: dict) -> dict:
    """Convert sim_output JSON to {cell_id: np.ndarray} dict."""
    return {
        int(k): np.array(v, dtype=np.float32)
        for k, v in output.get("obs", {}).items()
    }
