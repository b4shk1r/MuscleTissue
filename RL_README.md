# RL interface

External-policy control of the satellite-stem-cell (SSC) decisions in the
muscle-regeneration model, for reinforcement-learning experiments.

## Files

| file | role | edit? |
|---|---|---|
| `rl_env.py` | `MuscleRegenEnv` (gym-style `reset`/`step`/`close`) + `policy()` / `reward_fn()` stubs | **yes — this is the only file you touch** |
| `rl_driver.py` | in-process one-MCS-at-a-time CC3D stepper | no |
| `rl_stage.py` | stages a `.git`-free copy of the model for CC3D | no |
| `step_smoketest.py` | quick check that stepping works on your build | run it, don't edit |
| `Simulation/RLSteppable.py` | replaces `SSCSteppable`; applies actions, exports obs + raw signals | no |
| `Simulation/MuscleRegenRL.py` | steppable registration for the RL model | no |
| `MuscleRegenRL.cc3d` / `MuscleRegenRLQuarter.cc3d` | full / quarter RL model | no |

## Use

```bash
conda activate cc3d                       # compucell3d 4.10 + rtree (+ gymnasium)
python step_smoketest.py --rl --quarter   # ~1 min setup + ~20 s/step
```

```python
from rl_env import MuscleRegenEnv
env = MuscleRegenEnv(quarter=True, seed=0)
obs, info = env.reset()                          # obs = {cell_id: np.array(22,)}
obs, reward, done, trunc, info = env.step({cid: 0 for cid in obs})   # advance 1 MCS
env.close()
```

7 discrete actions (noop/migrate, activate, divide, differentiate, fuse-to-fiber,
fuse-to-myotube, apoptose); 22-float observation per SSC. Full detail:
`../docs/rl-quickstart.md` and the docstrings in `rl_env.py` /
`Simulation/RLSteppable.py`.

## How it steps CC3D

Not via `simservice` — that path is broken for this model (the spawned worker's
Python steppables never see the C++ cells). `rl_driver.CC3DDriver` runs CC3D's
normal CML main loop but pauses it before the `while` loop and steps it by hand,
in-process. Consequences:

- **one sim per process** (CC3D core + model module-globals are singletons) —
  run parallel rollouts as separate processes.
- no multiprocessing, so `rl_env` imports cleanly anywhere, no `__main__` guard.
- `seed=` seeds numpy + `random` in the model (`MUSCLEREGEN_SEED`); the Potts core
  RNG needs `<RandomSeed>` in `Simulation/MuscleRegen*.xml`.
- no mid-episode snapshot/restore; `reset()` is a fresh sim from the fixed injury.

## Status

Verified end-to-end on **CC3D 4.10**, one Mac core:

| model | setup | per MCS |
|---|---|---|
| quarter (`--quarter` / `MuscleRegenRLQuarter.cc3d`) | ~3 s | ~0.4 s |
| full lattice (`MuscleRegenRL.cc3d`) | ~6 s | ~1.7 s |

3 concurrent rollouts (separate processes) run clean — `python parallel_test.py`.
Remaining: SLURM script for the cluster + headless-Linux check; `_execute_action`
fidelity and custom-observation hooks are decisions for the collaborator. See
`../docs/upgrade-plan.md`.
