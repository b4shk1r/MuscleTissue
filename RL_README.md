# RL interface

External-policy control of the satellite-stem-cell (SSC) decisions in the
muscle-regeneration model, for reinforcement-learning experiments.

## Files

| file | role | edit? |
|---|---|---|
| `rl_env.py` | `MuscleRegenEnv` (gym-style `reset`/`step`/`close`) + `policy()` / `reward_fn()` stubs | **yes — this is the only file you touch** |
| `simservice_smoketest.py` | GO/NO-GO test that CC3D can be stepped one MCS at a time | run it, don't edit |
| `Simulation/RLSteppable.py` | replaces `SSCSteppable`; applies actions, exports obs + raw signals | no |
| `Simulation/MuscleRegenRL.py` | steppable registration for the RL model | no |
| `MuscleRegenRL.cc3d` / `MuscleRegenRLQuarter.cc3d` | full / quarter RL model | no |

## Use

```bash
conda activate cc3d                              # compucell3d + rtree + gymnasium
xvfb-run -a python simservice_smoketest.py --rl --quarter   # must pass first
python rl_env.py --quarter --steps 200                      # demo rollout
```

```python
from rl_env import MuscleRegenEnv
env = MuscleRegenEnv(quarter=True, output_dir="/tmp/run0", seed=0)
obs, info = env.reset()
obs, reward, done, trunc, info = env.step({cid: 0 for cid in obs})   # advance 1 MCS
```

7 discrete actions (noop/migrate, activate, divide, differentiate, fuse-to-fiber,
fuse-to-myotube, apoptose); 22-float observation per SSC. Full detail:
`../docs/rl-quickstart.md` and the docstrings in `rl_env.py` /
`Simulation/RLSteppable.py`.

## Status

Written, **not yet run end-to-end** (needs CC3D). The `service_cc3d(...)` call in
`rl_env.reset()` may need tweaking for the installed CC3D version — the smoke
test surfaces that. See `../docs/upgrade-plan.md` for the full plan and open
items.
