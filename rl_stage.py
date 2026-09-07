"""
rl_stage.py — stage a clean copy of the model for CC3D simservice.

CC3D's simservice ALWAYS copies the folder that holds the .cc3d file into its
output directory on startup (`copy_simulation_files_to_output_folder`, no way to
disable it). Our .cc3d files live in the repo root, so that copy would try to
duplicate `.git/` (read-only objects -> PermissionError) and the auto-chosen
output dir lands inside the repo -> "Output directory cannot be inside the
simulation folder". A stale `Simulation/_settings.sqlite` also pins `OutputLocation`
to wherever the GUI last ran.

Fix: copy just the files CC3D needs into a scratch dir (no .git, no settings db),
run simservice from there, and give it an output dir that is a *sibling* of that
scratch dir. Call `stage_model()` and pass the returned paths to `service_cc3d`.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent


def _scratch_root() -> str | None:
    """Node-local fast scratch on a SLURM job, else the OS default temp dir."""
    for var in ("SLURM_TMPDIR", "TMPDIR"):
        d = os.environ.get(var)
        if d and os.path.isdir(d):
            return d
    return None

# Only these are needed to run any MuscleRegen*/RL model.
_INCLUDE = [
    "Simulation/MuscleRegen.py",
    "Simulation/MuscleRegenRL.py",
    "Simulation/MuscleRegenSteppables.py",
    "Simulation/RLSteppable.py",
    "Simulation/MuscleRegen.xml",
    "Simulation/MuscleRegenQuarter.xml",
    "MuscleInit_Complete_Final.piff",
    "MuscleInit_Quarter.piff",
    "MuscleRegen.cc3d",
    "MuscleRegenQuarter.cc3d",
    "MuscleRegenRL.cc3d",
    "MuscleRegenRLQuarter.cc3d",
]

_staged: dict[str, Path] = {}


def stage_model(keep: bool = False) -> Path:
    """Copy the model into a fresh scratch dir. Returns that dir (the 'model root').

    Idempotent within a process. The dir is removed at interpreter exit unless
    keep=True.
    """
    if "root" in _staged and _staged["root"].exists():
        return _staged["root"]

    tmp = Path(tempfile.mkdtemp(prefix="muscleregen_rl_", dir=_scratch_root()))
    model = tmp / "model"
    (model / "Simulation").mkdir(parents=True)

    for rel in _INCLUDE:
        src = _REPO / rel
        if not src.exists():
            if rel.endswith(".piff"):
                continue  # quarter piff may not be generated yet; only one is needed
            raise FileNotFoundError(f"rl_stage: missing {src}")
        shutil.copy2(src, model / rel)

    _staged["root"] = model
    if not keep:
        atexit.register(lambda: shutil.rmtree(tmp, ignore_errors=True))
    return model


def sim_paths(cc3d_name: str, keep: bool = False) -> tuple[str, str]:
    """Returns (cc3d_sim_fname, output_dir) ready for service_cc3d(...).

    output_dir is a sibling of the staged model dir (never inside it, never in
    the repo, no .git to copy).
    """
    model = stage_model(keep=keep)
    cc3d_path = model / cc3d_name
    if not cc3d_path.exists():
        raise FileNotFoundError(f"rl_stage: {cc3d_name} not staged (check _INCLUDE)")
    out = model.parent / "out"
    out.mkdir(exist_ok=True)
    return str(cc3d_path), str(out)
