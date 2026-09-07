# Running this model on the Alliance Canada clusters

Headless batch execution of the muscle-regeneration ABM on a Digital Research
Alliance of Canada cluster, via an Apptainer container. Tested on **Nibi**
(2026-08) with CompuCell3D 4.10.0.

The clusters have no GUI and Alliance asks users not to install Anaconda, so
CompuCell3D goes in an Apptainer container (conda inside a container is allowed).

## Files

| file | purpose |
|------|---------|
| `cc3d.def`          | Apptainer recipe: miniforge + `compucell3d=4.10.*` + `rtree` + `gymnasium` |
| `run_cc3d.sh`       | SLURM script — one whole-sim run |
| `run_cc3d_array.sh` | SLURM script — N whole-sim replicates (`--array`) |
| `run_rl.sh`         | SLURM array — the RL loop (`rl_env.py` + a training script), one run per seed |

## One-time setup

```bash
ssh <user>@nibi.alliancecan.ca          # needs an SSH key registered in CCDB + MFA

cd ~/scratch
git clone <this-repo-url> ABM-of-Muscle-Regeneration-with-Microvascular-Remodeling
cd ABM-of-Muscle-Regeneration-with-Microvascular-Remodeling

module load apptainer
apptainer build cc3d.sif alliance_hpc/cc3d.def     # ~15 min, writes cc3d.sif in repo root
```

`cc3d.sif` and job output are git-ignored (see the repo-root `.gitignore`).

## Run

```bash
cd ~/scratch/ABM-of-Muscle-Regeneration-with-Microvascular-Remodeling

sbatch alliance_hpc/run_cc3d.sh          # one whole-sim run
sbatch alliance_hpc/run_cc3d_array.sh    # 20 whole-sim replicates

squeue -u $USER
```

Each whole-sim job:
1. rsyncs a **private copy** of the model into `~/scratch/muscleregen_results/<jobid>/model/`
   (CC3D 4.10 writes the `FileSteppable` text logs next to the model *source*, not
   the `-o` folder, so copies must not be shared between concurrent jobs);
2. flips `ifDataSave = 0 -> 1` in that copy (the published code ships with data
   logging off);
3. runs `cc3d.run_script` under `xvfb` (needed for the offscreen VTK context).

### RL loop

```bash
# ONE-TIME check that the RL driver works in the container (do this first):
salloc --account=def-tperkins_cpu --cpus-per-task=1 --mem=2G --time=0:20:0
apptainer exec --cleanenv --bind "$SLURM_TMPDIR" cc3d.sif \
    xvfb-run -a python step_smoketest.py --rl --quarter
exit

# then the array (edit --array in run_rl.sh for #seeds; pass your training script):
sbatch alliance_hpc/run_rl.sh train.py --quarter --episodes 500
sbatch alliance_hpc/run_rl.sh                       # no args -> runs the self-check
```

`run_rl.sh` is a job array — one independent training run per `$SLURM_ARRAY_TASK_ID`
(used as the seed). It runs your script inside the container under `xvfb`.
`rl_env.py` is only the environment; you supply `train.py` (the learner + the
`env.reset()/step()` loop + checkpointing to `$RL_OUT_DIR`). `rl_driver.py` runs
CC3D one MCS at a time in-process — **not** via `cc3d.run_script` and **not** via
simservice (which is broken for this model). `rl_stage.py` copies the model into
`$SLURM_TMPDIR` per process, so concurrent tasks never collide.

## Output

`~/scratch/muscleregen_results/<jobid>/`
- `model/*.txt` — the quantitative time series: `SSCdata.txt`, `CSAdata.txt`,
  `CapillaryData.txt`, `FibroblastData.txt`, `MacDynamics.txt`, `ecmDynamics.txt`,
  `fieldData.txt`, `logfile2MR.txt` (each has a header row; `mcs` is column 1).
- `lattice/.../LatticeData/Step_*.vtk` — full lattice + cytokine fields every
  100 MCS (open in ParaView, or read with `vtk`/`pyvista`).

CC3D prints **nothing per step** in headless mode — a quiet run is normal, not hung.

## Sizing (full run, 2688 MCS, Nibi, 1 core)

| | value |
|---|---|
| walltime | ~1.2 h with `ifDataSave=0`; longer with logging on — scripts request 12 h |
| memory   | peak ~1.5 GB — scripts request 4 GB |
| cores    | **1**. The Potts engine barely scales; multi-core is not worth it. Parallelise across replicates instead. |
| account  | `def-tperkins_cpu` (edit the `#SBATCH --account=` line for your allocation) |

`~/scratch` is auto-purged (~60 days) — move keepers to `~/projects/<def-acct>/`.

## Notes

- Reproduce the paper exactly: set `compucell3d=4.3.*` in `cc3d.def` and rebuild.
- `HistologyInitialization/` is a separate preprocessing model (histology images ->
  PIFF); not needed to run — the repo ships `MuscleInit_Complete_Final.piff`.
- If `apptainer build` refuses an unprivileged build, build in an `salloc` job or
  on a machine with Docker/root and `scp` the `.sif` over.
