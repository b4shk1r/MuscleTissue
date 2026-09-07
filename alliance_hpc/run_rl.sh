#!/bin/bash
# Run the RL loop (rl_env.py + your training script) on a Digital Research
# Alliance of Canada cluster, inside the CC3D Apptainer container.
#
#   sbatch alliance_hpc/run_rl.sh                       # runs TRAIN_SCRIPT below
#   sbatch alliance_hpc/run_rl.sh path/to/train.py --episodes 500
#
# This is a job ARRAY: one independent training run per array task, seeded by
# $SLURM_ARRAY_TASK_ID. Adjust --array for the number of replicates / seeds.
#
# Prereqs (see alliance_hpc/README.md):
#   - repo cloned into ~/scratch, this script run FROM the repo root
#   - container built:  apptainer build cc3d.sif alliance_hpc/cc3d.def
#     (cc3d.def pins compucell3d=4.10.* + rtree + gymnasium)
#
# BEFORE using this for real: run the headless check once on a compute node --
#   apptainer exec --cleanenv --bind "$SLURM_TMPDIR" cc3d.sif \
#     xvfb-run -a python step_smoketest.py --rl --quarter
# rl_driver.py monkeypatches CC3D's main loop and needs the offscreen GL context
# that xvfb provides; this proves it works in the container.

#SBATCH --job-name=muscleregen_rl
#SBATCH --account=def-tperkins_cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1          # one core per sim process (CC3D Potts is ~single-threaded)
#SBATCH --mem=2G                   # ~0.5 GB quarter, ~1.5 GB full lattice
#SBATCH --array=1-4                # 4 seeds; widen for more replicates
#SBATCH --output=muscleregen_rl-%A_%a.out

module load apptainer

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SIF="$REPO/cc3d.sif"

# Default payload: the pipeline self-check. Replace with your training script,
# e.g.  sbatch alliance_hpc/run_rl.sh train.py --quarter --episodes 500
TRAIN_SCRIPT="${1:-step_smoketest.py}"
shift || true
TRAIN_ARGS=("$@")
if [ "$TRAIN_SCRIPT" = "step_smoketest.py" ] && [ ${#TRAIN_ARGS[@]} -eq 0 ]; then
    TRAIN_ARGS=(--rl --quarter --steps 20)
fi

SEED="${SLURM_ARRAY_TASK_ID:-0}"
OUT="$HOME/scratch/rl_results/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/rep_${SEED}"
mkdir -p "$OUT"

echo "START $(date)  seed=$SEED  script=$TRAIN_SCRIPT ${TRAIN_ARGS[*]}"
# rl_stage.py stages the model into $SLURM_TMPDIR (node-local fast scratch);
# bind it so the container can see it. --cleanenv keeps the job env minimal but
# we pass the vars the RL code reads.
apptainer exec --cleanenv \
    --bind "$SLURM_TMPDIR" --bind "$OUT" --bind "$REPO" \
    --env MUSCLEREGEN_SEED="$SEED" \
    --env RL_OUT_DIR="$OUT" \
    "$SIF" \
    bash -c "cd '$REPO' && xvfb-run -a python '$TRAIN_SCRIPT' ${TRAIN_ARGS[*]}"
status=$?

echo "END $(date)  exit=$status"
echo "results: $OUT"
exit $status
