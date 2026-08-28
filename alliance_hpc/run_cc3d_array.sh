#!/bin/bash
# Job array: N independent stochastic replicates of the muscle-regeneration model.
# RandomSeed (MuscleRegen.xml) and random.seed (MuscleRegenSteppables.py) are both
# left commented out in the published code, so each task auto-seeds from system
# entropy and produces a distinct replicate -- no seed injection needed.
#
#   sbatch alliance_hpc/run_cc3d_array.sh      # run FROM the repo root
#
# For a parameter sweep instead of replicates: generate one MuscleRegen.xml per
# task before the apptainer call and vary it by $SLURM_ARRAY_TASK_ID.

#SBATCH --job-name=muscleregen_arr
#SBATCH --account=def-tperkins_cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=1-20
#SBATCH --output=muscleregen_arr-%A_%a.out

module load apptainer

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SIF="$REPO/cc3d.sif"
RESULTS="$HOME/scratch/muscleregen_results/${SLURM_ARRAY_JOB_ID}/rep_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS/model"

rsync -a --exclude='.git' --exclude='cc3d.sif' --exclude='alliance_hpc' \
      --exclude='*.out' "$REPO/" "$RESULTS/model/"
sed -i 's/^ifDataSave = 0/ifDataSave = 1/' \
    "$RESULTS/model/Simulation/MuscleRegenSteppables.py"

echo "START rep ${SLURM_ARRAY_TASK_ID} $(date)"
apptainer exec --cleanenv --bind "$RESULTS" "$SIF" \
    xvfb-run -a python -m cc3d.run_script \
        -i "$RESULTS/model/MuscleRegen.cc3d" \
        -o "$RESULTS/lattice" \
        -f 100
status=$?
echo "END rep ${SLURM_ARRAY_TASK_ID} $(date)  exit=$status"
wc -l "$RESULTS"/model/*.txt 2>/dev/null
exit $status
