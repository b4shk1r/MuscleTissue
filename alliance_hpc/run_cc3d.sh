#!/bin/bash
# One headless CompuCell3D run of the muscle-regeneration model on a
# Digital Research Alliance of Canada cluster (tested on Nibi).
#
#   sbatch alliance_hpc/run_cc3d.sh        # run this FROM the repo root
#
# Prereqs (see alliance_hpc/README.md):
#   - repo cloned into ~/scratch
#   - container built:  apptainer build cc3d.sif alliance_hpc/cc3d.def   (in repo root)

#SBATCH --job-name=muscleregen
#SBATCH --account=def-tperkins_cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=muscleregen-%j.out

module load apptainer

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
SIF="$REPO/cc3d.sif"
RESULTS="$HOME/scratch/muscleregen_results/$SLURM_JOB_ID"
mkdir -p "$RESULTS/model"

# Private per-job copy of the model. CC3D 4.10's FileSteppable writes its text
# logs (SSCdata.txt, CSAdata.txt, ...) next to the model SOURCE dir, not the -o
# folder -- so concurrent jobs sharing one copy would clobber each other's data.
rsync -a --exclude='.git' --exclude='cc3d.sif' --exclude='alliance_hpc' \
      --exclude='*.out' "$REPO/" "$RESULTS/model/"

# The published code ships with data logging OFF. Turn it on for batch runs.
sed -i 's/^ifDataSave = 0/ifDataSave = 1/' \
    "$RESULTS/model/Simulation/MuscleRegenSteppables.py"

echo "START $(date)"
apptainer exec --cleanenv --bind "$RESULTS" "$SIF" \
    xvfb-run -a python -m cc3d.run_script \
        -i "$RESULTS/model/MuscleRegen.cc3d" \
        -o "$RESULTS/lattice" \
        -f 100
status=$?
echo "END $(date)  exit=$status"

echo "==== text data (line counts) ===="
wc -l "$RESULTS"/model/*.txt 2>/dev/null
echo "results: $RESULTS"
exit $status
