
# CompuCell3D Muscle Regeneration Model

## Description
This CompuCell3D model is described in the eLife publication: "Agent-based model demonstrates the impact of nonlinear, complex interactions between cytokines on muscle regeneration."

## Installation

### Requirements
- CompuCell3D version 4.3.2
- Python version 3.7
- Conda environment package manager, we recommend Miniconda 

### Setup Instructions

1. **Install Miniconda**  
   We recommend installing Miniconda to manage conda environments and install Python.

2. **Create and Setup Conda Environment**
   ```
   conda create -n MuscleRegen python=3.7
   conda install -c conda-forge -c compucell3d rtree compucell3d=4.3.0
   ```

3. **Activate Conda Environment**
   ```
   conda activate MuscleRegen
   ```

4. **Run CompuCell3D Player GUI**
   ```
   python -m cc3d.player5
   ```

## Usage

- After completing the installation, open the `MuscleRegen.cc3d` file included in the model files through the CC3D player.
- This will run the model in the CompuCell3D Player.

## Running headless on an HPC cluster

To run the model without the GUI on a SLURM cluster (Digital Research Alliance of
Canada / Compute Canada), see [`alliance_hpc/README.md`](alliance_hpc/README.md).
It provides an Apptainer container recipe and SLURM job scripts for single runs
and replicate arrays.

## Additional Information

- This model was developed and tested using CompuCell3D version 4.3.2.
- For optimal performance, it is recommended to follow the installation and usage instructions as described.
