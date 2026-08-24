#!/bin/bash
#SBATCH --job-name=sandpile_np32
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=m25oc

# ARCHER2 submission script — 32 MPI ranks across 2 nodes.
# This is the multi-node run that satisfies the "must scale across more
# than one node" part of the coursework spec — it forces inter-node
# communication over the Slingshot interconnect rather than everything
# staying on-chip.

module load PrgEnv-gnu

srun --distribution=block:block --hint=nomultithread \
    ./sandpile_2d --size 1024 --maxh 4 --steps 200000 --out results/sand_np32.ppm
