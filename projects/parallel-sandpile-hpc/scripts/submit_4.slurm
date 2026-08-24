#!/bin/bash
#SBATCH --job-name=sandpile_np4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=m25oc

# ARCHER2 submission script — 4 MPI ranks, single node.
# Run from the directory containing the compiled `sandpile_2d` binary.

module load PrgEnv-gnu

srun --distribution=block:block --hint=nomultithread \
    ./sandpile_2d --size 1024 --maxh 4 --steps 200000 --out results/sand_np4.ppm
