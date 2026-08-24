#!/bin/bash
#SBATCH --job-name=sandpile_np16
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=m25oc

# ARCHER2 submission script — 16 MPI ranks, single node
# (ARCHER2 nodes have 128 cores, so 16 ranks comfortably fit on one node).

module load PrgEnv-gnu

srun --distribution=block:block --hint=nomultithread \
    ./sandpile_2d --size 1024 --maxh 4 --steps 200000 --out results/sand_np16.ppm
