#!/bin/bash
#SBATCH --job-name=sandpile_32
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=standard
#SBATCH --account=m25oc-s2901349
#SBATCH --qos=short

module load PrgEnv-gnu

srun ./sandpile_2d 512
