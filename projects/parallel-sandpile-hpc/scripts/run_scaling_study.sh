#!/bin/bash
# run_scaling_study.sh
#
# Submits the 4/16/32-rank SLURM jobs in sequence on ARCHER2 and waits
# for each to complete before submitting the next, so the timing lines
# in the .out files can be pulled straight into analysis/results.csv.
#
# Usage (from the repo root, after `make`):
#   bash scripts/run_scaling_study.sh

set -e
mkdir -p results

for np in 4 16 32; do
    echo "Submitting np=$np ..."
    jobid=$(sbatch --parsable scripts/submit_${np}.slurm)
    echo "  job id: $jobid — waiting for completion"
    while squeue -j "$jobid" 2>/dev/null | grep -q "$jobid"; do
        sleep 10
    done
    echo "  done. Timing line:"
    grep "Wall time" slurm-${jobid}.out || echo "  (no timing line found — check slurm-${jobid}.out)"
done

echo "All scaling runs complete. See analysis/results.csv for the recorded figures."
