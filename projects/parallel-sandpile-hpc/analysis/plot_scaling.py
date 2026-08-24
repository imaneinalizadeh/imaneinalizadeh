#!/usr/bin/env python3
"""
plot_scaling.py

Reads results.csv (rank count, wall time, speedup, efficiency measured
on ARCHER2) and produces a two-panel scaling plot: speedup vs. ideal
linear speedup, and parallel efficiency vs. rank count.

Usage:
    python3 plot_scaling.py [--csv results.csv] [--out scaling_plot.png]
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(path):
    ranks, wall_times, speedups, efficiencies = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["ranks"]))
            wall_times.append(float(row["wall_time_s"]))
            speedups.append(float(row["speedup"]))
            efficiencies.append(float(row["efficiency_pct"]))
    return ranks, wall_times, speedups, efficiencies


def make_plot(ranks, speedups, efficiencies, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ideal = [speedups[0] * (r / ranks[0]) for r in ranks]
    ax1.plot(ranks, ideal, "--", color="gray", label="Ideal linear")
    ax1.plot(ranks, speedups, "o-", color="#1f77b4", label="Measured (ARCHER2)")
    ax1.set_xlabel("MPI ranks")
    ax1.set_ylabel("Speedup (relative to 4 ranks)")
    ax1.set_title("Strong scaling — speedup")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ranks, efficiencies, "o-", color="#d62728")
    ax2.axhline(100, linestyle=":", color="gray")
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel("Parallel efficiency (%)")
    ax2.set_title("Strong scaling — efficiency")
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3)

    fig.suptitle("2D MPI Sandpile — Strong Scaling on ARCHER2 (2 nodes max)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--csv", default=os.path.join(here, "results.csv"))
    parser.add_argument("--out", default=os.path.join(here, "scaling_plot.png"))
    args = parser.parse_args()

    ranks, wall_times, speedups, efficiencies = load_results(args.csv)
    make_plot(ranks, speedups, efficiencies, args.out)


if __name__ == "__main__":
    main()
