# Parallel Sandpile HPC

A 2D-decomposed MPI implementation of the Bak-Tang-Wiesenfeld Abelian
sandpile cellular automaton, built for the MPP (Massively Parallel
Processing) coursework on **ARCHER2** during my MSc in High Performance
Computing and Data Science at EPCC, University of Edinburgh.

**Measured strong scaling on ARCHER2:** 4 ranks → 177.4s, 16 ranks →
66.9s (2.65×, 66.3% efficiency), 32 ranks across 2 nodes → 36.8s
(4.82×, 60.2% efficiency). Full numbers and discussion in
[`analysis/scaling_report.md`](analysis/scaling_report.md).

![Scaling plot](analysis/scaling_plot.png)

## What it does

Every cell in an `L x L` grid holds a sand height. Each step,
simultaneously for every cell: if a cell is above a threshold it loses
4 units of sand to its four neighbours. Repeated over thousands of
steps this produces avalanches that cascade across the grid until the
pile reaches a stable configuration. It's a classic model of
**self-organised criticality** and a good stress-test for a stencil-
style parallel decomposition, since correctness depends entirely on
getting the halo exchange right.

## Why this implementation is worth a look

- **2D Cartesian decomposition** (`MPI_Dims_create` + `MPI_Cart_create`),
  not a naive 1D row split — smaller halo perimeter per rank.
- **Non-blocking halo exchange**: all four `MPI_Irecv`/`MPI_Isend`
  pairs are posted before any `MPI_Waitall`, so the four neighbour
  exchanges overlap instead of serialising. An earlier blocking
  version with allocation inside the loop was timing out on ARCHER2 —
  see [`docs/design.md`](docs/design.md) for the full story.
- **Decomposition-independent correctness**: the initial condition is
  a deterministic hash of each cell's *global* coordinate rather than
  a per-rank random seed, so the converged grid is provably identical
  whether you run on 1 rank or 32 — verified by an automated test,
  not just eyeballed.
- **Mixed boundary conditions**: non-periodic top/bottom, periodic
  left/right, handled via `MPI_PROC_NULL` neighbours and
  `MPI_Cart_create` periods.

## Repository structure

```
parallel-sandpile-hpc/
├── src/
│   ├── sandpile_2d.c      Main MPI simulation
│   ├── arraymalloc.c/.h   Contiguous 2D int array allocator
│   └── sandio.c/.h        Binary PPM writer for visualisation
├── scripts/
│   ├── submit_4.slurm     ARCHER2 job — 4 ranks, 1 node
│   ├── submit_16.slurm    ARCHER2 job — 16 ranks, 1 node
│   ├── submit_32.slurm    ARCHER2 job — 32 ranks, 2 nodes
│   └── run_scaling_study.sh   Submits all three and collects timings
├── analysis/
│   ├── results.csv        Measured ARCHER2 timings (see above)
│   ├── plot_scaling.py    Generates the scaling plot from results.csv
│   ├── scaling_plot.png
│   └── scaling_report.md  Discussion of the efficiency numbers
├── tests/
│   └── test_convergence.py   Black-box correctness tests (pytest)
├── docs/
│   └── design.md          Design rationale and debugging history
├── Makefile
└── requirements.txt        (Python deps for analysis/tests only — the sim itself is pure C+MPI)
```

## Building and running

```bash
# On a normal machine (needs an MPI implementation — e.g. `apt install libopenmpi-dev`)
make
mpirun --oversubscribe -np 4 ./sandpile_2d --size 256 --maxh 4 --steps 20000 --out sand.ppm
convert sand.ppm sand.png   # optional, needs ImageMagick

# On ARCHER2
module load PrgEnv-gnu
make
sbatch scripts/submit_4.slurm
```

### CLI options

| Flag | Default | Meaning |
|------|--------:|---------|
| `--size N`  | 256   | Grid is N × N (must divide evenly across the process grid) |
| `--maxh N`  | 4     | Toppling threshold |
| `--steps N` | 20000 | Safety cap on iterations |
| `--seed N`  | 42    | Seed for the (decomposition-independent) initial pile |
| `--out F`   | sand.ppm | Output PPM path |

## Running the tests

```bash
make            # builds ./sandpile_2d
python3 -m pytest tests/ -v
```

The key test (`test_decomposition_independence`) runs the same
problem at 1, 2, 4, and 8 ranks and asserts the converged grids are
byte-identical — this is the test that would catch a halo-exchange
bug that "mostly" works but silently corrupts the result.

## Author

Iman Ein Alizadeh — MSc High Performance Computing and Data Science,
EPCC, University of Edinburgh.
