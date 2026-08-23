# Parallel Abelian Sandpile — MPI + 2D Domain Decomposition

A high-performance parallel implementation of the Abelian sandpile model using MPI with 2D domain decomposition, non-blocking communication, and convergence tracking. Developed as part of the MSc HPC coursework at EPCC, University of Edinburgh, and run at scale on the ARCHER2 national supercomputer.

---

## What It Does

The Abelian sandpile model is a cellular automaton defined on a 2D grid. Each cell holds a number of sand grains. When a cell exceeds a threshold (4 grains), it topples — distributing one grain to each of its four neighbours. This continues until the system reaches a stable state (no cell exceeds the threshold). At scale, simulating this requires efficient parallel decomposition across hundreds of cores.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MPI Process Grid                   │
│                                                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │  Rank 0  │  │  Rank 1  │  │  Rank 2  │  ...    │
│   │          │←→│          │←→│          │         │
│   │ Local    │  │ Local    │  │ Local    │         │
│   │ Subgrid  │  │ Subgrid  │  │ Subgrid  │         │
│   └────↕─────┘  └────↕─────┘  └────↕─────┘         │
│        ↕             ↕             ↕                │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │  Rank N  │  │  Rank N+1│  │  Rank N+2│         │
│   └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────┘
         ↓
   MPI_Allreduce for global convergence check
```

**Key design decisions:**
- 2D Cartesian topology via `MPI_Cart_create`
- Non-blocking halo exchange using `MPI_Isend` / `MPI_Irecv` with `MPI_Waitall`
- Convergence tracked via `MPI_Allreduce` on local change counts
- Centralised I/O via `MPI_Gatherv` to rank 0 only
- Non-periodic boundary in i-direction, periodic in j-direction

---

## Results

| Processes | Nodes | Time (s) | Speedup | Efficiency |
|-----------|-------|----------|---------|------------|
| 4 | 1 | 177.43 | 1.00× | 100% |
| 16 | 1 | 66.88 | 2.65× | 66.3% |
| 32 | 2 | 36.83 | 4.82× | 60.2% |

Grid size: 512×512. Tested on ARCHER2 (AMD EPYC 7742, 128 cores per node).

---

## Repository Structure

```
parallel-sandpile-hpc/
├── src/
│   ├── sandpile_2d.c          # Main MPI implementation
│   ├── arraymalloc.c          # 2D array memory utilities
│   ├── sandio.c               # PPM image output
│   └── sandpile_serial.c      # Serial baseline for comparison
├── scripts/
│   ├── submit_4proc.sh        # SLURM job: 4 processes
│   ├── submit_16proc.sh       # SLURM job: 16 processes
│   └── submit_32proc.sh       # SLURM job: 32 processes, 2 nodes
├── results/
│   ├── scaling_results.csv    # Timing data across process counts
│   ├── speedup_plot.png       # Speedup curve
│   └── sandpile_512.png       # Final sandpile visualisation
├── docs/
│   └── report.pdf             # Full coursework report
└── README.md
```

---

## Running on ARCHER2

### 1. Load modules and compile

```bash
module load PrgEnv-gnu
module load imagemagick/7.1.0

mpicc -O3 -o sandpile_2d src/sandpile_2d.c src/arraymalloc.c src/sandio.c -lm
```

### 2. Submit a job

```bash
sbatch scripts/submit_16proc.sh
```

### 3. Visualise output

```bash
convert sand.ppm sand.png
scp username@login.archer2.ac.uk:/path/to/sand.png ~/Desktop/
```

---

## Running Locally (small grid)

```bash
mpicc -O3 -o sandpile_2d src/sandpile_2d.c src/arraymalloc.c src/sandio.c -lm
mpirun -np 4 ./sandpile_2d 256
```

Requires OpenMPI or MPICH installed locally.

---

## Dependencies

| Tool | Purpose |
|------|---------|
| MPI (OpenMPI / MPICH) | Parallel communication |
| GCC | C compilation |
| ImageMagick | PPM → PNG conversion |
| Python + matplotlib | Performance analysis plots |

---

## Performance Analysis

Speedup and efficiency were analysed using the Karp-Flatt metric to separate serial fraction from parallel overhead. Communication cost grows with process count due to increased halo exchange surface area relative to local subdomain size, explaining the sub-linear speedup at higher process counts.

---

## Academic Context

**Course:** Massively Parallel Programming (MPP) — MSc HPC and Data Science
**Institution:** EPCC, University of Edinburgh
**Supercomputer:** ARCHER2 (UK National Supercomputing Service)
