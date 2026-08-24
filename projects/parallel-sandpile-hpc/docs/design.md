# Design Notes — 2D MPI Sandpile

## The model

The Bak-Tang-Wiesenfeld Abelian sandpile: an `L x L` grid of integer
heights. Each step, simultaneously for every cell:

```
if h > MAXH:               h -= 4
for each of 4 neighbours:  if that neighbour's PREVIOUS height > MAXH:
                                h += 1
```

Because every cell only needs its own and its neighbours' values from
the *previous* step, this is naturally data-parallel: split the grid
into blocks, exchange a 1-cell halo each step, and every rank can
update its own block independently.

## Decomposition

- 2D Cartesian topology via `MPI_Dims_create` + `MPI_Cart_create`,
  rather than a 1D row/column split — halves the halo perimeter for a
  given rank count once you're past a handful of ranks, since a 2D
  block has a smaller surface-to-area ratio than a 1D strip.
- **Boundary conditions:** non-periodic in the row (i) direction
  (`MPI_PROC_NULL` neighbours at the top/bottom — sand "falls off"),
  periodic in the column (j) direction (`MPI_Cart_create(..., periods
  = {0, 1}, ...)` — the pile wraps left/right).

## Why non-blocking communication

An earlier draft used allocation *inside* the per-step loop and
blocking `MPI_Sendrecv` calls in sequence (north, then south, then
east, then west). That version timed out on ARCHER2 at anything past
a trivially small grid. Two fixes:

1. **Halo buffers allocated once, outside the loop** — the original
   bug. `malloc`/`free` every step for millions of steps dominated the
   runtime.
2. **All four `MPI_Irecv` + `MPI_Isend` posted before any
   `MPI_Waitall`** — lets the four neighbour exchanges overlap in
   flight instead of forcing them to complete one at a time.

## Why the initial condition is seeded by global coordinate, not rank

Early versions seeded each rank's local RNG stream independently
(`srand(seed + rank * constant)`). That makes the *global* initial
pile depend on how many ranks you're running with — different
decomposition, different per-rank RNG state, different starting grid.
That silently broke the most important correctness check: verifying
that np=1 and np=4 converge to the *same physical result*. The fix
was to derive each cell's initial height from a hash of its **global**
`(row, col)` coordinate, which is decomposition-independent by
construction — see `tests/test_convergence.py::test_decomposition_independence`.

## Convergence

Every rank counts cells whose height changed this step;
`MPI_Allreduce(..., MPI_SUM)` gives the global change count. The
simulation stops when that hits zero (a genuinely stable pile) or
`--steps` is exhausted, whichever comes first — this is a correctness
property, not just a performance shortcut: a fixed step count risks
either stopping mid-avalanche or wasting cycles after the pile has
already settled.

## Output gathering

Rather than a derived MPI subarray type for `MPI_Gatherv`, each rank's
local block is flattened and sent with a plain `MPI_Gather`; rank 0
reassembles the full grid by looking up each sender's Cartesian
coordinates and copying its block into the right offset. This trades
a bit of O(L²) work on rank 0 (done once, at the very end) for
noticeably simpler, easier-to-verify code than a custom MPI datatype
would be — a reasonable trade for output that happens once per run,
not once per step.
