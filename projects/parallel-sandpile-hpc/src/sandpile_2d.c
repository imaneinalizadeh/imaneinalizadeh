/*
 * sandpile_2d.c
 *
 * 2D-decomposed MPI implementation of the Bak-Tang-Wiesenfeld Abelian
 * sandpile cellular automaton.
 *
 * Rule (applied simultaneously to every cell each step):
 *   if h[i][j] > MAXH:            h[i][j] -= 4
 *   for each of the 4 neighbours: if that neighbour's height (from
 *                                   the PREVIOUS step) > MAXH, h[i][j] += 1
 *
 * This is the standard "simultaneous toppling" formulation, which is
 * what makes the automaton trivially data-parallel: every rank only
 * needs a 1-cell halo of its neighbours' PREVIOUS heights to compute
 * its own next step exactly.
 *
 * Domain decomposition: 2D Cartesian grid of MPI ranks (MPI_Dims_create
 * + MPI_Cart_create). Boundary conditions: non-periodic in the
 * i-direction (rows) — sand falls off the top/bottom edges — and
 * periodic in the j-direction (columns) — the pile wraps around
 * left/right. This matches the coursework specification.
 *
 * Communication: non-blocking halo exchange (MPI_Isend / MPI_Irecv +
 * MPI_Waitall) so the four directions overlap rather than serialising.
 *
 * Convergence: every rank counts how many of its own cells changed
 * this step; MPI_Allreduce(SUM) gives the global change count. The
 * simulation stops when that reaches zero (a stable pile) or when
 * --steps is hit, whichever comes first.
 *
 * Output: the final grid is gathered to rank 0 with MPI_Gatherv and
 * written to a binary PPM via sandio.c.
 *
 * Build:   see ../Makefile  (mpicc -O2 -Wall)
 * Run:     mpirun -np <P> ./sandpile_2d --size 256 --maxh 4 --steps 20000
 *
 * Author:  Iman Ein Alizadeh
 * Course:  MSc High Performance Computing and Data Science, EPCC,
 *          University of Edinburgh — MPP coursework (ARCHER2)
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "arraymalloc.h"
#include "sandio.h"

#define DEFAULT_L      256
#define DEFAULT_MAXH   4
#define DEFAULT_STEPS  20000
#define REPORT_EVERY   1000

typedef struct {
    int L;          /* global grid is L x L */
    int maxh;       /* toppling threshold */
    int max_steps;  /* safety cap on iterations */
    int seed;       /* RNG seed for initial pile */
    char out_file[256];
} Config;

static void parse_args(int argc, char **argv, Config *cfg)
{
    int i;
    cfg->L = DEFAULT_L;
    cfg->maxh = DEFAULT_MAXH;
    cfg->max_steps = DEFAULT_STEPS;
    cfg->seed = 42;
    strncpy(cfg->out_file, "sand.ppm", sizeof(cfg->out_file) - 1);
    cfg->out_file[sizeof(cfg->out_file) - 1] = '\0';

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            cfg->L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--maxh") == 0 && i + 1 < argc) {
            cfg->maxh = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            cfg->max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg->seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(cfg->out_file, argv[++i], sizeof(cfg->out_file) - 1);
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    int dims[2] = {0, 0};
    int periods[2];
    int coords[2];
    MPI_Comm cart_comm;
    int north, south, east, west;
    Config cfg;
    int local_Lx, local_Ly;
    int **oldh, **newh;
    int i, j, step;
    int local_changes, global_changes;
    double t_start, t_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    parse_args(argc, argv, &cfg);

    /* --- Set up 2D Cartesian topology --- */
    MPI_Dims_create(size, 2, dims);
    /* non-periodic in i (rows), periodic in j (columns) */
    periods[0] = 0;
    periods[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);

    if (cfg.L % dims[0] != 0 || cfg.L % dims[1] != 0) {
        if (rank == 0) {
            fprintf(stderr,
                "Error: grid size %d must be divisible by both process-grid "
                "dimensions (%d x %d). Choose a size divisible by %d.\n",
                cfg.L, dims[0], dims[1], dims[0] * dims[1]);
        }
        MPI_Abort(cart_comm, 1);
    }

    local_Lx = cfg.L / dims[0];
    local_Ly = cfg.L / dims[1];

    /* Padded arrays: 1-cell halo on every side */
    oldh = alloc2d_int(local_Lx + 2, local_Ly + 2);
    newh = alloc2d_int(local_Lx + 2, local_Ly + 2);
    if (oldh == NULL || newh == NULL) {
        fprintf(stderr, "Rank %d: allocation failure\n", rank);
        MPI_Abort(cart_comm, 2);
    }

    /* --- Initial condition: deterministic function of GLOBAL (row, col)
     * coordinates, not rank. This is important for correctness testing:
     * seeding per-rank would make the initial pile depend on how many
     * processes you run with (different decomposition = different
     * per-rank RNG streams = different global pile), which would make
     * it impossible to verify that np=1 and np=4 converge to the same
     * physical result. A simple positional hash keeps the initial
     * condition identical regardless of process count. */
    for (i = 1; i <= local_Lx; i++) {
        for (j = 1; j <= local_Ly; j++) {
            int gi = coords[0] * local_Lx + (i - 1);
            int gj = coords[1] * local_Ly + (j - 1);
            unsigned int h = (unsigned int) cfg.seed;
            h = h * 2654435761u + (unsigned int) gi * 40503u;
            h = h * 2654435761u + (unsigned int) gj * 65599u;
            oldh[i][j] = (int) (h % (unsigned int) (cfg.maxh + 3));
        }
    }

    /* Reusable halo buffers, allocated once outside the loop (this was
     * the original bug in the coursework draft — allocating inside the
     * loop tanked performance and caused job timeouts on ARCHER2). */
    int *send_north = malloc(local_Ly * sizeof(int));
    int *recv_north  = malloc(local_Ly * sizeof(int));
    int *send_south = malloc(local_Ly * sizeof(int));
    int *recv_south  = malloc(local_Ly * sizeof(int));
    int *send_west  = malloc(local_Lx * sizeof(int));
    int *recv_west   = malloc(local_Lx * sizeof(int));
    int *send_east  = malloc(local_Lx * sizeof(int));
    int *recv_east   = malloc(local_Lx * sizeof(int));

    MPI_Barrier(cart_comm);
    t_start = MPI_Wtime();

    for (step = 0; step < cfg.max_steps; step++) {
        MPI_Request reqs[8];
        int nreq = 0;

        /* Pack boundary rows/columns */
        for (j = 0; j < local_Ly; j++) {
            send_north[j] = oldh[1][j + 1];
            send_south[j] = oldh[local_Lx][j + 1];
        }
        for (i = 0; i < local_Lx; i++) {
            send_west[i] = oldh[i + 1][1];
            send_east[i] = oldh[i + 1][local_Ly];
        }

        /* Non-blocking halo exchange — all 8 messages posted before
         * any Wait, so the four directions overlap in flight. */
        MPI_Irecv(recv_north, local_Ly, MPI_INT, north, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_south, local_Ly, MPI_INT, south, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_west,  local_Lx, MPI_INT, west,  2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_east,  local_Lx, MPI_INT, east,  3, cart_comm, &reqs[nreq++]);

        MPI_Isend(send_north, local_Ly, MPI_INT, north, 1, cart_comm, &reqs[nreq++]);
        MPI_Isend(send_south, local_Ly, MPI_INT, south, 0, cart_comm, &reqs[nreq++]);
        MPI_Isend(send_west,  local_Lx, MPI_INT, west,  3, cart_comm, &reqs[nreq++]);
        MPI_Isend(send_east,  local_Lx, MPI_INT, east,  2, cart_comm, &reqs[nreq++]);

        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        /* Unpack into halo cells. MPI_PROC_NULL neighbours (the
         * non-periodic top/bottom edges) receive into a buffer that
         * is never written by a real neighbour, so it stays at 0 —
         * sand simply "falls off" the edge, which is the intended
         * boundary condition. */
        if (north != MPI_PROC_NULL) {
            for (j = 0; j < local_Ly; j++) oldh[0][j + 1] = recv_north[j];
        } else {
            for (j = 0; j < local_Ly; j++) oldh[0][j + 1] = 0;
        }
        if (south != MPI_PROC_NULL) {
            for (j = 0; j < local_Ly; j++) oldh[local_Lx + 1][j + 1] = recv_south[j];
        } else {
            for (j = 0; j < local_Ly; j++) oldh[local_Lx + 1][j + 1] = 0;
        }
        for (i = 0; i < local_Lx; i++) oldh[i + 1][0] = recv_west[i];
        for (i = 0; i < local_Lx; i++) oldh[i + 1][local_Ly + 1] = recv_east[i];

        /* Apply the toppling rule */
        local_changes = 0;
        for (i = 1; i <= local_Lx; i++) {
            for (j = 1; j <= local_Ly; j++) {
                int h = oldh[i][j];
                if (h > cfg.maxh) h -= 4;
                if (oldh[i - 1][j] > cfg.maxh) h += 1;
                if (oldh[i + 1][j] > cfg.maxh) h += 1;
                if (oldh[i][j - 1] > cfg.maxh) h += 1;
                if (oldh[i][j + 1] > cfg.maxh) h += 1;

                if (h != oldh[i][j]) local_changes++;
                newh[i][j] = h;
            }
        }

        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, cart_comm);

        /* Swap grids */
        { int **tmp = oldh; oldh = newh; newh = tmp; }

        if (rank == 0 && step % REPORT_EVERY == 0) {
            printf("[step %6d] global changes = %d\n", step, global_changes);
            fflush(stdout);
        }

        if (global_changes == 0) {
            if (rank == 0) {
                printf("Converged after %d steps.\n", step + 1);
            }
            step++;
            break;
        }
    }

    MPI_Barrier(cart_comm);
    t_end = MPI_Wtime();

    if (rank == 0) {
        printf("Ranks: %d (grid %d x %d)  Steps: %d  Wall time: %.2f s\n",
               size, dims[0], dims[1], step, t_end - t_start);
    }

    /* --- Gather final grid to rank 0 for visualisation --- */
    {
        int *local_flat = malloc((size_t) local_Lx * local_Ly * sizeof(int));
        int *global_flat = NULL;
        int *recvcounts = NULL, *displs = NULL;

        for (i = 0; i < local_Lx; i++) {
            for (j = 0; j < local_Ly; j++) {
                local_flat[i * local_Ly + j] = oldh[i + 1][j + 1];
            }
        }

        if (rank == 0) {
            global_flat = malloc((size_t) cfg.L * cfg.L * sizeof(int));
            recvcounts = malloc(size * sizeof(int));
            displs = malloc(size * sizeof(int));
        }

        /* Gather raw blocks (order = rank order); rank 0 then
         * reassembles by Cartesian coordinate. This keeps the
         * communication simple (Gather, not a derived subarray type)
         * at the cost of an O(L^2) reassembly pass on rank 0 only. */
        int local_count = local_Lx * local_Ly;
        int *all_blocks = NULL;
        if (rank == 0) all_blocks = malloc((size_t) size * local_count * sizeof(int));

        MPI_Gather(local_flat, local_count, MPI_INT,
                   all_blocks, local_count, MPI_INT, 0, cart_comm);

        if (rank == 0) {
            int r;
            for (r = 0; r < size; r++) {
                int rc[2];
                MPI_Cart_coords(cart_comm, r, 2, rc);
                int base_i = rc[0] * local_Lx;
                int base_j = rc[1] * local_Ly;
                for (i = 0; i < local_Lx; i++) {
                    for (j = 0; j < local_Ly; j++) {
                        global_flat[(base_i + i) * cfg.L + (base_j + j)] =
                            all_blocks[r * local_count + i * local_Ly + j];
                    }
                }
            }
            write_ppm(cfg.out_file, global_flat, cfg.L, cfg.L, cfg.maxh);
            printf("Wrote final grid to %s\n", cfg.out_file);
            free(global_flat);
            free(all_blocks);
            free(recvcounts);
            free(displs);
        }
        free(local_flat);
    }

    free(send_north); free(recv_north);
    free(send_south); free(recv_south);
    free(send_west);  free(recv_west);
    free(send_east);  free(recv_east);
    free2d_int(oldh);
    free2d_int(newh);

    MPI_Finalize();
    return 0;
}
