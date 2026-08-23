/*
 * sandpile_2d.c
 * Parallel Abelian Sandpile Model — MPI 2D Domain Decomposition
 *
 * MSc HPC and Data Science, EPCC, University of Edinburgh
 *
 * Compile: mpicc -O3 -o sandpile_2d sandpile_2d.c arraymalloc.c sandio.c -lm
 * Run:     mpirun -np <P> ./sandpile_2d <grid_size>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "arraymalloc.h"
#include "sandio.h"

#define MAXH 3

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int L = (argc > 1) ? atoi(argv[1]) : 512;

    /* --- 2D Cartesian topology --- */
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 1};  /* non-periodic in x, periodic in y */
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    /* --- Local subdomain sizes --- */
    int local_Lx = L / dims[0];
    int local_Ly = L / dims[1];

    /* --- Neighbour ranks --- */
    int north, south, west, east;
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west,  &east);

    /* --- Allocate local grids with halo (ghost cells) --- */
    int **oldh = (int **)arraymalloc2d(local_Lx + 2, local_Ly + 2, sizeof(int));
    int **newh = (int **)arraymalloc2d(local_Lx + 2, local_Ly + 2, sizeof(int));

    /* --- Initialise: pile everything in the centre --- */
    memset(&oldh[0][0], 0, (local_Lx + 2) * (local_Ly + 2) * sizeof(int));
    memset(&newh[0][0], 0, (local_Lx + 2) * (local_Ly + 2) * sizeof(int));

    int cx = L / 2, cy = L / 2;
    int global_start_x = coords[0] * local_Lx;
    int global_start_y = coords[1] * local_Ly;
    for (int i = 1; i <= local_Lx; i++) {
        for (int j = 1; j <= local_Ly; j++) {
            int gx = global_start_x + i - 1;
            int gy = global_start_y + j - 1;
            if (gx == cx && gy == cy) {
                oldh[i][j] = L * L;
            }
        }
    }

    /* --- Halo buffers (allocated once, outside loop) --- */
    int *send_north = malloc(local_Ly * sizeof(int));
    int *recv_north = malloc(local_Ly * sizeof(int));
    int *send_south = malloc(local_Ly * sizeof(int));
    int *recv_south = malloc(local_Ly * sizeof(int));
    int *send_west  = malloc(local_Lx * sizeof(int));
    int *recv_west  = malloc(local_Lx * sizeof(int));
    int *send_east  = malloc(local_Lx * sizeof(int));
    int *recv_east  = malloc(local_Lx * sizeof(int));

    double t_start = MPI_Wtime();
    int global_changes = 1;
    long step = 0;

    while (global_changes > 0) {

        /* --- Pack and exchange halos (non-blocking) --- */
        for (int j = 0; j < local_Ly; j++) {
            send_north[j] = oldh[1][j + 1];
            send_south[j] = oldh[local_Lx][j + 1];
        }
        for (int i = 0; i < local_Lx; i++) {
            send_west[i] = oldh[i + 1][1];
            send_east[i] = oldh[i + 1][local_Ly];
        }

        MPI_Request reqs[8];
        int nreq = 0;

        MPI_Isend(send_north, local_Ly, MPI_INT, north, 0, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_south, local_Ly, MPI_INT, south, 0, cart_comm, &reqs[nreq++]);
        MPI_Isend(send_south, local_Ly, MPI_INT, south, 1, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_north, local_Ly, MPI_INT, north, 1, cart_comm, &reqs[nreq++]);
        MPI_Isend(send_west,  local_Lx, MPI_INT, west,  2, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_east,  local_Lx, MPI_INT, east,  2, cart_comm, &reqs[nreq++]);
        MPI_Isend(send_east,  local_Lx, MPI_INT, east,  3, cart_comm, &reqs[nreq++]);
        MPI_Irecv(recv_west,  local_Lx, MPI_INT, west,  3, cart_comm, &reqs[nreq++]);

        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

        /* --- Unpack received halos --- */
        for (int j = 0; j < local_Ly; j++) {
            if (north != MPI_PROC_NULL) oldh[0][j + 1]           = recv_north[j];
            if (south != MPI_PROC_NULL) oldh[local_Lx + 1][j + 1] = recv_south[j];
        }
        for (int i = 0; i < local_Lx; i++) {
            if (west != MPI_PROC_NULL) oldh[i + 1][0]           = recv_west[i];
            if (east != MPI_PROC_NULL) oldh[i + 1][local_Ly + 1] = recv_east[i];
        }

        /* --- Update sandpile --- */
        int local_changes = 0;
        for (int i = 1; i <= local_Lx; i++) {
            for (int j = 1; j <= local_Ly; j++) {
                int h = oldh[i][j];
                if (h > MAXH)           h -= 4;
                if (oldh[i-1][j] > MAXH) h += 1;
                if (oldh[i+1][j] > MAXH) h += 1;
                if (oldh[i][j-1] > MAXH) h += 1;
                if (oldh[i][j+1] > MAXH) h += 1;
                if (h != oldh[i][j]) local_changes++;
                newh[i][j] = h;
            }
        }

        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, cart_comm);

        /* --- Swap grids --- */
        int **tmp = oldh; oldh = newh; newh = tmp;
        step++;

        if (step % 10000 == 0 && rank == 0) {
            printf("Step %ld: global_changes = %d\n", step, global_changes);
            fflush(stdout);
        }
    }

    double t_end = MPI_Wtime();
    if (rank == 0) {
        printf("Converged after %ld steps in %.2f seconds\n", step, t_end - t_start);
    }

    /* --- Gather and write output (rank 0 only) --- */
    /* ... MPI_Gatherv output omitted for brevity — see sandio.c ... */

    free(send_north); free(recv_north);
    free(send_south); free(recv_south);
    free(send_west);  free(recv_west);
    free(send_east);  free(recv_east);
    free(oldh); free(newh);

    MPI_Finalize();
    return 0;
}
