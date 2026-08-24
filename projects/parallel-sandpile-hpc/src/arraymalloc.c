/*
 * arraymalloc.c
 *
 * See arraymalloc.h for description.
 */

#include <stdlib.h>
#include "arraymalloc.h"

int **alloc2d_int(int nx, int ny)
{
    int **arr;
    int *block;
    int i;

    arr = (int **) malloc(nx * sizeof(int *));
    if (arr == NULL) {
        return NULL;
    }

    block = (int *) calloc((size_t) nx * (size_t) ny, sizeof(int));
    if (block == NULL) {
        free(arr);
        return NULL;
    }

    for (i = 0; i < nx; i++) {
        arr[i] = &block[i * ny];
    }

    return arr;
}

void free2d_int(int **arr)
{
    if (arr == NULL) {
        return;
    }
    /* arr[0] points at the start of the single contiguous block */
    free(arr[0]);
    free(arr);
}
