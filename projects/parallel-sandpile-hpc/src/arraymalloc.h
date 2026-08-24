/*
 * arraymalloc.h
 *
 * Small helper for allocating contiguous 2D integer arrays as a single
 * block of memory (rather than an array of pointers to separately
 * malloc'd rows). This keeps the halo-exchange code simple (one
 * pointer, row-major indexing) while still allowing h[i][j] syntax.
 *
 * Part of: parallel-sandpile-hpc
 * Author:  Iman Ein Alizadeh
 */

#ifndef ARRAYMALLOC_H
#define ARRAYMALLOC_H

/*
 * Allocate an (nx) x (ny) array of int, indexable as arr[i][j].
 * Returns NULL on failure.
 */
int **alloc2d_int(int nx, int ny);

/*
 * Free an array allocated with alloc2d_int.
 */
void free2d_int(int **arr);

#endif /* ARRAYMALLOC_H */
