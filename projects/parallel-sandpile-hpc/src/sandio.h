/*
 * sandio.h
 *
 * Writes a global sandpile height array to a binary PPM (.ppm) file
 * for visual inspection. Convert to PNG afterwards with, e.g.:
 *
 *   convert sand.ppm sand.png     (ImageMagick)
 *
 * Height values are clamped and mapped to a simple 4-colour palette
 * (empty / settling / near-critical / toppling) so the avalanche
 * fronts are visible at a glance.
 *
 * Part of: parallel-sandpile-hpc
 * Author:  Iman Ein Alizadeh
 */

#ifndef SANDIO_H
#define SANDIO_H

/*
 * Write an nx * ny grid (row-major, contiguous int block pointed to
 * by 'data') to a binary PPM file at 'filename'. maxh is the
 * toppling threshold used to choose colour bands.
 *
 * Returns 0 on success, non-zero on failure.
 */
int write_ppm(const char *filename, const int *data, int nx, int ny, int maxh);

#endif /* SANDIO_H */
