/*
 * sandio.c
 *
 * See sandio.h for description.
 */

#include <stdio.h>
#include "sandio.h"

/* Simple 4-band colour palette, indexed by height relative to maxh */
static void colour_for_height(int h, int maxh, unsigned char *r, unsigned char *g, unsigned char *b)
{
    if (h <= 0) {
        /* empty */
        *r = 10; *g = 10; *b = 40;
    } else if (h < maxh - 1) {
        /* settling — blue to cyan */
        *r = 20; *g = 120; *b = 200;
    } else if (h < maxh) {
        /* near-critical — amber */
        *r = 230; *g = 160; *b = 20;
    } else {
        /* at/above threshold — bright red (about to topple) */
        *r = 230; *g = 30; *b = 30;
    }
}

int write_ppm(const char *filename, const int *data, int nx, int ny, int maxh)
{
    FILE *fp;
    int i, j;
    unsigned char r, g, b;

    fp = fopen(filename, "wb");
    if (fp == NULL) {
        return 1;
    }

    fprintf(fp, "P6\n%d %d\n255\n", ny, nx);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            colour_for_height(data[i * ny + j], maxh, &r, &g, &b);
            fputc(r, fp);
            fputc(g, fp);
            fputc(b, fp);
        }
    }

    fclose(fp);
    return 0;
}
