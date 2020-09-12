#ifndef DATA_SETUP_H
#define DATA_SETUP_H

#include "../constants.h"
#include "../grid.h"

void target_init(struct grid_t grid, uint nsteps,
                 const float *restrict u, const float *restrict v, const float *restrict phi,
                 const float *restrict eta, const float *restrict coefx, const float *restrict coefy,
                 const float *restrict coefz, const float *restrict vp, const float *restrict source);

void target_finalize(struct grid_t grid, uint nsteps,
                     const float *restrict u, const float *restrict v, const float *restrict phi,
                     const float *restrict eta, const float *restrict coefx, const float *restrict coefy,
                     const float *restrict coefz, const float *restrict vp, const float *restrict source);

void kernel_add_source(struct grid_t grid,
                       float *restrict u, const float *restrict source, llint istep,
                       llint sx, llint sy, llint sz);

void find_min_max_u(struct grid_t grid,
                    const float *restrict u, float *restrict min_u, float *restrict max_u);

#endif
