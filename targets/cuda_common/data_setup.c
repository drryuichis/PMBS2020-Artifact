#include "../data_setup.h"

#include <float.h>
#include <math.h>

void target_init(struct grid_t grid, uint nsteps,
                 const float *restrict u, const float *restrict v, const float *restrict phi,
                 const float *restrict eta, const float *restrict coefx, const float *restrict coefy,
                 const float *restrict coefz, const float *restrict vp, const float *restrict source)
{
    // Nothing needed
}

void target_finalize(struct grid_t grid, uint nsteps,
                     const float *restrict u, const float *restrict v, const float *restrict phi,
                     const float *restrict eta, const float *restrict coefx, const float *restrict coefy,
                     const float *restrict coefz, const float *restrict vp, const float *restrict source)
{
    // Nothing needed
}

void kernel_add_source(struct grid_t grid,
                       float *restrict u, const float *restrict source, llint istep,
                       llint sx, llint sy, llint sz)
{
    // Nothing needed
}

extern void find_min_max_u_cuda(
    const float *restrict u, llint u_size, float *restrict min_u, float *restrict max_u
);

void find_min_max_u(struct grid_t grid,
                    const float *restrict u, float *restrict min_u, float *restrict max_u)
{
    const llint nx = grid.nx;
    const llint ny = grid.ny;
    const llint nz = grid.nz;
    const llint lx = grid.lx;
    const llint ly = grid.ly;
    const llint lz = grid.lz;

    llint u_size = (nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
    find_min_max_u_cuda(u, u_size, min_u, max_u);
}
