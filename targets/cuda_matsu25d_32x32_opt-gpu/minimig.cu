#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "../../constants.h"

#define N_RADIUS 4
#define N_THREADS_PER_PLANE_DIM_X 32
#define N_THREADS_PER_PLANE_DIM_Y 32

#define Z(val) ((val > 8) ? (val - 9) : val)

#define INNER_STENCIL(R0, R1, R2, R3, R4, R5, R6, R7, R8) i++; if (i >= x4) { return; } \
    zrs##R8 = u[IDX3_l(i+N_RADIUS,j,k)]; \
    __syncthreads(); \
    if (threadIdx.y < 2 * N_RADIUS) { \
        s_u[threadIdx.y + (threadIdx.y/N_RADIUS)*N_THREADS_PER_PLANE_DIM_Y][suk] = \
            u[IDX3_l(i,j0+threadIdx.y+(threadIdx.y/N_RADIUS)*N_THREADS_PER_PLANE_DIM_Y-N_RADIUS,k)]; \
    } \
    if (threadIdx.x < 2 * N_RADIUS) { \
        s_u[suj][threadIdx.x + (threadIdx.x/N_RADIUS)*N_THREADS_PER_PLANE_DIM_X] = \
            u[IDX3_l(i,j,k0+threadIdx.x+(threadIdx.x/N_RADIUS)*N_THREADS_PER_PLANE_DIM_X-N_RADIUS)]; \
    } \
    s_u[suj][suk] = u[IDX3_l(i,j,k)]; \
    __syncthreads(); \
    if (j < y4 && k < z4) { \
        float lap = __fmaf_rn(coef0, zrs##R4 \
                    , __fmaf_rn(coefx_1, __fadd_rn(zrs##R5,zrs##R3) \
                    , __fmaf_rn(coefy_1, __fadd_rn(s_u[suj+1][suk],s_u[suj-1][suk]) \
                    , __fmaf_rn(coefz_1, __fadd_rn(s_u[suj][suk+1],s_u[suj][suk-1]) \
                    , __fmaf_rn(coefx_2, __fadd_rn(zrs##R6,zrs##R2) \
                    , __fmaf_rn(coefy_2, __fadd_rn(s_u[suj+2][suk],s_u[suj-2][suk]) \
                    , __fmaf_rn(coefz_2, __fadd_rn(s_u[suj][suk+2],s_u[suj][suk-2]) \
                    , __fmaf_rn(coefx_3, __fadd_rn(zrs##R7,zrs##R1) \
                    , __fmaf_rn(coefy_3, __fadd_rn(s_u[suj+3][suk],s_u[suj-3][suk]) \
                    , __fmaf_rn(coefz_3, __fadd_rn(s_u[suj][suk+3],s_u[suj][suk-3]) \
                    , __fmaf_rn(coefx_4, __fadd_rn(zrs##R8,zrs##R0) \
                    , __fmaf_rn(coefy_4, __fadd_rn(s_u[suj+4][suk],s_u[suj-4][suk]) \
                    , __fmul_rn(coefz_4, __fadd_rn(s_u[suj][suk+4],s_u[suj][suk-4]) \
        ))))))))))))); \
        v[IDX3_l(i,j,k)] = __fmaf_rn(2.f, zrs##R4, \
            __fmaf_rn(vp[IDX3(i,j,k)], lap, -v[IDX3_l(i,j,k)]) \
        ); \
    }

#define PML_STENCIL(R0, R1, R2, R3, R4, R5, R6, R7, R8) i++; if (i >= x4) { return; } \
    zrs##R8 = u[IDX3_l(i+N_RADIUS,j,k)]; \
    __syncthreads(); \
    if (threadIdx.y < 2 * N_RADIUS) { \
        s_u[threadIdx.y + (threadIdx.y/N_RADIUS)*N_THREADS_PER_PLANE_DIM_Y][suk] = \
            u[IDX3_l(i,j0+threadIdx.y+(threadIdx.y/N_RADIUS)*N_THREADS_PER_PLANE_DIM_Y-N_RADIUS,k)]; \
    } \
    if (threadIdx.x < 2 * N_RADIUS) { \
        s_u[suj][threadIdx.x + (threadIdx.x/N_RADIUS)*N_THREADS_PER_PLANE_DIM_X] = \
            u[IDX3_l(i,j,k0+threadIdx.x+(threadIdx.x/N_RADIUS)*N_THREADS_PER_PLANE_DIM_X-N_RADIUS)]; \
    } \
    s_u[suj][suk] = u[IDX3_l(i,j,k)]; \
    __syncthreads(); \
    if (j < y4 && k < z4) { \
        float lap = __fmaf_rn(coef0, zrs##R4 \
                    , __fmaf_rn(coefx_1, __fadd_rn(zrs##R5,zrs##R3) \
                    , __fmaf_rn(coefy_1, __fadd_rn(s_u[suj+1][suk],s_u[suj-1][suk]) \
                    , __fmaf_rn(coefz_1, __fadd_rn(s_u[suj][suk+1],s_u[suj][suk-1]) \
                    , __fmaf_rn(coefx_2, __fadd_rn(zrs##R6,zrs##R2) \
                    , __fmaf_rn(coefy_2, __fadd_rn(s_u[suj+2][suk],s_u[suj-2][suk]) \
                    , __fmaf_rn(coefz_2, __fadd_rn(s_u[suj][suk+2],s_u[suj][suk-2]) \
                    , __fmaf_rn(coefx_3, __fadd_rn(zrs##R7,zrs##R1) \
                    , __fmaf_rn(coefy_3, __fadd_rn(s_u[suj+3][suk],s_u[suj-3][suk]) \
                    , __fmaf_rn(coefz_3, __fadd_rn(s_u[suj][suk+3],s_u[suj][suk-3]) \
                    , __fmaf_rn(coefx_4, __fadd_rn(zrs##R8,zrs##R0) \
                    , __fmaf_rn(coefy_4, __fadd_rn(s_u[suj+4][suk],s_u[suj-4][suk]) \
                    , __fmul_rn(coefz_4, __fadd_rn(s_u[suj][suk+4],s_u[suj][suk-4]) \
        ))))))))))))); \
        const float s_eta_c = eta[IDX3_eta1(i,j,k)]; \
        v[IDX3_l(i,j,k)] = __fdiv_rn( \
            __fmaf_rn( \
                __fmaf_rn(2.f, s_eta_c, \
                    __fsub_rn(2.f, \
                        __fmul_rn(s_eta_c, s_eta_c) \
                    ) \
                ), \
                zrs##R4, \
                __fmaf_rn( \
                    vp[IDX3(i,j,k)], \
                    __fadd_rn(lap, phi[IDX3(i,j,k)]), \
                    -v[IDX3_l(i,j,k)] \
                ) \
            ), \
            __fmaf_rn(2.f, s_eta_c, 1.f) \
        ); \
        phi[IDX3(i,j,k)] = __fdiv_rn( \
                __fsub_rn( \
                    phi[IDX3(i,j,k)], \
                    __fmaf_rn( \
                    __fmul_rn( \
                        __fsub_rn(eta[IDX3_eta1(i+1,j,k)], eta[IDX3_eta1(i-1,j,k)]), \
                        __fsub_rn(zrs##R5,zrs##R3) \
                    ), hdx_2, \
                    __fmaf_rn( \
                    __fmul_rn( \
                        __fsub_rn(eta[IDX3_eta1(i,j+1,k)], eta[IDX3_eta1(i,j-1,k)]), \
                        __fsub_rn(s_u[suj+1][suk], s_u[suj-1][suk]) \
                    ), hdy_2, \
                    __fmul_rn( \
                        __fmul_rn( \
                            __fsub_rn(eta[IDX3_eta1(i,j,k+1)], eta[IDX3_eta1(i,j,k-1)]), \
                            __fsub_rn(s_u[suj][suk+1], s_u[suj][suk-1]) \
                        ), \
                    hdz_2) \
                    )) \
                ) \
            , \
            __fadd_rn(1.f, s_eta_c) \
        ); \
    }

__global__ void target_inner_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta
) {
    __shared__ float s_u[N_THREADS_PER_PLANE_DIM_Y+2*N_RADIUS][N_THREADS_PER_PLANE_DIM_X+2*N_RADIUS];

    const llint j0 = y3 + blockIdx.y * blockDim.y;
    const llint k0 = z3 + blockIdx.x * blockDim.x;

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float zrs0, zrs1, zrs2, zrs3, zrs4, zrs5, zrs6, zrs7, zrs8;

    // Preparation
    zrs0 = u[IDX3_l(x3-4,j,k)];
    zrs1 = u[IDX3_l(x3-3,j,k)];
    zrs2 = u[IDX3_l(x3-2,j,k)];
    zrs3 = u[IDX3_l(x3-1,j,k)];
    zrs4 = u[IDX3_l(x3+0,j,k)];
    zrs5 = u[IDX3_l(x3+1,j,k)];
    zrs6 = u[IDX3_l(x3+2,j,k)];
    zrs7 = u[IDX3_l(x3+3,j,k)];

    llint i = x3-1;
    while (true) {
        INNER_STENCIL(0, 1, 2, 3, 4, 5, 6, 7, 8);
        INNER_STENCIL(1, 2, 3, 4, 5, 6, 7, 8, 0);
        INNER_STENCIL(2, 3, 4, 5, 6, 7, 8, 0, 1);
        INNER_STENCIL(3, 4, 5, 6, 7, 8, 0, 1, 2);
        INNER_STENCIL(4, 5, 6, 7, 8, 0, 1, 2, 3);
        INNER_STENCIL(5, 6, 7, 8, 0, 1, 2, 3, 4);
        INNER_STENCIL(6, 7, 8, 0, 1, 2, 3, 4, 5);
        INNER_STENCIL(7, 8, 0, 1, 2, 3, 4, 5, 6);
        INNER_STENCIL(8, 0, 1, 2, 3, 4, 5, 6, 7);
    }
}

__global__ void target_pml_3d_kernel(
    llint nx, llint ny, llint nz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta
) {
    __shared__ float s_u[N_THREADS_PER_PLANE_DIM_Y+2*N_RADIUS][N_THREADS_PER_PLANE_DIM_X+2*N_RADIUS];

    const llint j0 = y3 + blockIdx.y * blockDim.y;
    const llint k0 = z3 + blockIdx.x * blockDim.x;

    const llint j = j0 + threadIdx.y;
    const llint k = k0 + threadIdx.x;

    const llint suj = threadIdx.y + N_RADIUS;
    const llint suk = threadIdx.x + N_RADIUS;

    float zrs0, zrs1, zrs2, zrs3, zrs4, zrs5, zrs6, zrs7, zrs8;

    // Preparation
    zrs0 = u[IDX3_l(x3-4,j,k)];
    zrs1 = u[IDX3_l(x3-3,j,k)];
    zrs2 = u[IDX3_l(x3-2,j,k)];
    zrs3 = u[IDX3_l(x3-1,j,k)];
    zrs4 = u[IDX3_l(x3+0,j,k)];
    zrs5 = u[IDX3_l(x3+1,j,k)];
    zrs6 = u[IDX3_l(x3+2,j,k)];
    zrs7 = u[IDX3_l(x3+3,j,k)];

    llint i = x3-1;
    while (true) {
        PML_STENCIL(0, 1, 2, 3, 4, 5, 6, 7, 8);
        PML_STENCIL(1, 2, 3, 4, 5, 6, 7, 8, 0);
        PML_STENCIL(2, 3, 4, 5, 6, 7, 8, 0, 1);
        PML_STENCIL(3, 4, 5, 6, 7, 8, 0, 1, 2);
        PML_STENCIL(4, 5, 6, 7, 8, 0, 1, 2, 3);
        PML_STENCIL(5, 6, 7, 8, 0, 1, 2, 3, 4);
        PML_STENCIL(6, 7, 8, 0, 1, 2, 3, 4, 5);
        PML_STENCIL(7, 8, 0, 1, 2, 3, 4, 5, 6);
        PML_STENCIL(8, 0, 1, 2, 3, 4, 5, 6, 7);
    }
}

__global__ void kernel_add_source_kernel(float *g_u, llint idx, float source) {
    g_u[idx] += source;
}

extern "C" void target(
    uint nsteps, double *time_kernel,
    llint nx, llint ny, llint nz,
    llint x1, llint x2, llint x3, llint x4, llint x5, llint x6,
    llint y1, llint y2, llint y3, llint y4, llint y5, llint y6,
    llint z1, llint z2, llint z3, llint z4, llint z5, llint z6,
    llint lx, llint ly, llint lz,
    llint sx, llint sy, llint sz,
    float hdx_2, float hdy_2, float hdz_2,
    const float *__restrict__ coefx, const float *__restrict__ coefy, const float *__restrict__ coefz,
    float *__restrict__ u, const float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta, const float *__restrict__ source
) {
    struct timespec start, end;

    const llint size_u = (nx + 2 * lx) * (ny + 2 * ly) * (nz + 2 * lz);
    const llint size_v = size_u;
    const llint size_phi = nx*ny*nz;
    const llint size_vp = size_phi;
    const llint size_eta = (nx+2)*(ny+2)*(nz+2);

    const llint size_u_ext = (nx + 2 * lx)
                           * ((((ny+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y + 1) * N_THREADS_PER_PLANE_DIM_Y) + 2 * ly)
                           * ((((nz+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X + 1) * N_THREADS_PER_PLANE_DIM_X) + 2 * lz);

    float *d_u, *d_v, *d_vp, *d_phi, *d_eta;
    cudaMalloc(&d_u, sizeof(float) * size_u_ext);
    cudaMalloc(&d_v, sizeof(float) * size_u_ext);
    cudaMalloc(&d_vp, sizeof(float) * size_vp);
    cudaMalloc(&d_phi, sizeof(float) * size_phi);
    cudaMalloc(&d_eta, sizeof(float) * size_eta);

    cudaMemcpy(d_u, u, sizeof(float) * size_u, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(float) * size_v, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vp, vp, sizeof(float) * size_vp, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, phi, sizeof(float) * size_phi, cudaMemcpyHostToDevice);
    cudaMemcpy(d_eta, eta, sizeof(float) * size_eta, cudaMemcpyHostToDevice);

    const llint xmin = 0; const llint xmax = nx;
    const llint ymin = 0; const llint ymax = ny;

    dim3 threadsPerBlock(N_THREADS_PER_PLANE_DIM_X, N_THREADS_PER_PLANE_DIM_Y, 1);

    int num_streams = 7;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking);
    }

    const uint npo = 100;
    for (uint istep = 1; istep <= nsteps; ++istep) {
        clock_gettime(CLOCK_REALTIME, &start);

        dim3 n_block_front(
            (z2-z1+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (ny+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y);
        target_pml_3d_kernel<<<n_block_front, threadsPerBlock, 0, streams[1]>>>(nx,ny,nz,
            xmin,xmax,ymin,ymax,z1,z2,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_top(
            (z4-z3+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (y2-y1+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y);
        target_pml_3d_kernel<<<n_block_top, threadsPerBlock, 0, streams[2]>>>(nx,ny,nz,
            xmin,xmax,y1,y2,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_left(
            (z4-z3+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (y4-y3+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y,
            1);
        target_pml_3d_kernel<<<n_block_left, threadsPerBlock, 0, streams[3]>>>(nx,ny,nz,
            x1,x2,y3,y4,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_center(
            (z4-z3+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (y4-y3+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y);
        target_inner_3d_kernel<<<n_block_center, threadsPerBlock, 0, streams[0]>>>(nx,ny,nz,
            x3,x4,y3,y4,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_right(
            (z4-z3+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (y4-y3+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y,
            1);
        target_pml_3d_kernel<<<n_block_right, threadsPerBlock, 0, streams[4]>>>(nx,ny,nz,
            x5,x6,y3,y4,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_bottom(
            (z4-z3+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (y6-y5+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y,
            1);
        target_pml_3d_kernel<<<n_block_bottom, threadsPerBlock, 0, streams[5]>>>(nx,ny,nz,
            xmin,xmax,y5,y6,z3,z4,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        dim3 n_block_back(
            (z6-z5+N_THREADS_PER_PLANE_DIM_X-1) / N_THREADS_PER_PLANE_DIM_X,
            (ny+N_THREADS_PER_PLANE_DIM_Y-1) / N_THREADS_PER_PLANE_DIM_Y,
            1);
        target_pml_3d_kernel<<<n_block_back, threadsPerBlock, 0, streams[6]>>>(nx,ny,nz,
            xmin,xmax,ymin,ymax,z5,z6,
            lx,ly,lz,
            hdx_2, hdy_2, hdz_2,
            coefx[0]+coefy[0]+coefz[0],
            coefx[1], coefx[2], coefx[3], coefx[4],
            coefy[1], coefy[2], coefy[3], coefy[4],
            coefz[1], coefz[2], coefz[3], coefz[4],
            d_u, d_v, d_vp,
            d_phi, d_eta);

        for (int i = 0; i < num_streams; i++) {
            cudaStreamSynchronize(streams[i]);
        }

        kernel_add_source_kernel<<<1, 1>>>(d_v, IDX3_l(sx,sy,sz), source[istep]);
        clock_gettime(CLOCK_REALTIME, &end);
        *time_kernel += (end.tv_sec  - start.tv_sec) +
                        (double)(end.tv_nsec - start.tv_nsec) / 1.0e9;

        float *t = d_u;
        d_u = d_v;
        d_v = t;

        // Print out
        if (istep % npo == 0) {
            printf("time step %u / %u\n", istep, nsteps);
        }
    }


    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }


    cudaMemcpy(u, d_u, sizeof(float) * size_u, cudaMemcpyDeviceToHost);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_vp);
    cudaFree(d_phi);
    cudaFree(d_eta);
}
