#!/bin/bash

source time_run.sh

PWD=`pwd`
REPORT=$PWD/../results/PMBS_V100.md

NVCC_VERSION=`nvcc --version | tail -n 1 | cut -d ' ' -f 6`
NVCC_NAME="CUDA $NVCC_VERSION"

cd ..

echo "## Time Measurements on Tesla V100 for Grid Size of 1000" > $REPORT
echo "" >> $REPORT

NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_opt-gpu             "CUDA Global Memory (3D 8x8x8 Blocking)"                            nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8_gpml_opt-gpu        "CUDA Global Memory (3D 8x8x8 Blocking) with global pml tiling"     nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_32x32x1_opt-gpu           "CUDA Global Memory (3D 32x32x1 Blocking)"                          nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_16x16x4_opt-gpu           "CUDA Global Memory (3D 16x16x4 Blocking)"                          nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x4_opt-gpu             "CUDA Global Memory (3D 8x8x4 Blocking)"                            nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_4x4x4_opt-gpu             "CUDA Global Memory (3D 4x4x4 Blocking)"                            nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_gmem_8x8x8i_9x9x9pm_opt-gpu    "CUDA Global Memory (3D 9x9x9 Blocking for PML regions)"            nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_u_opt-gpu                 "CUDA Shared Memory on U"                                           nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_eta_1_opt-gpu             "CUDA Shared Memory on Eta with one checking"                       nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_smem_eta_3_opt-gpu             "CUDA Shared Memory on Eta with three checkings"                    nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_micikevicius_8x8_opt-gpu       "CUDA Paulius Micikevicius (2D 8x8)"                                nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_micikevicius_16x16_opt-gpu     "CUDA Paulius Micikevicius (2D 16x16)"                              nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70 -maxrregcount=64"    time_run    cuda_micikevicius_32x32_opt-gpu     "CUDA Paulius Micikevicius (2D 32x32)"                              nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_micikevicius_32x16_opt-gpu     "CUDA Paulius Micikevicius (2D 32x16)"                              nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_micikevicius_16x32_opt-gpu     "CUDA Paulius Micikevicius (2D 16x32)"                              nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70 -maxrregcount=64"    time_run    cuda_micikevicius_64x16_opt-gpu     "CUDA Paulius Micikevicius (2D 64x16)"                              nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70 -maxrregcount=64"    time_run    cuda_micikevicius_16x64_opt-gpu     "CUDA Paulius Micikevicius (2D 16x64)"                              nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_micikevicius_gmem_opt-gpu      "CUDA Paulius Micikevicius with global memory"                      nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_nguyen25d_16x16_opt-gpu        "CUDA Anthony Nguyen (2.5D 16x16)"                                  nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_nguyen25d_16x8_opt-gpu         "CUDA Anthony Nguyen (2.5D 16x8)"                                   nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_nguyen25d_8x16_opt-gpu         "CUDA Anthony Nguyen (2.5D 8x16)"                                   nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_nguyen25d_8x8_opt-gpu          "CUDA Anthony Nguyen (2.5D 8x8)"                                    nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_nguyen25d_irt_16x16_opt-gpu    "CUDA Anthony Nguyen (2.5D 16x16) using index rotation"             nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_nguyen25d_idx_16x16_opt-gpu    "CUDA Anthony Nguyen (2.5D 16x16) using index offset"               nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_matsu25d_8x8_opt-gpu           "CUDA Kazuaki Matsumuta (2.5D 8x8)"                                 nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_matsu25d_16x8_opt-gpu          "CUDA Kazuaki Matsumuta (2.5D 16x8)"                                nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_matsu25d_16x16_opt-gpu         "CUDA Kazuaki Matsumuta (2.5D 16x16)"                               nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_matsu25d_32x16_opt-gpu         "CUDA Kazuaki Matsumuta (2.5D 32x16)"                               nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70 -maxrregcount=64"    time_run    cuda_matsu25d_32x32_opt-gpu         "CUDA Kazuaki Matsumuta (2.5D 32x32)"                               nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"
NVCCFLAGS="-arch=sm_70                 "    time_run    cuda_semi_opt-gpu                   "CUDA Semi-Stencil on 3D Blocks"                                    nvcc    "$NVCC_NAME"    $REPORT     "--grid 1000 --warm-up --niters 5 $*"

