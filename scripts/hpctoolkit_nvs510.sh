#!/bin/bash

source hpctoolkit_run.sh

cd ..

HPCTK_LM=-lineinfo

NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_8x8x8_opt-gpu             nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_8x8x8_gpml_opt-gpu        nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_32x32x1_opt-gpu           nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_16x16x4_opt-gpu           nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_8x8x4_opt-gpu             nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_4x4x4_opt-gpu             nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_gmem_8x8x8i_9x9x9pm_opt-gpu    nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_smem_u_opt-gpu                 nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_smem_eta_1_opt-gpu             nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_smem_eta_3_opt-gpu             nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_micikevicius_8x8_opt-gpu       nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_micikevicius_16x16_opt-gpu     nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30 -maxrregcount=64"    hpctoolkit_run  cuda_micikevicius_32x32_opt-gpu     nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_micikevicius_32x16_opt-gpu     nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_micikevicius_16x32_opt-gpu     nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30 -maxrregcount=64"    hpctoolkit_run  cuda_micikevicius_64x16_opt-gpu     nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30 -maxrregcount=64"    hpctoolkit_run  cuda_micikevicius_16x64_opt-gpu     nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_micikevicius_gmem_opt-gpu      nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_nguyen25d_16x16_opt-gpu        nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_nguyen25d_16x8_opt-gpu         nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_nguyen25d_8x16_opt-gpu         nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_nguyen25d_8x8_opt-gpu          nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_nguyen25d_irt_16x16_opt-gpu    nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_nguyen25d_idx_16x16_opt-gpu    nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_matsu25d_8x8_opt-gpu           nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_matsu25d_16x8_opt-gpu          nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_matsu25d_16x16_opt-gpu         nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_matsu25d_32x16_opt-gpu         nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30 -maxrregcount=64"    hpctoolkit_run  cuda_matsu25d_32x32_opt-gpu         nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
# NVCCFLAGS="-arch=sm_30                 "    hpctoolkit_run  cuda_semi_opt-gpu                   nvcc    $HPCTK_LM   "-e gpu=nvidia,pc"  "--grid 300 $*"
