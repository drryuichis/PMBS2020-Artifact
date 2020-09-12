
**[2020] Total E&P RT USA, LLC. and Rice University**
**All Rights Reserved.**

**NOTICE**:  All information contained herein is, and remains
the property of Total E&P RT USA, LLC. and Rice University.  
The intellectual and technical concepts contained
herein are proprietary to Total E&P RT USA, LLC. and Rice University 
and may be covered by U.S. and Foreign Patents,
patents in process, and are protected by trade secret or copyright law.

----

The code is provided as it is, use it at your own risk and expect no support, the authors are not liable under any circumstances.
Feedback is welcome.

contact: ryuichi@rice.edu

# PMBS2020-Artifact

This repository provides the artifact for our paper "Accelerating High-Order Stencil on GPUs" (available at https://arxiv.org/abs/2009.04619). This artifact repository (https://github.com/rsrice/PMBS2020-Artifact) contains our implementations for a 25-point seismic modeling stencil in CUDA along with code to apply the boundary conditions. The scripts used in our empirical evaluation are also included for reproducibility.

## Environment

The details of our experiment environment can be found in the paper.
In short, we conduct our own experiments on three machines:
IBM POWER9/NVIDIA Tesla V100 32GB,
IBM POWER8NVL/NVIDIA Tesla P100 16GB,
and Intel Xeon E3-1245 v6/NVIDIA NVS 510 2GB.

## Setup

At first, clone this repository to your local machine:

```
git clone https://github.com/rsrice/PMBS2020-Artifact
```

## Compilation

There are several CUDA implementations in this repository inside the `targets` folder,
and we refer to each implementation using its sub-directory name.

To compile the kernel, at the root of this repository, run 
`NVCC=<nvcc_flags> TARGET=<kernel_identifier> COMPILER=nvcc make clean all`.
For example, to compile `cuda_gmem_8x8x8_opt-gpu`, use the following command:

```
NVCCFLAGS="-arch=sm_70" TARGET=cuda_gmem_8x8x8_opt-gpu COMPILER=nvcc make clean all
```

After success compilation,
a binary with the name `main_cuda_gmem_8x8x8_opt-gpu_nvcc` will be generated at the root of the folder.

## Execution

The binary takes several command line arguments, main ones are:
* `--grid=N` will change the grid size (default to `100`) of the stencil,
* `--nsteps=N` to adjust the number of loop steps (default to `1000`),
* `--niters=N` allows to run the whole execution multiple times and to take the average execution time (default to `1`),
* `--warm-up` toggles whether we warm up the execution running one whole iteration once (default to `false`).

As our default values to these flags are intended for easing development,
in our experiments, we apply `--grid 1000 --warm-up --niters 5` to all the executions.

## Evaluation

To ease evaluation, go to the folder with all scripts:

```
cd PMBS2020-Artifact/scripts
```

### Time Measurement

For each machine, we use a specific `pmbs_<machine_identifier>.sh` to run all implementations and capture the time measurement results. For example, on V100, simply running `pmbs_v100.sh` will save all the results into `results` folder.

### HPCToolkit

Similarly, run `hpctoolkit_<machine_identifier>.sh` for generating HPCToolkit databases for all implementations.

### Nsight Compute

We run `ns_<machine_identifier>.sh` to instument our code using Nsight Compute.

### NVProf

`nvprof_<machine_identifier>.sh` provides kernel profiling results used in our roofline performance analysis.

### Empirical Roofline Toolkit (ERT)

We used an ERT fork (available at https://github.com/rsrice/cs-roofline-toolkit-fork) with better Python 3 support and other minor changes to better fit our environment.

