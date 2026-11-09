/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * Standalone extraction of the original GPU integrate kernel of the grid
 * backend, as of CP2K revision c05efa7fa7 (before the separable kernels were
 * introduced). The kernel and the device helpers it needs are copied verbatim
 * from
 *
 *   - src/grid/gpu/grid_gpu_integrate.cu       (integrate_kernel)
 *   - src/grid/gpu/grid_gpu_internal_header.h  (helpers, smem_task_reduced)
 *   - src/grid/gpu/grid_gpu_context.h          (task_info, kernel_params)
 *
 * kernel_params / kernel_buffers are trimmed down to the fields that the
 * integrate kernel actually reads. task_info is kept as is so that the memory
 * layout (and global memory traffic) matches the production code.
 *
 * The driver generates random tasks on a periodic grid, runs the kernel and
 * checks the result against a serial CPU port of the same algorithm that uses
 * the generic monomial evaluation (so the hand unrolled switch cases of the
 * kernel are validated against it).
 *
 * Build:
 *   nvcc  -O3 -arch=sm_70 -o integrate_original.x
 *         integrate_original_standalone.cu
 *   hipcc -O3 --offload-arch=gfx90a -o integrate_original.x
 *         integrate_original_standalone.cu
 *
 * Usage:
 *   ./integrate_original.x [-n ntasks] [-g grid_points] [-l lmax]
 *                          [-d lp_diff] [-o cell] [-D 0|1 distributed]
 *                          [-r nrep] [-s seed]
 *
 *   -o 0: non orthogonal cell, 1: axis aligned orthorhombic (default),
 *      2: rotated orthorhombic (handled as non orthogonal)
 */

#include <algorithm>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset
#define gpuMemcpyToSymbol(sym, src, size)                                      \
  hipMemcpyToSymbol(HIP_SYMBOL(sym), src, size)
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDestroy hipEventDestroy
#else
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset
#define gpuMemcpyToSymbol(sym, src, size) cudaMemcpyToSymbol(sym, src, size)
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDestroy cudaEventDestroy
#endif

#define GPU_CHECK(cmd)                                                         \
  do {                                                                         \
    const gpuError_t error = (cmd);                                            \
    if (error != gpuSuccess) {                                                 \
      fprintf(stderr, "GPU error: %s at %s:%d\n", gpuGetErrorString(error),    \
              __FILE__, __LINE__);                                             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

namespace rocm_backend {

/*******************************************************************************
 * Data structures (from grid_gpu_context.h)
 ******************************************************************************/
struct task_info {
  int level;
  int iatom;
  int jatom;
  int iset;
  int jset;
  int ipgf;
  int jpgf;
  int ikind, jkind;
  int border_mask;
  int block_num;
  int max_cab_size{0};
  double radius;
  double ra[3], rb[3], rp[3];
  double rab2;
  double zeta, zetb, zetp, prefactor, off_diag_twice;
  double rab[3];
  int lp_max{0};
  size_t coef_offset{0};
  int la_max, lb_max, la_min, lb_min, first_coseta, first_cosetb, ncoseta,
      ncosetb, nsgfa, nsgfb, nsgf_seta, nsgf_setb, maxcoa, maxcob;
  int sgfa, sgfb, subblock_offset;
  double3 roffset;
  int3 cube_size;
  int3 lb_cube;
  int3 cube_center;
  double discrete_radius;
  bool apply_border_mask;
  bool block_transposed;
  bool skip_task;
};

// trimmed: only the buffers used by the integrate kernel.
struct kernel_buffers {
  double *grid{nullptr};
  double *coef{nullptr};
};

// trimmed: only the parameters used by the integrate kernel.
struct kernel_params {
  int first_task{0};
  int3 grid_full_size_ = make_int3(0, 0, 0);
  int3 grid_local_size_ = make_int3(0, 0, 0);
  int3 grid_lower_corner_ = make_int3(0, 0, 0);
  int3 grid_border_width_ = make_int3(0, 0, 0);
  double dh_[9];
  double dh_inv_[9];
  task_info *tasks;
  char la_min_diff{0};
  char lb_min_diff{0};
  char la_max_diff{0};
  char lb_max_diff{0};
  kernel_buffers buffers_dev;
  int ntasks{0};
};

/*******************************************************************************
 * Helpers (from grid_gpu_internal_header.h)
 ******************************************************************************/
struct orbital {
  int l[3];
};

__constant__ orbital coset_inv[1330];

template <typename T, typename T3> struct smem_task_reduced {
  // radius: true cutoff used for the sphere-membership test.
  // discrete_radius: box-aligned cutoff (always >= radius) used only for
  // sizing the cube / trimming box boundaries.
  T radius, discrete_radius;
  T norm_lattice_vector_z_2, norm_lattice_vector_y_2;
  T norm_inverse_lattice_vector_y, norm_inverse_lattice_vector_x;
  int3 cube_center, lb_cube, cube_size, window_size, window_shift;
  T3 roffset;
  T zetp;
  char lp;
  bool apply_border_mask;
};

__host__ __device__ __inline__ int ncoset(const int l) {
  static const int table[] = {1,  // l=0
                              4,  // l=1
                              10, // l=2 ...
                              20,  35,  56,  84,  120, 165, 220,  286,
                              364, 455, 560, 680, 816, 969, 1140, 1330};
  return table[l];
}

__host__ __device__ __inline__ int coset(int lx, int ly, int lz) {
  const int l = lx + ly + lz;
  if (l == 0) {
    return 0;
  } else {
    return ncoset(l - 1) + ((l - lx) * (l - lx + 1)) / 2 + lz;
  }
}

__inline__ __device__ unsigned int block_index() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

__inline__ __device__ unsigned int thread_global_index() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__inline__ __device__ double3
compute_coordinates(const double *__restrict__ dh_, const double x,
                    const double y, const double z) {

  double3 r3;
  // I make no distinction between orthorhombic and non orthorhombic
  // cases

  r3.x = z * dh_[6] + y * dh_[3] + x * dh_[0];
  r3.y = z * dh_[7] + y * dh_[4] + x * dh_[1];
  r3.z = z * dh_[8] + y * dh_[5] + x * dh_[2];
  return r3;
}

__host__ __inline__ __device__ void
convert_to_lattice_coordinates(const double *dh_inv_,
                               const double3 *__restrict__ const rp,
                               double3 *__restrict__ rp_c) {
  rp_c->x = dh_inv_[0] * rp->x + dh_inv_[3] * rp->y + dh_inv_[6] * rp->z;
  rp_c->y = dh_inv_[1] * rp->x + dh_inv_[4] * rp->y + dh_inv_[7] * rp->z;
  rp_c->z = dh_inv_[2] * rp->x + dh_inv_[5] * rp->y + dh_inv_[8] * rp->z;
}

__host__ __inline__ __device__ void
convert_from_lattice_coordinates_to_cartesian(
    const double *__restrict__ dh_, const double3 *__restrict__ const rp,
    double3 *__restrict__ rp_c) {
  rp_c->x = dh_[0] * rp->x + dh_[3] * rp->y + dh_[6] * rp->z;
  rp_c->y = dh_[1] * rp->x + dh_[4] * rp->y + dh_[7] * rp->z;
  rp_c->z = dh_[2] * rp->x + dh_[5] * rp->y + dh_[8] * rp->z;
}

template <typename T, typename T3, bool orthorhombic_>
__inline__ T compute_cube_properties(
    const T radius, const T *const __restrict__ dh_,
    const T *const __restrict__ dh_inv_, const T3 *__restrict__ rp,
    T3 *__restrict__ roffset, int3 *__restrict__ cubecenter,
    int3 *__restrict__ lb_cube, int3 *__restrict__ cube_size) {

  /* center of the gaussian in the lattice coordinates */
  T3 rp1, rp2, rp3;

  /* it is in the lattice vector frame */
  convert_to_lattice_coordinates(dh_inv_, rp, &rp1);

  /* compute the grid point that is the closest to the sphere center. */
  cubecenter->x = std::floor(rp1.x);
  cubecenter->y = std::floor(rp1.y);
  cubecenter->z = std::floor(rp1.z);

  /* seting up the cube parameters */

  if (orthorhombic_) {
    T norm1, norm2, norm3;
    norm1 = dh_[0] * dh_[0] + dh_[1] * dh_[1] + dh_[2] * dh_[2];
    norm2 = dh_[3] * dh_[3] + dh_[4] * dh_[4] + dh_[5] * dh_[5];
    norm3 = dh_[6] * dh_[6] + dh_[7] * dh_[7] + dh_[8] * dh_[8];

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);
    norm3 = std::sqrt(norm3);

    const T disr_radius =
        std::min(norm1, std::min(norm2, norm3)) *
        (std::max(1,
                  (int)ceil(radius / std::min(norm1, std::min(norm2, norm3)))));

    rp2.x = cubecenter->x;
    rp2.y = cubecenter->y;
    rp2.z = cubecenter->z;

    /* convert the cube center from grid points coordinates to cartesian */
    convert_from_lattice_coordinates_to_cartesian(dh_, &rp2, roffset);
    /* cube center */
    roffset->x -= rp->x;
    roffset->y -= rp->y;
    roffset->z -= rp->z;

    rp2.x = disr_radius;
    rp2.y = disr_radius;
    rp2.z = disr_radius;

    /* it is in the lattice vector frame */
    convert_to_lattice_coordinates(dh_inv_, &rp2, &rp3);
    /* lower and upper bounds */
    lb_cube->x = std::ceil(-1e-8 - rp3.x);
    lb_cube->y = std::ceil(-1e-8 - rp3.y);
    lb_cube->z = std::ceil(-1e-8 - rp3.z);

    /* it is in the lattice vector frame */
    convert_to_lattice_coordinates(dh_inv_, roffset, &rp2);

    /* Express the offset in lattice coordinates */
    roffset->x = rp2.x;
    roffset->y = rp2.y;
    roffset->z = rp2.z;

    /* compute the cube size ignoring periodicity */
    cube_size->x = 2 - 2 * lb_cube->x;
    cube_size->y = 2 - 2 * lb_cube->y;
    cube_size->z = 2 - 2 * lb_cube->z;
    return disr_radius;
  } else {
    int3 ub_cube;

    lb_cube->x = INT_MAX;
    ub_cube.x = INT_MIN;
    lb_cube->y = INT_MAX;
    ub_cube.y = INT_MIN;
    lb_cube->z = INT_MAX;
    ub_cube.z = INT_MIN;

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        for (int k = -1; k <= 1; k++) {
          T3 r;
          r.x = rp->x + ((T)i) * radius;
          r.y = rp->y + ((T)j) * radius;
          r.z = rp->z + ((T)k) * radius;
          convert_to_lattice_coordinates(dh_inv_, &r, roffset);

          lb_cube->x = std::min(lb_cube->x, (int)std::floor(roffset->x));
          ub_cube.x = std::max(ub_cube.x, (int)std::ceil(roffset->x));

          lb_cube->y = std::min(lb_cube->y, (int)std::floor(roffset->y));
          ub_cube.y = std::max(ub_cube.y, (int)std::ceil(roffset->y));

          lb_cube->z = std::min(lb_cube->z, (int)std::floor(roffset->z));
          ub_cube.z = std::max(ub_cube.z, (int)std::ceil(roffset->z));
        }
      }
    }
    /* compute the cube size ignoring periodicity */
    cube_size->x = ub_cube.x - lb_cube->x;
    cube_size->y = ub_cube.y - lb_cube->y;
    cube_size->z = ub_cube.z - lb_cube->z;

    /* compute the offset in lattice coordinates */

    roffset->x = cubecenter->x - rp1.x;
    roffset->y = cubecenter->y - rp1.y;
    roffset->z = cubecenter->z - rp1.z;

    // shift the boundaries compared to the cube center so that the
    // specialization ortho / non ortho is minimal
    lb_cube->x -= cubecenter->x;
    lb_cube->y -= cubecenter->y;
    lb_cube->z -= cubecenter->z;

    return radius;
  }
}

__host__ __inline__ __device__ void
compute_window_size(const int3 grid_size, const int border_mask,
                    const int3 &border_width, int3 &window_size,
                    int3 &window_shift) {
  window_shift.x = 0;
  window_shift.y = 0;
  window_shift.z = 0;

  window_size.x = grid_size.x - 1;
  window_size.y = grid_size.y - 1;
  window_size.z = grid_size.z - 1;

  if (border_mask & (1 << 0))
    window_shift.x += border_width.x;
  if (border_mask & (1 << 1))
    window_size.x -= border_width.x;
  if (border_mask & (1 << 2))
    window_shift.y += border_width.y;
  if (border_mask & (1 << 3))
    window_size.y -= border_width.y;
  if (border_mask & (1 << 4))
    window_shift.z += border_width.z;
  if (border_mask & (1 << 5))
    window_size.z -= border_width.z;
}

__host__ __inline__ __device__ int wrap_grid_index(const int idx,
                                                   const int full_size) {
  int idx2 = idx % full_size;
  if (idx2 < 0)
    idx2 += full_size;
  return idx2;
}

template <typename T, typename T3, bool distributed__>
__device__ __inline__ void
setup_task_cube_center(const kernel_params &dev_,
                       smem_task_reduced<T, T3> &task) {
  task.cube_center.x += task.lb_cube.x - dev_.grid_lower_corner_.x;
  task.cube_center.y += task.lb_cube.y - dev_.grid_lower_corner_.y;
  task.cube_center.z += task.lb_cube.z - dev_.grid_lower_corner_.z;

  if (distributed__) {
    if (task.apply_border_mask) {
      compute_window_size(
          dev_.grid_local_size_,
          dev_.tasks[dev_.first_task + block_index()].border_mask,
          dev_.grid_border_width_, task.window_size, task.window_shift);
    }
  }
}

template <typename T, typename T3>
__host__ __device__ __inline__ void
fill_task_reduced_fields(const kernel_params &dev, const task_info &glb_task,
                         smem_task_reduced<T, T3> &task) {
  task.zetp = glb_task.zeta + glb_task.zetb;
  task.radius = glb_task.radius;
  task.discrete_radius = glb_task.discrete_radius;

  // angular momentum range for the actual collocate/integrate operation.
  task.lp =
      glb_task.la_max + dev.la_max_diff + glb_task.lb_max + dev.lb_max_diff;

  task.cube_size.x = glb_task.cube_size.x;
  task.cube_size.y = glb_task.cube_size.y;
  task.cube_size.z = glb_task.cube_size.z;

  task.cube_center.x = glb_task.cube_center.x;
  task.cube_center.y = glb_task.cube_center.y;
  task.cube_center.z = glb_task.cube_center.z;

  task.roffset.x = glb_task.roffset.x;
  task.roffset.y = glb_task.roffset.y;
  task.roffset.z = glb_task.roffset.z;

  task.lb_cube.x = glb_task.lb_cube.x;
  task.lb_cube.y = glb_task.lb_cube.y;
  task.lb_cube.z = glb_task.lb_cube.z;

  task.norm_lattice_vector_z_2 = dev.dh_[6] * dev.dh_[6] +
                                 dev.dh_[7] * dev.dh_[7] +
                                 dev.dh_[8] * dev.dh_[8];
  task.norm_lattice_vector_y_2 = dev.dh_[3] * dev.dh_[3] +
                                 dev.dh_[4] * dev.dh_[4] +
                                 dev.dh_[5] * dev.dh_[5];

  task.norm_inverse_lattice_vector_y =
      sqrt(dev.dh_inv_[3] * dev.dh_inv_[3] + dev.dh_inv_[4] * dev.dh_inv_[4] +
           dev.dh_inv_[5] * dev.dh_inv_[5]);
  task.norm_inverse_lattice_vector_x =
      sqrt(dev.dh_inv_[0] * dev.dh_inv_[0] + dev.dh_inv_[1] * dev.dh_inv_[1] +
           dev.dh_inv_[2] * dev.dh_inv_[2]);

  task.apply_border_mask = glb_task.apply_border_mask;
}

// Same as fill_smem_task_reduced in grid_gpu_internal_header.h, with the field
// copy factored out so that the CPU reference can share it.
template <typename T, typename T3>
__device__ __inline__ void
fill_smem_task_reduced(const kernel_params &dev, const int task_id,
                       smem_task_reduced<T, T3> &task) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    fill_task_reduced_fields(dev, dev.tasks[task_id], task);
  }
  __syncthreads();
}

template <typename T, typename T3>
__host__ __device__ __inline__ T
calculate_ymix_ymax_boundaries(smem_task_reduced<T, T3> &task, const int z,
                               int &ymin, int &ymax) {
  T kremain = 0.0;
  ymin = (2 * (z + task.lb_cube.z) - 1) / 2;
  ymin *= ymin;
  kremain = task.discrete_radius * task.discrete_radius -
            ((T)ymin) * task.norm_lattice_vector_z_2;
  ymin = ceil(-1.0e-8 -
              sqrt(fmax(0.0, kremain)) * task.norm_inverse_lattice_vector_y);
  ymax = 1 - ymin - task.lb_cube.y;
  ymin = ymin - task.lb_cube.y;
  return kremain;
}

template <typename T, typename T3>
__host__ __device__ __inline__ void
calculate_xmin_xmax_boundaries(smem_task_reduced<T, T3> &task, const int y,
                               const T kremain, int &xmin, int &xmax) {
  xmin = (2 * (y + task.lb_cube.y) - 1) / 2;
  xmin *= xmin;
  xmin = ceil(-1.0e-8 -
              sqrt(fmax(0.0, kremain - xmin * task.norm_lattice_vector_y_2)) *
                  task.norm_inverse_lattice_vector_x);
  xmax = 1 - xmin - task.lb_cube.x;
  xmin -= task.lb_cube.x;
}

/*******************************************************************************
 * The integrate kernel (verbatim from grid_gpu_integrate.cu)
 ******************************************************************************/
template <typename T, typename T3, bool distributed__, bool orthogonal_,
          int lbatch = 20>
__global__
__launch_bounds__(64) void integrate_kernel(const kernel_params dev_) {
  if (dev_.tasks[dev_.first_task + block_index()].skip_task)
    return;

  const int tid = thread_global_index();

  // __shared__ T dh_inv_[9];
  __shared__ T dh_[9];

  // for (int d = tid; d < 9; d += blockDim.x * blockDim.y * blockDim.z)
  //   dh_inv_[d] = dev_.dh_inv_[d];

  if (tid < 9)
    dh_[tid] = dev_.dh_[tid];

  __shared__ smem_task_reduced<T, T3> task;
  fill_smem_task_reduced(dev_, dev_.first_task + block_index(), task);

  if (tid == 0) {
    setup_task_cube_center<T, T3, distributed__>(dev_, task);
  }
  __syncthreads();

  __shared__ T accumulator[lbatch][64];

  // we use a multi pass algorithm here because shared memory usage (or
  // register) would become too high for high angular momentum

  const short int size_loop =
      (ncoset(task.lp) / lbatch + ((ncoset(task.lp) % lbatch) != 0)) * lbatch;
  const short int length = ncoset(task.lp);
  for (int ico = 0; ico < size_loop; ico += lbatch) {
#pragma unroll lbatch
    for (int i = 0; i < lbatch; i++)
      accumulator[i][tid] = 0.0;

    __syncthreads();

    for (int z = threadIdx.z; z < task.cube_size.z; z += blockDim.z) {
      int z2 = wrap_grid_index(z + task.cube_center.z, dev_.grid_full_size_.z);

      if (distributed__) {
        // known at compile time. Will be stripped away
        if (task.apply_border_mask) {
          /* check if the point is within the window */
          if ((z2 < task.window_shift.z) || (z2 > task.window_size.z)) {
            continue;
          }
        }
      }

      /* compute the coordinates of the point in atomic coordinates */
      int ymin = 0;
      int ymax = task.cube_size.y - 1;
      T kremain = 0.0;

      if (orthogonal_ && !task.apply_border_mask) {
        kremain = calculate_ymix_ymax_boundaries(task, z, ymin, ymax);
      }

      for (int y = ymin + threadIdx.y; y <= ymax; y += blockDim.y) {
        int y2 =
            wrap_grid_index(y + task.cube_center.y, dev_.grid_full_size_.y);

        if (distributed__) {
          /* check if the point is within the window */
          if (task.apply_border_mask) {
            if ((y2 < task.window_shift.y) || (y2 > task.window_size.y)) {
              continue;
            }
          }
        }

        int xmin = 0;
        int xmax = task.cube_size.x - 1;

        if (orthogonal_ && !task.apply_border_mask) {
          calculate_xmin_xmax_boundaries<T, T3>(task, y, kremain, xmin, xmax);
        }

        for (int x = xmin + threadIdx.x; x <= xmax; x += blockDim.x) {
          int x2 =
              wrap_grid_index(x + task.cube_center.x, dev_.grid_full_size_.x);

          if (distributed__) {
            /* check if the point is within the window */
            if (task.apply_border_mask) {
              if ((x2 < task.window_shift.x) || (x2 > task.window_size.x)) {
                continue;
              }
            }
          }

          // I make no distinction between orthogonal and non orthogonal
          // cases
          T3 r3;

          r3 = compute_coordinates(dh_, (x + task.lb_cube.x + task.roffset.x),
                                   (y + task.lb_cube.y + task.roffset.y),
                                   (z + task.lb_cube.z + task.roffset.z));
          // check if the point is inside the sphere or not. Note that it does
          // not apply for the orthogonal case when the full sphere is inside
          // the region of interest.
          const T r3x2 = r3.x * r3.x;
          const T r3y2 = r3.y * r3.y;
          const T r3z2 = r3.z * r3.z;

          if (distributed__) {
            // check if the point is inside the sphere or not. Note that it does
            // not apply for the orthorhombic case when the full sphere is
            // inside the region of interest.

            if (((task.radius * task.radius) <= (r3x2 + r3y2 + r3z2)) &&
                (!orthogonal_ || task.apply_border_mask))
              continue;
          } else {
            // we do not need to do this test for the orthorhombic case
            if ((!orthogonal_) &&
                ((task.radius * task.radius) <= (r3x2 + r3y2 + r3z2)))
              continue;
          }

          // read the next point assuming that reading is non blocking untill
          // the register is actually needed for computation. This is true on
          // NVIDIA hardware

          const int grid_index =
              (z2 * dev_.grid_local_size_.y + y2) * dev_.grid_local_size_.x +
              x2;
          T grid_value = __ldg(&dev_.buffers_dev.grid[grid_index]);

          const T r3xy = r3.x * r3.y;
          const T r3xz = r3.x * r3.z;
          const T r3yz = r3.y * r3.z;

          grid_value *= exp(-(r3x2 + r3y2 + r3z2) * task.zetp);

          switch (ico / lbatch) {
          case 0: {
            accumulator[0][tid] += grid_value;

            if (task.lp >= 1) {
              accumulator[1][tid] += grid_value * r3.x;
              accumulator[2][tid] += grid_value * r3.y;
              accumulator[3][tid] += grid_value * r3.z;
            }

            if (task.lp >= 2) {
              accumulator[4][tid] += grid_value * r3x2;
              accumulator[5][tid] += grid_value * r3xy;
              accumulator[6][tid] += grid_value * r3xz;
              accumulator[7][tid] += grid_value * r3y2;
              accumulator[8][tid] += grid_value * r3yz;
              accumulator[9][tid] += grid_value * r3z2;
            }
            if (task.lp >= 3) {
              T tmp = grid_value * r3x2;
              accumulator[10][tid] += tmp * r3.x;
              accumulator[11][tid] += tmp * r3.y;
              accumulator[12][tid] += tmp * r3.z;
              tmp = grid_value * r3.x;
              accumulator[13][tid] += tmp * r3y2;
              accumulator[14][tid] += tmp * r3yz;
              accumulator[15][tid] += tmp * r3z2;
              tmp = grid_value * r3y2;
              accumulator[16][tid] += tmp * r3.y;
              accumulator[17][tid] += tmp * r3.z;
              tmp = grid_value * r3z2;
              accumulator[18][tid] += tmp * r3.y;
              accumulator[19][tid] += tmp * r3.z;
            }
          } break;
          case 1: {
            if (task.lp >= 4) {
              T tmp = grid_value * r3x2;
              accumulator[0][tid] += tmp * r3x2;
              accumulator[1][tid] += tmp * r3xy;
              accumulator[2][tid] += tmp * r3xz;
              accumulator[3][tid] += tmp * r3y2;
              accumulator[4][tid] += tmp * r3yz;
              accumulator[5][tid] += tmp * r3z2;
              tmp = grid_value * r3y2;
              accumulator[6][tid] += tmp * r3xy;
              accumulator[7][tid] += tmp * r3xz;
              tmp = grid_value * r3z2;
              accumulator[8][tid] += tmp * r3xy;
              accumulator[9][tid] += tmp * r3xz;
            }
            if (task.lp >= 4) {
              T tmp = grid_value * r3y2;
              accumulator[10][tid] += tmp * r3y2;
              accumulator[11][tid] += tmp * r3yz;
              accumulator[12][tid] += tmp * r3z2;
              accumulator[13][tid] += grid_value * r3yz * r3z2;
              accumulator[14][tid] += grid_value * r3z2 * r3z2;
            }
            if (task.lp >= 5) {
              T tmp = grid_value * r3x2 * r3x2;
              accumulator[15][tid] += tmp * r3.x;
              accumulator[16][tid] += tmp * r3.y;
              accumulator[17][tid] += tmp * r3.z;
              tmp = grid_value * r3x2;
              accumulator[18][tid] += tmp * r3.x * r3y2;
              accumulator[19][tid] += tmp * r3xy * r3.z;
            }
          } break;
          case 2: {
            T tmp = grid_value * r3x2;
            accumulator[0][tid] += tmp * r3.x * r3z2;
            accumulator[1][tid] += tmp * r3y2 * r3.y;
            accumulator[2][tid] += tmp * r3y2 * r3.z;
            accumulator[3][tid] += tmp * r3.y * r3z2;
            accumulator[4][tid] += tmp * r3z2 * r3.z;
            tmp = grid_value * r3.x * r3y2;
            accumulator[5][tid] += tmp * r3y2;
            accumulator[6][tid] += tmp * r3yz;
            accumulator[7][tid] += tmp * r3z2;
            tmp = grid_value * r3.x * r3z2;
            accumulator[8][tid] += tmp * r3yz;
            accumulator[9][tid] += tmp * r3z2;
            tmp = grid_value * r3y2 * r3.y;
            accumulator[10][tid] += tmp * r3y2;
            accumulator[11][tid] += tmp * r3yz;
            accumulator[12][tid] += tmp * r3z2;
            accumulator[13][tid] += grid_value * r3y2 * r3z2 * r3.z;
            accumulator[14][tid] += grid_value * r3.y * r3z2 * r3z2;
            accumulator[15][tid] += grid_value * r3z2 * r3z2 * r3.z;
            if (task.lp >= 6) {
              tmp = grid_value * r3x2 * r3x2;
              accumulator[16][tid] += tmp * r3x2; // x^6
              accumulator[17][tid] += tmp * r3xy; // x^5 y
              accumulator[18][tid] += tmp * r3xz; // x^5 z
              accumulator[19][tid] += tmp * r3y2; // x^4 y^2
            }
          } break;
          case 3: {
            T tmp = grid_value * r3x2;
            accumulator[0][tid] += tmp * r3x2 * r3yz;  // x^4 y z
            accumulator[1][tid] += tmp * r3x2 * r3z2;  // x^4 z^2
            accumulator[2][tid] += tmp * r3y2 * r3xy;  // x^3 y^3
            accumulator[3][tid] += tmp * r3y2 * r3xz;  // x^3 y^2 z
            accumulator[4][tid] += tmp * r3xy * r3z2;  // x^3 y z^2
            accumulator[5][tid] += tmp * r3z2 * r3xz;  // x^3 z^3
            accumulator[6][tid] += tmp * r3y2 * r3y2;  // x^2 y^4
            accumulator[7][tid] += tmp * r3y2 * r3yz;  // x^3 y^2 z
            accumulator[8][tid] += tmp * r3y2 * r3z2;  // x^2 y^2 z^2
            accumulator[9][tid] += tmp * r3z2 * r3yz;  // x^2 y z^3
            accumulator[10][tid] += tmp * r3z2 * r3z2; // x^2 z^4
            tmp = grid_value * r3y2 * r3y2;
            accumulator[11][tid] += tmp * r3xy; // x y^5
            accumulator[12][tid] += tmp * r3xz; // x y^4 z
            accumulator[13][tid] +=
                grid_value * r3y2 * r3xy * r3z2; // x y^3 z^2
            accumulator[14][tid] +=
                grid_value * r3y2 * r3z2 * r3xz; // x y^2 z^3
            accumulator[15][tid] += grid_value * r3xy * r3z2 * r3z2; // x y z^4
            accumulator[16][tid] += grid_value * r3z2 * r3z2 * r3xz; // x z^5
            accumulator[17][tid] += tmp * r3y2;                      // y^6
            accumulator[18][tid] += tmp * r3yz;                      // y^5 z
            accumulator[19][tid] += tmp * r3z2;                      // y^4 z^2
          } break;
          default:
            for (int ic = 0; (ic < lbatch) && ((ic + ico) < length); ic++) {
              auto &co = coset_inv[ic + ico];
              T tmp = 1.0;
              for (int po = 0; po < (co.l[2] >> 1); po++)
                tmp *= r3z2;
              if (co.l[2] & 0x1)
                tmp *= r3.z;
              for (int po = 0; po < (co.l[1] >> 1); po++)
                tmp *= r3y2;
              if (co.l[1] & 0x1)
                tmp *= r3.y;
              for (int po = 0; po < (co.l[0] >> 1); po++)
                tmp *= r3x2;
              if (co.l[0] & 0x1)
                tmp *= r3.x;
              accumulator[ic][tid] += tmp * grid_value;
            }
            break;
          }
        }
      }
    }

    const int max_i = min(length - ico, lbatch);

    __syncthreads();

    // we know there is only 1 wavefront in each block lbatch threads could
    // reduce the values saved by all threads of the warp and save results do a
    // shuffle_down by hand

    if (tid < 32) {
      for (int i = 0; i < max_i; i++) {
        // Load local value
        T val = accumulator[i][tid] + accumulator[i][tid + 32];

        // Warp-level reduction (32 lanes) After this loop, lane 0 of warp 0
        // holds the result All threads should execute this operation so
        // __activemask() is incorrect here.

        // The ifdef statement is needed because hip does not necessarily
        // support the instruction either. Reverting back to __shfl_down is
        // required.
        //
        for (int offset = 16; offset > 0; offset >>= 1) {
#if defined(__CUDACC__)
          val += __shfl_down_sync(0xffffffff, val, offset);
#else
          val += __shfl_down(val, offset);
#endif
        }

        if (tid == 0)
          accumulator[i][0] = val;
      }
#if defined(__CUDACC__)
      __syncwarp();
#endif
    }

#if !defined(__CUDACC__)
    __syncthreads();
#endif

    if (tid < min(length - ico, lbatch)) {
      const size_t coef_offset =
          dev_.tasks[dev_.first_task + block_index()].coef_offset;
      dev_.buffers_dev.coef[coef_offset + tid + ico] = accumulator[tid][0];
    }
    __syncthreads();
  }
}

/*******************************************************************************
 * Host launcher (same dispatch as context_info::integrate_one_grid_level)
 ******************************************************************************/
void launch_integrate(const kernel_params &params, const int ntasks,
                      const bool distributed, const bool orthogonal) {
  const dim3 threads_per_block(4, 4, 4);
  if (distributed) {
    if (orthogonal) {
      integrate_kernel<double, double3, true, true, 20>
          <<<ntasks, threads_per_block, 0>>>(params);
    } else {
      integrate_kernel<double, double3, true, false, 20>
          <<<ntasks, threads_per_block, 0>>>(params);
    }
  } else {
    if (orthogonal) {
      integrate_kernel<double, double3, false, true, 20>
          <<<ntasks, threads_per_block, 0>>>(params);
    } else {
      integrate_kernel<double, double3, false, false, 20>
          <<<ntasks, threads_per_block, 0>>>(params);
    }
  }
}

} // namespace rocm_backend

using namespace rocm_backend;

/*******************************************************************************
 * Serial CPU port of integrate_kernel, using the generic monomial evaluation.
 ******************************************************************************/
static void integrate_reference(const kernel_params &dev,
                                const std::vector<task_info> &tasks,
                                const std::vector<orbital> &coset_inv_host,
                                const std::vector<double> &grid,
                                const bool distributed, const bool orthogonal,
                                std::vector<double> &coef) {
  for (int itask = 0; itask < (int)tasks.size(); itask++) {
    const task_info &glb_task = tasks[itask];
    if (glb_task.skip_task)
      continue;

    smem_task_reduced<double, double3> task;
    fill_task_reduced_fields(dev, glb_task, task);
    task.cube_center.x += task.lb_cube.x - dev.grid_lower_corner_.x;
    task.cube_center.y += task.lb_cube.y - dev.grid_lower_corner_.y;
    task.cube_center.z += task.lb_cube.z - dev.grid_lower_corner_.z;
    if (distributed && task.apply_border_mask)
      compute_window_size(dev.grid_local_size_, glb_task.border_mask,
                          dev.grid_border_width_, task.window_size,
                          task.window_shift);

    const int length = ncoset(task.lp);
    std::vector<double> acc(length, 0.0);

    for (int z = 0; z < task.cube_size.z; z++) {
      const int z2 =
          wrap_grid_index(z + task.cube_center.z, dev.grid_full_size_.z);
      if (distributed && task.apply_border_mask &&
          ((z2 < task.window_shift.z) || (z2 > task.window_size.z)))
        continue;

      int ymin = 0, ymax = task.cube_size.y - 1;
      double kremain = 0.0;
      if (orthogonal && !task.apply_border_mask)
        kremain = calculate_ymix_ymax_boundaries(task, z, ymin, ymax);

      for (int y = ymin; y <= ymax; y++) {
        const int y2 =
            wrap_grid_index(y + task.cube_center.y, dev.grid_full_size_.y);
        if (distributed && task.apply_border_mask &&
            ((y2 < task.window_shift.y) || (y2 > task.window_size.y)))
          continue;

        int xmin = 0, xmax = task.cube_size.x - 1;
        if (orthogonal && !task.apply_border_mask)
          calculate_xmin_xmax_boundaries(task, y, kremain, xmin, xmax);

        for (int x = xmin; x <= xmax; x++) {
          const int x2 =
              wrap_grid_index(x + task.cube_center.x, dev.grid_full_size_.x);
          if (distributed && task.apply_border_mask &&
              ((x2 < task.window_shift.x) || (x2 > task.window_size.x)))
            continue;

          const double xs = x + task.lb_cube.x + task.roffset.x;
          const double ys = y + task.lb_cube.y + task.roffset.y;
          const double zs = z + task.lb_cube.z + task.roffset.z;
          double r[3];
          r[0] = zs * dev.dh_[6] + ys * dev.dh_[3] + xs * dev.dh_[0];
          r[1] = zs * dev.dh_[7] + ys * dev.dh_[4] + xs * dev.dh_[1];
          r[2] = zs * dev.dh_[8] + ys * dev.dh_[5] + xs * dev.dh_[2];
          const double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

          const bool outside = (task.radius * task.radius) <= r2;
          if (distributed) {
            if (outside && (!orthogonal || task.apply_border_mask))
              continue;
          } else if (!orthogonal && outside) {
            continue;
          }

          const int grid_index =
              (z2 * dev.grid_local_size_.y + y2) * dev.grid_local_size_.x + x2;
          const double gv = grid[grid_index] * exp(-r2 * task.zetp);

          for (int ic = 0; ic < length; ic++) {
            const orbital &co = coset_inv_host[ic];
            acc[ic] += gv * pow(r[0], co.l[0]) * pow(r[1], co.l[1]) *
                       pow(r[2], co.l[2]);
          }
        }
      }
    }

    for (int ic = 0; ic < length; ic++)
      coef[glb_task.coef_offset + ic] = acc[ic];
  }
}

static void invert3x3(const double *a, double *inv) {
  const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) -
                     a[1] * (a[3] * a[8] - a[5] * a[6]) +
                     a[2] * (a[3] * a[7] - a[4] * a[6]);
  inv[0] = (a[4] * a[8] - a[5] * a[7]) / det;
  inv[1] = (a[2] * a[7] - a[1] * a[8]) / det;
  inv[2] = (a[1] * a[5] - a[2] * a[4]) / det;
  inv[3] = (a[5] * a[6] - a[3] * a[8]) / det;
  inv[4] = (a[0] * a[8] - a[2] * a[6]) / det;
  inv[5] = (a[2] * a[3] - a[0] * a[5]) / det;
  inv[6] = (a[3] * a[7] - a[4] * a[6]) / det;
  inv[7] = (a[1] * a[6] - a[0] * a[7]) / det;
  inv[8] = (a[0] * a[4] - a[1] * a[3]) / det;
}

static size_t align_up_elems(const size_t n, const size_t a) {
  return ((n + a - 1) / a) * a;
}

int main(int argc, char **argv) {
  int ntasks = 10000;
  int npts = 96;
  int lmax = 3;
  int lp_diff = 0;
  int cell = 1;
  bool distributed = false;
  int nrep = 10;
  unsigned seed = 42;

  int opt;
  while ((opt = getopt(argc, argv, "n:g:l:d:o:D:r:s:h")) != -1) {
    switch (opt) {
    case 'n':
      ntasks = atoi(optarg);
      break;
    case 'g':
      npts = atoi(optarg);
      break;
    case 'l':
      lmax = atoi(optarg);
      break;
    case 'd':
      lp_diff = atoi(optarg);
      break;
    case 'o':
      cell = atoi(optarg);
      break;
    case 'D':
      distributed = atoi(optarg) != 0;
      break;
    case 'r':
      nrep = atoi(optarg);
      break;
    case 's':
      seed = atoi(optarg);
      break;
    default:
      fprintf(stderr,
              "usage: %s [-n ntasks] [-g grid_points] [-l lmax] [-d lp_diff]"
              " [-o 0|1|2] [-D 0|1] [-r nrep] [-s seed]\n",
              argv[0]);
      return 1;
    }
  }

  if (2 * lmax + lp_diff > 18) {
    fprintf(stderr, "2 * lmax + lp_diff must be <= 18\n");
    return 1;
  }

  // inverse coset mapping
  std::vector<orbital> coset_inv_host(1330);
  for (int lx = 0; lx <= 18; lx++)
    for (int ly = 0; ly <= 18 - lx; ly++)
      for (int lz = 0; lz <= 18 - lx - ly; lz++)
        coset_inv_host[coset(lx, ly, lz)] = {{lx, ly, lz}};
  GPU_CHECK(gpuMemcpyToSymbol(coset_inv, coset_inv_host.data(),
                              sizeof(orbital) * 1330));

  // grid and lattice. dh_[3 * i + j] is component j of lattice vector i.
  kernel_params params;
  const double h = 0.12;
  double dh[9] = {h, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, h};
  // Only an axis aligned cell is treated as orthogonal, as cell%orthorhombic
  // does on the fortran side. compute_cube_properties<..., true> converts the
  // cartesian vector (d, d, d) to lattice coordinates, which only gives a
  // valid cube when the lattice vectors are along x, y and z. NB:
  // grid_info::check_orthogonality (grid_gpu_context.h) also flags rotated
  // orthogonal cells as orthogonal, which sends them down that path.
  const bool orthogonal = (cell == 1);
  if (cell == 0) {
    dh[3] = 0.3 * h; // skew b along a
    dh[6] = 0.2 * h; // skew c along a
    dh[7] = 0.25 * h;
  } else if (cell == 2) {
    // orthogonal cell rotated by the euler angles (0.3, 0.5, 0.7)
    const double c1 = cos(0.3), s1 = sin(0.3), c2 = cos(0.5), s2 = sin(0.5),
                 c3 = cos(0.7), s3 = sin(0.7);
    const double R[9] = {c1 * c3 - c2 * s1 * s3,
                         -c1 * s3 - c2 * c3 * s1,
                         s1 * s2,
                         c3 * s1 + c1 * c2 * s3,
                         c1 * c2 * c3 - s1 * s3,
                         -c1 * s2,
                         s2 * s3,
                         c3 * s2,
                         c2};
    for (int i = 0; i < 9; i++)
      dh[i] = h * R[i];
  }
  memcpy(params.dh_, dh, sizeof(dh));
  invert3x3(params.dh_, params.dh_inv_);
  params.grid_full_size_ = make_int3(npts, npts, npts);
  params.grid_local_size_ = params.grid_full_size_;
  params.grid_lower_corner_ = make_int3(0, 0, 0);
  params.grid_border_width_ = make_int3(4, 4, 4);
  params.la_max_diff = lp_diff / 2;
  params.lb_max_diff = lp_diff - lp_diff / 2;
  params.ntasks = ntasks;

  double dh_max = 0.0;
  for (int i = 0; i < 3; i++)
    dh_max = std::max(dh_max, sqrt(dh[3 * i] * dh[3 * i] +
                                   dh[3 * i + 1] * dh[3 * i + 1] +
                                   dh[3 * i + 2] * dh[3 * i + 2]));

  const double box[3] = {npts * (dh[0] + dh[3] + dh[6]),
                         npts * (dh[1] + dh[4] + dh[7]),
                         npts * (dh[2] + dh[5] + dh[8])};

  // random tasks
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  std::vector<task_info> tasks(ntasks);
  size_t coef_size = 0;
  for (int i = 0; i < ntasks; i++) {
    task_info &t = tasks[i];
    memset(&t, 0, sizeof(task_info));
    t.zeta = 0.5 + 30.0 * unif(gen);
    t.zetb = 0.5 + 30.0 * unif(gen);
    t.zetp = t.zeta + t.zetb;
    t.la_max = (int)(unif(gen) * (lmax + 1)) % (lmax + 1);
    t.lb_max = (int)(unif(gen) * (lmax + 1)) % (lmax + 1);
    t.radius = sqrt(-log(1e-10) / t.zetp);
    for (int d = 0; d < 3; d++) {
      t.ra[d] = box[d] * unif(gen);
      t.rab[d] = 2.0 * unif(gen) - 1.0;
      t.rb[d] = t.ra[d] + t.rab[d];
      t.rp[d] = t.ra[d] + t.rab[d] * t.zetb / t.zetp;
    }
    t.border_mask =
        (distributed && unif(gen) < 0.5) ? (1 + (int)(unif(gen) * 63)) & 63 : 0;
    t.apply_border_mask = (t.border_mask != 0);
    t.skip_task = (2 * t.radius < dh_max);
    // grid_gpu_context.cu uses la_max + lb_max + 6 without a cap, which
    // reads past the end of the ncoset table for la_max + lb_max > 12.
    t.lp_max = std::min(t.la_max + t.lb_max + 6, 18);
    t.coef_offset = coef_size;
    coef_size += align_up_elems(ncoset(t.lp_max), 4);

    if (orthogonal && !t.apply_border_mask) {
      t.discrete_radius = compute_cube_properties<double, double3, true>(
          t.radius, params.dh_, params.dh_inv_, (double3 *)t.rp, &t.roffset,
          &t.cube_center, &t.lb_cube, &t.cube_size);
    } else {
      t.discrete_radius = compute_cube_properties<double, double3, false>(
          t.radius, params.dh_, params.dh_inv_, (double3 *)t.rp, &t.roffset,
          &t.cube_center, &t.lb_cube, &t.cube_size);
    }
  }

  const size_t grid_size = (size_t)npts * npts * npts;
  std::vector<double> grid(grid_size);
  for (auto &g : grid)
    g = 2.0 * unif(gen) - 1.0;

  // device buffers
  task_info *tasks_dev;
  double *grid_dev, *coef_dev;
  GPU_CHECK(gpuMalloc((void **)&tasks_dev, sizeof(task_info) * ntasks));
  GPU_CHECK(gpuMalloc((void **)&grid_dev, sizeof(double) * grid_size));
  GPU_CHECK(gpuMalloc((void **)&coef_dev, sizeof(double) * coef_size));
  GPU_CHECK(gpuMemcpy(tasks_dev, tasks.data(), sizeof(task_info) * ntasks,
                      gpuMemcpyHostToDevice));
  GPU_CHECK(gpuMemcpy(grid_dev, grid.data(), sizeof(double) * grid_size,
                      gpuMemcpyHostToDevice));
  GPU_CHECK(gpuMemset(coef_dev, 0, sizeof(double) * coef_size));

  params.tasks = tasks_dev;
  params.buffers_dev.grid = grid_dev;
  params.buffers_dev.coef = coef_dev;
  params.first_task = 0;

  printf("ntasks %d, grid %d^3, lmax %d, lp_diff %d, orthogonal %d, "
         "distributed %d\n",
         ntasks, npts, lmax, lp_diff, orthogonal, distributed);

  // reference
  std::vector<double> coef_gpu(coef_size), coef_ref(coef_size, 0.0);
  integrate_reference(params, tasks, coset_inv_host, grid, distributed,
                      orthogonal, coef_ref);

  gpuEvent_t start, stop;
  GPU_CHECK(gpuEventCreate(&start));
  GPU_CHECK(gpuEventCreate(&stop));
  const double tolerance = 1e-10;
  bool all_passed = true;

  auto run = [&](const char *name, auto &&launch) {
    // warm-up + correctness
    GPU_CHECK(gpuMemset(coef_dev, 0, sizeof(double) * coef_size));
    launch();
    GPU_CHECK(gpuGetLastError());
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpy(coef_gpu.data(), coef_dev, sizeof(double) * coef_size,
                        gpuMemcpyDeviceToHost));

    // timing
    GPU_CHECK(gpuEventRecord(start, 0));
    for (int rep = 0; rep < nrep; rep++)
      launch();
    GPU_CHECK(gpuEventRecord(stop, 0));
    GPU_CHECK(gpuEventSynchronize(stop));
    float elapsed_ms = 0.0;
    GPU_CHECK(gpuEventElapsedTime(&elapsed_ms, start, stop));

    double max_rel = 0.0;
    int worst_task = -1;
    for (int i = 0; i < ntasks; i++) {
      if (tasks[i].skip_task)
        continue;
      const int lp = tasks[i].la_max + params.la_max_diff + tasks[i].lb_max +
                     params.lb_max_diff;
      double norm = 0.0, diff = 0.0;
      for (int ic = 0; ic < ncoset(lp); ic++) {
        const size_t k = tasks[i].coef_offset + ic;
        norm = std::max(norm, fabs(coef_ref[k]));
        diff = std::max(diff, fabs(coef_ref[k] - coef_gpu[k]));
      }
      const double rel = diff / std::max(norm, 1e-300);
      if (rel > max_rel) {
        max_rel = rel;
        worst_task = i;
      }
    }
    const bool passed = max_rel < tolerance;
    all_passed = all_passed && passed;
    printf("%-28s %9.3f ms per launch (%d reps), max rel diff %.3e (task %d) "
           "-> %s\n",
           name, elapsed_ms / std::max(nrep, 1), nrep, max_rel, worst_task,
           passed ? "PASSED" : "FAILED");
  };

  run("integrate_kernel",
      [&]() { launch_integrate(params, ntasks, distributed, orthogonal); });

  GPU_CHECK(gpuEventDestroy(start));
  GPU_CHECK(gpuEventDestroy(stop));
  GPU_CHECK(gpuFree(tasks_dev));
  GPU_CHECK(gpuFree(grid_dev));
  GPU_CHECK(gpuFree(coef_dev));
  return all_passed ? 0 : 1;
}
