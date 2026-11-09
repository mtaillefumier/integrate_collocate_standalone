/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * Standalone driver for the original GPU kernels that transform between the
 * matrix blocks and the polynomial coefficients of the tasks, as of CP2K
 * revision c05efa7fa7:
 *
 *   - calculate_coefficients (collocate: pab -> cab -> cxyz)
 *   - compute_hab            (integrate: cxyz -> cab -> hab)
 *
 * The kernels are copied verbatim from src/grid/gpu/grid_gpu_collocate.cu and
 * src/grid/gpu/grid_gpu_integrate.cu. The CP2K headers they need
 * (grid_gpu_internal_header.h, grid_gpu_prepare_pab.h, grid_gpu_process_vab.h
 * and their own includes) are inlined verbatim at the top of this file, so
 * that it builds without the CP2K source tree.
 *
 * The driver generates nblocks matrix blocks, each with one set per atom and
 * all npgf x npgf primitive pairs as tasks, random sphi, pab and coefficients.
 * The GPU results are checked against a serial CPU port of the same
 * operations. Only GRID_FUNC_AB (density) and integrate without forces are
 * exercised.
 *
 * Build:
 *   nvcc -O3 -arch=sm_70 -D__OFFLOAD_CUDA -diag-suppress 177
 *        -o coefficients_original.x coefficients_original_standalone.cu
 *
 *   (-diag-suppress 177 silences the warnings about the unused static
 *   helpers of offload_runtime.h)
 *   hipcc -O3 --offload-arch=gfx90a -D__OFFLOAD_HIP -o coefficients_original.x
 *        coefficients_original_standalone.cu
 *
 * Usage:
 *   ./coefficients_original.x [-b nblocks] [-l lmax] [-p npgf] [-c nctr]
 *                             [-R 0|1] [-t 0|1] [-r nrep] [-s seed]
 *                             [-m 0|1|2]
 *
 *   -l: angular momentum of the sets (default 3), -R 1: random per block
 *   -p: primitives per set, -c: contractions per l (nsgf = nctr (l + 1)^2)
 *   -t 1: integrate tau (compute_hab<..., true>)
 *   -m 0: collocate, 1: integrate, 2: both (default)
 */

#include <algorithm>
#include <chrono>
#include <random>
#include <unistd.h>
#include <vector>


/* ===== begin of src/grid/gpu/grid_gpu_internal_header.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * Authors :
 - Dr Mathieu Taillefumier (ETH Zurich / CSCS)
 - Advanced Micro Devices, Inc.
*/

#ifndef GRID_GPU_INTERNAL_HEADER_H
#define GRID_GPU_INTERNAL_HEADER_H

#include <algorithm>
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRID_DEVICE __device__
extern "C" {

/* ===== begin of src/grid/common/grid_basis_set.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_BASIS_SET_H
#define GRID_BASIS_SET_H

/*******************************************************************************
 * \brief Internal representation of a basis set.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  int nset;
  int nsgf;
  int maxco;
  int maxpgf;
  int *lmin;
  int *lmax;
  int *npgf;
  int *nsgf_set;
  int *first_sgf;
  double *sphi;
  double *zet;
} grid_basis_set;

#ifndef __cplusplus

/*******************************************************************************
 * \brief Allocates a basis set which can be passed to grid_create_task_list.
 *
 * \param nset            Number of sets this basis is composed of.
 * \param nsgf            Size of contracted spherical basis, ie. the block size
 * \param maxco           Maximum number of Cartesian functions across all sets.
 * \param maxpgf          Maximum number of primitive Gaussians across all sets.
 *
 *      The following params are given for each set:
 *
 * \param lmin            Lowest angular momentum.
 * \param lmax            Highest angular momentum.
 * \param npgf            Number of primitive Gaussians, ie. exponents.
 * \param nsgf_set        Number of spherical basis functions
 * \param first_sgf       Index of first spherical basis function (one based).
 * \param sphi            Transformation matrix for (de-)contracting the basis.
 * \param zet             Exponents of primitive Gaussians.
 *
 * \param basis_set       Handle to the created basis set.
 *
 * \author Ole Schuett
 ******************************************************************************/
void grid_create_basis_set(const int nset, const int nsgf, const int maxco,
                           const int maxpgf, const int lmin[nset],
                           const int lmax[nset], const int npgf[nset],
                           const int nsgf_set[nset], const int first_sgf[nset],
                           const double sphi[nsgf][maxco],
                           const double zet[nset][maxpgf],
                           grid_basis_set **basis_set);

/*******************************************************************************
 * \brief Deallocates given basis set.
 * \author Ole Schuett
 ******************************************************************************/
void grid_free_basis_set(grid_basis_set *basis_set);

#endif

#endif

// EOF
/* ===== end of src/grid/common/grid_basis_set.h ===== */

/* ===== begin of src/grid/common/grid_constants.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_CONSTANTS_H
#define GRID_CONSTANTS_H

enum grid_func {
  GRID_FUNC_AB = 100,
  GRID_FUNC_DADB = 200,
  GRID_FUNC_ADBmDAB_X = 301,
  GRID_FUNC_ADBmDAB_Y = 302,
  GRID_FUNC_ADBmDAB_Z = 303,
  GRID_FUNC_ARDBmDARB_XX = 411,
  GRID_FUNC_ARDBmDARB_XY = 412,
  GRID_FUNC_ARDBmDARB_XZ = 413,
  GRID_FUNC_ARDBmDARB_YX = 421,
  GRID_FUNC_ARDBmDARB_YY = 422,
  GRID_FUNC_ARDBmDARB_YZ = 423,
  GRID_FUNC_ARDBmDARB_ZX = 431,
  GRID_FUNC_ARDBmDARB_ZY = 432,
  GRID_FUNC_ARDBmDARB_ZZ = 433,
  GRID_FUNC_DABpADB_X = 501,
  GRID_FUNC_DABpADB_Y = 502,
  GRID_FUNC_DABpADB_Z = 503,
  GRID_FUNC_DX = 601,
  GRID_FUNC_DY = 602,
  GRID_FUNC_DZ = 603,
  GRID_FUNC_DXDY = 701,
  GRID_FUNC_DYDZ = 702,
  GRID_FUNC_DZDX = 703,
  GRID_FUNC_DXDX = 801,
  GRID_FUNC_DYDY = 802,
  GRID_FUNC_DZDZ = 803,
  GRID_FUNC_DAB_X = 901,
  GRID_FUNC_DAB_Y = 902,
  GRID_FUNC_DAB_Z = 903,
  GRID_FUNC_ADB_X = 904,
  GRID_FUNC_ADB_Y = 905,
  GRID_FUNC_ADB_Z = 906,
  GRID_FUNC_CORE_X = 1001,
  GRID_FUNC_CORE_Y = 1002,
  GRID_FUNC_CORE_Z = 1003,
};

enum grid_backend {
  GRID_BACKEND_AUTO = 10,
  GRID_BACKEND_REF = 11,
  GRID_BACKEND_CPU = 12,
  GRID_BACKEND_DGEMM = 13,
  GRID_BACKEND_GPU = 14,
};

#endif

// EOF
/* ===== end of src/grid/common/grid_constants.h ===== */
}


/* ===== begin of src/grid/gpu/grid_gpu_context.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * Authors :
 - Dr Mathieu Taillefumier (ETH Zurich / CSCS)
 - Advanced Micro Devices, Inc.
*/

#ifndef GRID_GPU_CONTEXT_H
#define GRID_GPU_CONTEXT_H

#ifdef __OFFLOAD_HIP
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#endif
#include <array>
#include <climits>
#include <vector>

extern "C" {
}


/* ===== begin of src/offload/offload_library.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef OFFLOAD_LIBRARY_H
#define OFFLOAD_LIBRARY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
/*******************************************************************************
 * \brief Initalize runtime.
 * \author Rocco Meli
 ******************************************************************************/
void offload_init(void);

/*******************************************************************************
 * \brief Returns the number of available devices.
 * \author Ole Schuett
 ******************************************************************************/
int offload_get_device_count(void);

/*******************************************************************************
 * \brief Selects the chosen device to be used.
 * \author Ole Schuett
 ******************************************************************************/
void offload_set_chosen_device(int device_id);

/*******************************************************************************
 * \brief Returns the chosen device.
 * \author Ole Schuett
 ******************************************************************************/
int offload_get_chosen_device(void);

/*******************************************************************************
 * \brief Activates the device selected via offload_set_chosen_device()
 * \author Ole Schuett
 ******************************************************************************/
void offload_activate_chosen_device(void);

/*******************************************************************************
 * \brief Starts a timing range.
 * \author Ole Schuett
 ******************************************************************************/
void offload_timeset(const char *message);

/*******************************************************************************
 * \brief Ends a timing range.
 * \author Ole Schuett
 ******************************************************************************/
void offload_timestop(void);

/*******************************************************************************
 * \brief Gets free and total device memory.
 * \author Ole Schuett
 ******************************************************************************/
void offload_mem_info(size_t *free, size_t *total);

/*******************************************************************************
 * \brief Allocate pinned memory (or simple malloc when there is no gpu)
 ******************************************************************************/
int offload_host_malloc(void **ptr__, const size_t size__);

/*******************************************************************************
 * \brief free pinned memory (or simple free when there is no gpu)
 ******************************************************************************/
int offload_host_free(void *ptr__);

void offload_device_synchronize(void);

#ifdef __cplusplus
}
#endif

#endif

// EOF
/* ===== end of src/offload/offload_library.h ===== */

/* ===== begin of src/offload/offload_runtime.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef OFFLOAD_RUNTIME_H
#define OFFLOAD_RUNTIME_H

#if !defined(__LIBXSTREAM)
#undef __OFFLOAD_OPENCL
#elif !defined(__OFFLOAD_OPENCL)
#define __OFFLOAD_OPENCL
#endif
/* TODO: implement support or missing features */
#if defined(__OFFLOAD_OPENCL)
#if !defined(__NO_OFFLOAD_GRID)
#define __NO_OFFLOAD_GRID
#endif
#if !defined(__NO_OFFLOAD_PW)
#define __NO_OFFLOAD_PW
#endif
#endif

#if defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP) ||                       \
    defined(__OFFLOAD_OPENCL)
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#if !defined(__OFFLOAD)
#define __OFFLOAD
#endif

#if defined(__OFFLOAD_CUDA)
#include <cuda_runtime.h>
#elif defined(__OFFLOAD_HIP)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#elif defined(__OFFLOAD_OPENCL)
/* Types, macros, and functions provided by libxstream_cp2k.h. */
#include <libxstream/libxstream_cp2k.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__OFFLOAD_CUDA)
typedef cudaStream_t offloadStream_t;
typedef cudaEvent_t offloadEvent_t;
typedef cudaError_t offloadError_t;
#define offloadSuccess cudaSuccess
#elif defined(__OFFLOAD_HIP)
typedef hipStream_t offloadStream_t;
typedef hipEvent_t offloadEvent_t;
typedef hipError_t offloadError_t;
#define offloadSuccess hipSuccess
#endif

/*******************************************************************************
 * \brief Check given Cuda status and upon failure abort with a nice message.
 * \author Ole Schuett
 ******************************************************************************/
#if !defined(OFFLOAD_CHECK)
#define OFFLOAD_CHECK(CMD)                                                     \
  do {                                                                         \
    const offloadError_t error = (CMD);                                        \
    if (error != offloadSuccess) {                                             \
      const char *const name = offloadGetErrorName(error);                     \
      if (NULL != name && '\0' != *name) {                                     \
        fprintf(stderr, "ERROR: \"%s\" at %s:%i\n", name, __FILE__, __LINE__); \
      } else {                                                                 \
        fprintf(stderr, "ERROR %i: %s:%i\n", (int)error, __FILE__, __LINE__);  \
      }                                                                        \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#endif

#if !defined(__OFFLOAD_OPENCL)
/*******************************************************************************
 * \brief Wrapper around cudaGetErrorName.
 ******************************************************************************/
static inline const char *offloadGetErrorName(offloadError_t error) {
#if defined(__OFFLOAD_CUDA)
  return cudaGetErrorName(error);
#elif defined(__OFFLOAD_HIP)
  return hipGetErrorName(error);
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaGetLastError.
 ******************************************************************************/
static inline offloadError_t offloadGetLastError(void) {
#if defined(__OFFLOAD_CUDA)
  return cudaGetLastError();
#elif defined(__OFFLOAD_HIP)
  return hipGetLastError();
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemsetAsync.
 ******************************************************************************/
static inline void offloadMemsetAsync(void *ptr, const int val,
                                      const size_t size,
                                      offloadStream_t stream) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaMemsetAsync(ptr, val, size, stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipMemsetAsync(ptr, val, size, stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemset.
 ******************************************************************************/
static inline void offloadMemset(void *ptr, const int val, size_t size) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaMemset(ptr, val, size));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipMemset(ptr, val, size));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemcpyAsync(...,cudaMemcpyHostToDevice,...).
 ******************************************************************************/
static inline void offloadMemcpyAsyncHtoD(void *ptr_dev, const void *ptr_hst,
                                          const size_t size,
                                          offloadStream_t stream) {
#if defined(__OFFLOAD_UNIFIED_MEMORY)
  if (ptr_dev == ptr_hst) { /* fast-path only sensible without offsets */
    return;
  }
#endif
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(
      cudaMemcpyAsync(ptr_dev, ptr_hst, size, cudaMemcpyHostToDevice, stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(
      hipMemcpyAsync(ptr_dev, ptr_hst, size, hipMemcpyHostToDevice, stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemcpyAsync(...,cudaMemcpyDeviceToHost,...).
 ******************************************************************************/
static inline void offloadMemcpyAsyncDtoH(void *ptr_hst, const void *ptr_dev,
                                          const size_t size,
                                          const offloadStream_t stream) {
#if defined(__OFFLOAD_UNIFIED_MEMORY)
  if (ptr_hst == ptr_dev) { /* fast-path only sensible without offsets */
    return;
  }
#endif
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(
      cudaMemcpyAsync(ptr_hst, ptr_dev, size, cudaMemcpyDeviceToHost, stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(
      hipMemcpyAsync(ptr_hst, ptr_dev, size, hipMemcpyDeviceToHost, stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemcpyAsync(...,cudaMemcpyDeviceToDevice).
 ******************************************************************************/
static inline void offloadMemcpyAsyncDtoD(void *dst, const void *src,
                                          const size_t size,
                                          const offloadStream_t stream) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(
      hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemcpy(...,cudaMemcpyHostToDevice).
 ******************************************************************************/
static inline void offloadMemcpyHtoD(void *ptr_dev, const void *ptr_hst,
                                     const size_t size) {
#if defined(__OFFLOAD_UNIFIED_MEMORY)
  if (ptr_dev == ptr_hst) { /* fast-path only sensible without offsets */
    return;
  }
#endif
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaMemcpy(ptr_dev, ptr_hst, size, cudaMemcpyHostToDevice));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipMemcpy(ptr_dev, ptr_hst, size, hipMemcpyHostToDevice));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemcpy(...,cudaMemcpyDeviceToHost).
 ******************************************************************************/
static inline void offloadMemcpyDtoH(void *ptr_dev, const void *ptr_hst,
                                     const size_t size) {
#if defined(__OFFLOAD_UNIFIED_MEMORY)
  if (ptr_dev == ptr_hst) { /* fast-path only sensible without offsets */
    return;
  }
#endif
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaMemcpy(ptr_dev, ptr_hst, size, cudaMemcpyDeviceToHost));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipMemcpy(ptr_dev, ptr_hst, size, hipMemcpyDeviceToHost));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMemcpyToSymbol.
 ******************************************************************************/
static inline void offloadMemcpyToSymbol(const void *symbol, const void *src,
                                         const size_t count) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(
      cudaMemcpyToSymbol(symbol, src, count, 0, cudaMemcpyHostToDevice));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(
      hipMemcpyToSymbol(symbol, src, count, 0, hipMemcpyHostToDevice));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaEventCreate.
 ******************************************************************************/
static inline void offloadEventCreate(offloadEvent_t *event) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaEventCreate(event));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipEventCreate(event));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaEventDestroy.
 ******************************************************************************/
static inline void offloadEventDestroy(offloadEvent_t event) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaEventDestroy(event));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipEventDestroy(event));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaStreamCreate.
 ******************************************************************************/
static inline void offloadStreamCreate(offloadStream_t *stream) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaStreamCreate(stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipStreamCreate(stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaStreamDestroy.
 ******************************************************************************/
static inline void offloadStreamDestroy(offloadStream_t stream) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaStreamDestroy(stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipStreamDestroy(stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaEventSynchronize.
 ******************************************************************************/
static inline void offloadEventSynchronize(offloadEvent_t event) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaEventSynchronize(event));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipEventSynchronize(event));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaStreamSynchronize.
 ******************************************************************************/
static inline void offloadStreamSynchronize(offloadStream_t stream) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaStreamSynchronize(stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipStreamSynchronize(stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaEventRecord.
 ******************************************************************************/
static inline void offloadEventRecord(offloadEvent_t event,
                                      offloadStream_t stream) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaEventRecord(event, stream));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipEventRecord(event, stream));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMallocHost.
 ******************************************************************************/
static inline void offloadMallocHost(void **ptr, size_t size) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaMallocHost(ptr, size));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipHostMalloc(ptr, size, hipHostMallocDefault)); // inconsistent
#else
  assert(NULL != ptr);
  *ptr = malloc(size);
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaMalloc.
 ******************************************************************************/
static inline void offloadMalloc(void **ptr, size_t size) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaMalloc(ptr, size));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipMalloc(ptr, size));
#else
  assert(NULL != ptr);
  *ptr = NULL;
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaFree.
 ******************************************************************************/
static inline void offloadFree(void *ptr) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaFree(ptr));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipFree(ptr));
#else
  assert(NULL == ptr);
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaFreeHost.
 ******************************************************************************/
static inline void offloadFreeHost(void *ptr) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaFreeHost(ptr));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipHostFree(ptr)); // inconsistent
#else
  free(ptr);
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaStreamWaitEvent.
 ******************************************************************************/
static inline void offloadStreamWaitEvent(offloadStream_t stream,
                                          offloadEvent_t event) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaStreamWaitEvent(stream, event, 0 /*flags*/));
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipStreamWaitEvent(stream, event, 0 /*flags*/));
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaEventQuery.
 ******************************************************************************/
static inline bool offloadEventQuery(offloadEvent_t event) {
#if defined(__OFFLOAD_CUDA)
  return offloadSuccess == cudaEventQuery(event);
#elif defined(__OFFLOAD_HIP)
  return offloadSuccess == hipEventQuery(event);
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaDeviceSynchronize.
 ******************************************************************************/
static inline void offloadDeviceSynchronize(void) {
#if defined(__OFFLOAD_CUDA)
  OFFLOAD_CHECK(cudaDeviceSynchronize());
#elif defined(__OFFLOAD_HIP)
  OFFLOAD_CHECK(hipDeviceSynchronize());
#endif
}

/*******************************************************************************
 * \brief Wrapper around cudaDeviceSetLimit(cudaLimitMallocHeapSize,...).
 ******************************************************************************/
static inline void offloadEnsureMallocHeapSize(const size_t required_size) {
#if defined(__OFFLOAD_CUDA)
  size_t current_size;
  OFFLOAD_CHECK(cudaDeviceGetLimit(&current_size, cudaLimitMallocHeapSize));
  if (current_size < required_size) {
    OFFLOAD_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, required_size));
  }
#elif defined(__OFFLOAD_HIP) && (HIP_VERSION >= 50300000)
  size_t current_size;
  OFFLOAD_CHECK(hipDeviceGetLimit(&current_size, hipLimitMallocHeapSize));
  if (current_size < required_size) {
    OFFLOAD_CHECK(hipDeviceSetLimit(hipLimitMallocHeapSize, required_size));
  }
#else
  (void)required_size; /* mark used */
#endif
}

#endif /* !defined(__OFFLOAD_OPENCL) */

#ifdef __cplusplus
}
#endif

#endif // defined(__OFFLOAD_CUDA) || defined(__OFFLOAD_HIP) ||
       // defined(__OFFLOAD_OPENCL)
#endif
/* ===== end of src/offload/offload_runtime.h ===== */

namespace rocm_backend {
// a little helper class in the same spirit than std::vector. it must exist
// somewhere. Maybe possible to get the same thing with std::vector and
// specific allocator.

inline size_t round_up16(size_t n) { return (n + 15) & ~size_t(15); }

class smem_parameters;
template <typename T> class gpu_vector {
  size_t allocated_size_{0};
  size_t current_size_{0};
  bool internal_allocation_{false};
  T *device_ptr_{nullptr};
  T *host_ptr_{nullptr};

  void move_from(gpu_vector &&other) noexcept {
    allocated_size_ = other.allocated_size_;
    current_size_ = other.current_size_;
    internal_allocation_ = other.internal_allocation_;
    device_ptr_ = other.device_ptr_;
    host_ptr_ = other.host_ptr_;

    other.allocated_size_ = 0;
    other.current_size_ = 0;
    other.internal_allocation_ = false;
    other.device_ptr_ = nullptr;
    other.host_ptr_ = nullptr;
  }

public:
  gpu_vector() = default;

  gpu_vector(const gpu_vector &) = delete;
  gpu_vector &operator=(const gpu_vector &) = delete;

  gpu_vector(gpu_vector &&other) noexcept { move_from(std::move(other)); }

  gpu_vector &operator=(gpu_vector &&other) noexcept {
    if (this != &other) {
      reset();
      move_from(std::move(other));
    }
    return *this;
  }

  // size is the number of elements not the memory size
  explicit gpu_vector(const size_t size__) {
    allocated_size_ = (size__ < 16) ? 16 : round_up16(size__);
    current_size_ = size__;
    internal_allocation_ = true;

#ifndef __OFFLOAD_UNIFIED_MEMORY
    offloadMalloc((void **)&device_ptr_, sizeof(T) * allocated_size_);
#else
    hipMallocManaged((void **)&device_ptr_, sizeof(T) * allocated_size_);
#endif
    assert(device_ptr_ != nullptr);
  }

  gpu_vector(const size_t size__, void *ptr__) {
    allocated_size_ = size__;
    current_size_ = size__;
    internal_allocation_ = false;
    device_ptr_ = static_cast<T *>(ptr__);
  }
  ~gpu_vector() { reset(); }

  inline size_t size() const { return current_size_; }

  inline void copy_to_gpu(const T *data__) {
    assert(device_ptr_ != nullptr);
    assert(data__ != nullptr);
    offloadMemcpyHtoD(device_ptr_, data__, sizeof(T) * current_size_);
  }

  inline void copy_to_gpu(const T *data__, offloadStream_t &stream__) {
    assert(device_ptr_ != nullptr);
    assert(data__ != nullptr);
    offloadMemcpyAsyncHtoD(device_ptr_, data__, sizeof(T) * current_size_,
                           stream__);
  }

  inline void copy_associated_host_to_gpu(offloadStream_t &stream__) {
    assert(device_ptr_ != nullptr);
    assert(host_ptr_ != nullptr);
    // If the second assert fails it means that the object was created without
    // host buffer. It should not happen in the current scenario

    offloadMemcpyAsyncHtoD(device_ptr_, host_ptr_, sizeof(T) * current_size_,
                           stream__);
  }

  inline void copy_from_gpu(T *data__, offloadStream_t &stream__) {
    assert(device_ptr_ != nullptr);
    assert(data__ != nullptr);
    offloadMemcpyAsyncDtoH(data__, device_ptr_, sizeof(T) * current_size_,
                           stream__);
  }

  inline void copy_gpu_to_associated_host(offloadStream_t &stream__) {
    assert(device_ptr_ != nullptr);
    assert(host_ptr_ != nullptr);
    // If the second assert fails it means that the object was created without
    // host buffer. It should not happen in the current scenario

    offloadMemcpyAsyncDtoH(host_ptr_, device_ptr_, sizeof(T) * current_size_,
                           stream__);
  }

  inline void zero(offloadStream_t &stream__) {
    assert(device_ptr_ != nullptr);
    // zero device grid buffers
    offloadMemsetAsync(device_ptr_, 0, sizeof(T) * current_size_, stream__);
  }

  inline void associate(void *host_ptr__, void *device_ptr__,
                        const size_t size__) {
    assert(host_ptr__ != nullptr);
    assert(device_ptr__ != nullptr);
    reset();
    internal_allocation_ = false;
    allocated_size_ = size__;
    current_size_ = size__;
    device_ptr_ = static_cast<T *>(device_ptr__);
    host_ptr_ = static_cast<T *>(host_ptr__);
  }

  inline void zero() {
    // zero device grid buffers
    offloadMemset(device_ptr_, 0, sizeof(T) * current_size_);
  }

  inline void copy_to_gpu(const std::vector<T> &data__) {
    assert(data__.size() == current_size_);
    // if it fails it means that the vector on the gpu does not have the right
    // size. two option then
    // - resize the gpu vector
    // - or the cpu vector and gpu vector are not representing the quantity.
    assert(device_ptr_ != nullptr);
    offloadMemcpyHtoD(device_ptr_, data__.data(), sizeof(T) * data__.size());
  }

  inline void resize(const size_t new_size__) {
    if (!internal_allocation_) {
      allocated_size_ = 0;
      device_ptr_ = nullptr;
      host_ptr_ = nullptr;
    }

    assert(new_size__ != 0);
    if (allocated_size_ < new_size__) {
      if (internal_allocation_ && device_ptr_ != nullptr)
        offloadFree(device_ptr_);
      device_ptr_ = nullptr;
      allocated_size_ = (new_size__ < 16) ? 16 : round_up16(new_size__);
      offloadMalloc((void **)&device_ptr_, sizeof(T) * allocated_size_);
      internal_allocation_ = true;
      host_ptr_ = nullptr;
    }
    assert(device_ptr_ != nullptr);
    current_size_ = new_size__;
  }

  // does not invalidate the pointer. The memory is still allocated
  inline void clear() { current_size_ = 0; }

  // reset the class and free memory
  inline void reset() {
    if (internal_allocation_ && (device_ptr_ != nullptr)) {
      offloadFree(device_ptr_);
    }

    allocated_size_ = 0;
    current_size_ = 0;
    device_ptr_ = nullptr;
    host_ptr_ = nullptr;
    internal_allocation_ = false;
  }

  inline T *data() { return device_ptr_; }
  inline const T *data() const { return device_ptr_; }
};

template <typename T> class grid_info {
private:
  int3 full_size_;
  int3 local_size_;
  // origin of the local part of the grid in grid point
  int3 lower_corner_;
  int3 border_width_;
  std::array<T, 9> dh_;
  std::array<T, 9> dh_inv_;
  bool orthogonal_{false};
  // lattice vectors along x, y and z
  bool axis_aligned_{false};
  bool is_distributed_{false};
  // largest cube size (in grid points, all directions) of the tasks of this
  // level. Used to size the shared memory of the separable kernels.
  int max_cube_size_{0};
  gpu_vector<T> grid_;
  // expansion of the cartesian monomials in lattice monomials, see
  // build_lattice_to_cartesian_matrix in grid_gpu_separable.h. Valid up to
  // lattice_to_cartesian_lmax_ (-1 when not computed for the current dh).
  gpu_vector<T> lattice_to_cartesian_;
  int lattice_to_cartesian_lmax_{-1};

public:
  grid_info(const grid_info &) = delete;
  grid_info &operator=(const grid_info &) = delete;

  grid_info(grid_info &&) noexcept = default;
  grid_info &operator=(grid_info &&) noexcept = default;

  grid_info() {};

  grid_info(const int *full_size__, const int *local_size__,
            const int *border_width__) {
    int roffset__[3] = {0, 0, 0};
    initialize(full_size__, local_size__, roffset__, border_width__);
  }

  ~grid_info() = default;

  inline void copy_to_gpu(const T *data, offloadStream_t &stream) {
    assert(data != nullptr);
    grid_.copy_to_gpu(data, stream);
  }

  inline void copy_to_gpu(offloadStream_t &stream) {
    grid_.copy_associated_host_to_gpu(stream);
  }

  inline void reset() { grid_.reset(); }

  /*
   * We do not allocate memory as the buffer is always coming from the outside
   * world. We only initialize the sizes, etc...
   */
  inline void resize(const int *full_size__, const int *local_size__,
                     const int *const roffset__,
                     const int *const border_width__) {
    initialize(full_size__, local_size__, roffset__, border_width__);
  }

  inline size_t size() const { return grid_.size(); }

  inline void zero(offloadStream_t &stream) { grid_.zero(stream); }

  inline void set_lattice_vectors(const T *dh__, const T *dh_inv__) {
    for (int i = 0; i < 9; ++i) {
      dh_[i] = dh__[i];
      dh_inv_[i] = dh_inv__[i];
    }
    axis_aligned_ = (dh_[1] == 0.0) && (dh_[2] == 0.0) && (dh_[3] == 0.0) &&
                    (dh_[5] == 0.0) && (dh_[6] == 0.0) && (dh_[7] == 0.0);
    lattice_to_cartesian_lmax_ = -1;
  }

  inline bool is_axis_aligned() const { return axis_aligned_; }

  inline int max_cube_size() const { return max_cube_size_; }
  inline void max_cube_size(const int size__) { max_cube_size_ = size__; }

  inline gpu_vector<T> &lattice_to_cartesian() { return lattice_to_cartesian_; }
  inline int lattice_to_cartesian_lmax() const {
    return lattice_to_cartesian_lmax_;
  }
  inline void lattice_to_cartesian_lmax(const int lmax__) {
    lattice_to_cartesian_lmax_ = lmax__;
  }

  inline void is_distributed(const bool distributed__) {
    is_distributed_ = distributed__;
  }

  /// Check if the lattice vectors form a orthogonal basis
  // The orthogonal code paths (cube setup, x / y trimming, separable kernels)
  // assume lattice vectors along x, y and z. A rotated orthogonal cell is
  // therefore handled as non orthogonal. ortho (cell%orthorhombic on the
  // fortran side) is only true for a diagonal hmat, up to 1e-14.
  void check_orthogonality(const bool ortho) {
    if (ortho)
      axis_aligned_ = true;
    orthogonal_ = axis_aligned_;
  }

  inline void copy_to_host(T *data__, offloadStream_t &stream) {
    assert(data__ != nullptr);
    grid_.copy_from_gpu(data__, stream);
  }

  inline void copy_to_host(offloadStream_t &stream) {
    grid_.copy_gpu_to_associated_host(stream);
  }

  inline void associate(void *host_ptr__, void *device_ptr__,
                        const size_t size__) {
    assert(host_ptr__ != nullptr);
    assert(device_ptr__ != nullptr);
    grid_.associate(host_ptr__, device_ptr__, size__);
  }
  inline bool is_distributed() { return is_distributed_; }

  inline int3 &full_size() { return full_size_; }

  inline int3 &local_size() { return local_size_; }

  inline int3 &lower_corner() { return lower_corner_; }

  inline int3 &border_width() { return border_width_; }

  inline T *data() { return grid_.data(); }
  inline const T *data() const { return grid_.data(); }

  inline gpu_vector<T> &grid() { return grid_; }
  inline const gpu_vector<T> &grid() const { return grid_; }

  inline T *dh() { return dh_.data(); }
  inline const T *dh() const { return dh_.data(); }

  inline T *dh_inv() { return dh_inv_.data(); }
  inline const T *dh_inv() const { return dh_inv_.data(); }

  inline bool is_orthogonal() const { return orthogonal_; }
  inline bool is_distributed() const { return is_distributed_; }

private:
  void initialize(const int *const full_size__, const int *const local_size__,
                  const int *const roffset__, const int *const border_width__) {
    // the calling code store things like this cube[z][y][x] (in fortran
    // cube(x,y,z)).

    full_size_.x = full_size__[0];
    full_size_.y = full_size__[1];
    full_size_.z = full_size__[2];

    local_size_.x = local_size__[0];
    local_size_.y = local_size__[1];
    local_size_.z = local_size__[2];

    lower_corner_.x = roffset__[0];
    lower_corner_.y = roffset__[1];
    lower_corner_.z = roffset__[2];

    is_distributed_ = (full_size_.x != local_size_.x) ||
                      (full_size_.y != local_size_.y) ||
                      (full_size_.z != local_size_.z);

    border_width_.x = border_width__[0];
    border_width_.y = border_width__[1];
    border_width_.z = border_width__[2];
  }
};

/*******************************************************************************
 * \brief Internal representation of a task.
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

/*******************************************************************************
 * \brief Device buffers used by the collocate and integrate kernels.
 *        Which fields are non-null depends on which kernel is launched.
 ******************************************************************************/
struct kernel_buffers {
  double *pab_block{nullptr};
  double *grid{nullptr};
  double *coef{nullptr};
  double *hab_block{nullptr};
  double *forces{nullptr};
  double *virial{nullptr};
  double *cab{nullptr};
};

/*******************************************************************************
 * \brief Parameters of the collocate kernel.
 ******************************************************************************/

struct kernel_params {
  int first_task{0};
  // max size of cab.
  int cab_size_{0};
  int3 grid_full_size_ = make_int3(0, 0, 0);
  int3 grid_local_size_ = make_int3(0, 0, 0);
  int3 grid_lower_corner_ = make_int3(0, 0, 0);
  int3 grid_border_width_ = make_int3(0, 0, 0);
  double dh_[9];
  double dh_inv_[9];
  task_info *tasks;
  int *block_offsets{nullptr};
  char la_min_diff{0};
  char lb_min_diff{0};
  char la_max_diff{0};
  char lb_max_diff{0};
  enum grid_func func;
  kernel_buffers buffers_dev;
  double **sphi_dev{nullptr};
  int ntasks{0};
  int *task_sorted_by_blocks_dev{nullptr};
  int *sorted_blocks_offset_dev{nullptr};
  int *num_tasks_per_block_dev{nullptr};
  int *cab_block_offset_dev{nullptr};
  // integrate: tasks with lp >= separable_lp_min are handled by
  // integrate_separable_kernel, the others by integrate_kernel.
  int separable_lp_min{INT_MAX};
};

/* regroup all information about the context. */
class context_info {
private:
  int device_id_{-1};
  int lmax_{0};
  // largest number of spherical functions of a set over all basis sets. Sizes
  // the shared memory of calculate_coefficients / compute_hab.
  int max_nsgf_set_{0};
  unsigned int checksum_{0};

public:
  int ntasks{0};
  int nlevels{0};
  int natoms{0};
  int nkinds{0};
  int nblocks{0};
  int cab_size_per_block_{0};
  std::vector<double *> sphi;
  std::vector<offloadStream_t> level_streams;
  offloadStream_t main_stream;
  int stats[2][20]; // [has_border_mask][lp]
  // all these tables are on the gpu. we can resize them copy to them and copy
  // from them
  gpu_vector<int> block_offsets_dev;
  gpu_vector<int> cab_block_offset_dev;
  gpu_vector<double> coef_dev_;
  gpu_vector<double> cab_dev_;
  gpu_vector<double> pab_block_;
  gpu_vector<double> hab_block_;
  gpu_vector<double> forces_;
  gpu_vector<double> virial_;
  gpu_vector<task_info> tasks_dev;
  gpu_vector<int> num_tasks_per_block_dev_;
  std::vector<grid_info<double>> grid_;
  std::vector<int> number_of_tasks_per_level_;
  std::vector<int> first_task_per_level_;
  std::vector<int> sphi_size;
  gpu_vector<double *> sphi_dev;
  gpu_vector<int> task_sorted_by_blocks_dev, sorted_blocks_offset_dev;
  bool calculate_forces{false};
  bool calculate_virial{false};
  bool compute_tau{false};
  bool apply_border_mask{false};
  // Default constructor
  context_info() = default;

  explicit context_info(int device_id__) {
    device_id_ = (device_id__ < 0) ? 0 : device_id__;
  }

  ~context_info() { clear(); }

  // Non-copyable
  context_info(const context_info &) = delete;
  context_info &operator=(const context_info &) = delete;

  // Movable
  context_info(context_info &&other) noexcept
      : device_id_{other.device_id_}, lmax_{other.lmax_},
        max_nsgf_set_{other.max_nsgf_set_}, checksum_{other.checksum_},
        ntasks{other.ntasks}, nlevels{other.nlevels}, natoms{other.natoms},
        nkinds{other.nkinds}, nblocks{other.nblocks},
        cab_size_per_block_{other.cab_size_per_block_},
        sphi{std::move(other.sphi)},
        level_streams{std::move(other.level_streams)},
        main_stream{other.main_stream},
        block_offsets_dev{std::move(other.block_offsets_dev)},
        cab_block_offset_dev{std::move(other.cab_block_offset_dev)},
        coef_dev_{std::move(other.coef_dev_)},
        cab_dev_{std::move(other.cab_dev_)},
        pab_block_{std::move(other.pab_block_)},
        hab_block_{std::move(other.hab_block_)},
        forces_{std::move(other.forces_)}, virial_{std::move(other.virial_)},
        tasks_dev{std::move(other.tasks_dev)},
        num_tasks_per_block_dev_{std::move(other.num_tasks_per_block_dev_)},
        grid_{std::move(other.grid_)},
        number_of_tasks_per_level_{std::move(other.number_of_tasks_per_level_)},
        first_task_per_level_{std::move(other.first_task_per_level_)},
        sphi_size{std::move(other.sphi_size)},
        sphi_dev{std::move(other.sphi_dev)},
        task_sorted_by_blocks_dev{std::move(other.task_sorted_by_blocks_dev)},
        sorted_blocks_offset_dev{std::move(other.sorted_blocks_offset_dev)},
        calculate_forces{other.calculate_forces},
        calculate_virial{other.calculate_virial},
        compute_tau{other.compute_tau},
        apply_border_mask{other.apply_border_mask} {

    // copy stats
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 20; ++j)
        stats[i][j] = other.stats[i][j];

    // leave other in a safe state (no ownership)
    other.device_id_ = -1;
    other.lmax_ = 0;
    other.max_nsgf_set_ = 0;
    other.checksum_ = 0;
    other.ntasks = 0;
    other.nlevels = 0;
    other.natoms = 0;
    other.nkinds = 0;
    other.nblocks = 0;
    other.main_stream = {};
    other.calculate_forces = false;
    other.calculate_virial = false;
    other.compute_tau = false;
    other.apply_border_mask = false;
  }

  context_info &operator=(context_info &&other) noexcept {
    if (this != &other) {
      clear(); // free current GPU resources and streams

      device_id_ = other.device_id_;
      lmax_ = other.lmax_;
      max_nsgf_set_ = other.max_nsgf_set_;
      checksum_ = other.checksum_;
      ntasks = other.ntasks;
      nlevels = other.nlevels;
      natoms = other.natoms;
      nkinds = other.nkinds;
      nblocks = other.nblocks;
      sphi = std::move(other.sphi);
      level_streams = std::move(other.level_streams);
      main_stream = other.main_stream;

      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 20; ++j)
          stats[i][j] = other.stats[i][j];

      block_offsets_dev = std::move(other.block_offsets_dev);
      cab_block_offset_dev = std::move(other.cab_block_offset_dev);
      coef_dev_ = std::move(other.coef_dev_);
      cab_dev_ = std::move(other.cab_dev_);
      pab_block_ = std::move(other.pab_block_);
      hab_block_ = std::move(other.hab_block_);
      forces_ = std::move(other.forces_);
      virial_ = std::move(other.virial_);
      tasks_dev = std::move(other.tasks_dev);
      num_tasks_per_block_dev_ = std::move(other.num_tasks_per_block_dev_);
      grid_ = std::move(other.grid_);
      number_of_tasks_per_level_ = std::move(other.number_of_tasks_per_level_);
      first_task_per_level_ = std::move(other.first_task_per_level_);
      sphi_size = std::move(other.sphi_size);
      sphi_dev = std::move(other.sphi_dev);
      task_sorted_by_blocks_dev = std::move(other.task_sorted_by_blocks_dev);
      sorted_blocks_offset_dev = std::move(other.sorted_blocks_offset_dev);

      calculate_forces = other.calculate_forces;
      calculate_virial = other.calculate_virial;
      compute_tau = other.compute_tau;
      apply_border_mask = other.apply_border_mask;

      // reset other
      other.device_id_ = -1;
      other.lmax_ = 0;
      other.max_nsgf_set_ = 0;
      other.checksum_ = 0;
      other.ntasks = 0;
      other.nlevels = 0;
      other.natoms = 0;
      other.nkinds = 0;
      other.nblocks = 0;
      other.main_stream = {};
      other.calculate_forces = false;
      other.calculate_virial = false;
      other.compute_tau = false;
      other.apply_border_mask = false;
    }
    return *this;
  }

  void clear() {
    if (device_id_ >= 0) {
      offload_set_chosen_device(device_id_);
      offload_activate_chosen_device();
    }

    tasks_dev.reset();
    block_offsets_dev.reset();
    cab_block_offset_dev.reset();
    coef_dev_.reset();
    cab_dev_.reset();
    task_sorted_by_blocks_dev.reset();
    sorted_blocks_offset_dev.reset();
    sphi_dev.reset();
    forces_.reset();
    virial_.reset();

    for (auto &phi : sphi) {
      if (phi != nullptr)
        offloadFree(phi);
    }
    sphi.clear();

    if (main_stream) {
      offloadStreamDestroy(main_stream);
      main_stream = {};
    }

    for (auto &stream : level_streams) {
      if (stream)
        offloadStreamDestroy(stream);
    }
    level_streams.clear();

    for (auto &g : grid_) {
      g.reset();
    }
    grid_.clear();
  }

  int lmax() const { return lmax_; }
  int max_nsgf_set() const { return max_nsgf_set_; }

  void initialize_basis_sets(const grid_basis_set **basis_sets,
                             const int nkinds__) {
    nkinds = nkinds__;
    if (nkinds__ > (int)sphi.size()) {
      for (auto &phi : sphi)
        if (phi != nullptr) {
          offloadFree(phi);
        }

      sphi_dev.resize(nkinds__);

      sphi.resize(nkinds__, nullptr);
      sphi_size.clear();
      sphi_size.resize(nkinds__, 0);
      sphi_dev.resize(nkinds__);
    }

    // Upload basis sets to device.
    for (int i = 0; i < nkinds__; i++) {
      const auto &basis_set = basis_sets[i];
      if (sphi_size[i] < basis_set->nsgf * basis_set->maxco) {
        offloadMalloc((void **)&sphi[i],
                      basis_set->nsgf * basis_set->maxco * sizeof(double));
        sphi_size[i] = basis_set->nsgf * basis_set->maxco;
      }
      //      offloadMemset(sphi[i], 0, sizeof(double) * sphi_size[i]);
      offloadMemcpyHtoD(sphi[i], basis_set->sphi,
                        basis_set->nsgf * basis_set->maxco * sizeof(double));
    }
    sphi_dev.copy_to_gpu(sphi);
    // Find largest angular momentum and set size.
    lmax_ = 0;
    max_nsgf_set_ = 0;
    for (int ikind = 0; ikind < nkinds; ikind++) {
      for (int iset = 0; iset < basis_sets[ikind]->nset; iset++) {
        lmax_ = std::max(lmax_, basis_sets[ikind]->lmax[iset]);
        max_nsgf_set_ =
            std::max(max_nsgf_set_, basis_sets[ikind]->nsgf_set[iset]);
      }
    }
  }

  void create_streams() {
    // allocate main hip stream
    offloadStreamCreate(&main_stream);

    // allocate one hip stream per grid level
    if ((int)level_streams.size() < nlevels) {
      level_streams.resize(nlevels);
      for (auto &stream : level_streams) {
        offloadStreamCreate(&stream);
      }
    }
  }

  void synchronize(offloadStream_t &stream) {
    offloadStreamSynchronize(stream);
  }

  void synchornize() {
    // wait for all the streams to finish
    offloadDeviceSynchronize();
  }

  void set_device() {
    offload_set_chosen_device(device_id_);
    offload_activate_chosen_device();
  }

  void collocate_one_grid_level(const int level, const enum grid_func func,
                                int *lp_diff);
  void integrate_one_grid_level(const int level, int *lp_diff);
  void compute_hab_coefficients();
  /* basic checksum computation for simple verification that the object is sane
   */
  void compute_checksum() { checksum_ = compute_checksum_(); }
  void verify_checksum() {
    if (checksum_ != compute_checksum_()) {
      fprintf(stderr, "This object does not seem to have the right structure.\n"
                      "A casting went wrong or the object is corrupted\n");
      abort();
    }
  }
  void calculate_all_coefficients(const enum grid_func func, int *lp_diff);

private:
  kernel_params set_kernel_parameters(const int level,
                                      const smem_parameters &smem_params);
  unsigned int compute_checksum_() {
    return natoms ^ ntasks ^ nlevels ^ nkinds ^ nblocks ^ 0x4F2C5D1A;
  }
};
} // namespace rocm_backend
#endif
/* ===== end of src/grid/gpu/grid_gpu_context.h ===== */

namespace rocm_backend {

#if defined(__OFFLOAD_CUDA) || defined(__HIP_PLATFORM_NVIDIA__)
#if __CUDA_ARCH__ < 600
__device__ __inline__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif
#endif

/*******************************************************************************
 * \brief Orbital angular momentum.
 ******************************************************************************/
struct orbital {
  int l[3];
};

__constant__ orbital coset_inv[1330];
__constant__ int binomial_coef[19][19];

/*******************************************************************************
 * \brief Differences in angular momentum.
 ******************************************************************************/
struct ldiffs_value {
  int la_max_diff{0};
  int la_min_diff{0};
  int lb_max_diff{0};
  int lb_min_diff{0};
};

/*******************************************************************************
 * \brief data needed for calculating the coefficients, forces, and stress
 ******************************************************************************/
template <typename T> struct smem_task {
  bool block_transposed;
  T ra[3];
  T rb[3];
  T rp[3];
  T rab[3];
  T zeta;
  T zetb;
  T zetp;
  T prefactor;
  T off_diag_twice;
  char la_max;
  char lb_max;
  char la_min;
  char lb_min;
  char lp;
  // size of the cab matrix
  unsigned short int n1;
  unsigned short int n2;
  // size of entire spherical basis
  unsigned short int nsgfa;
  unsigned short int nsgfb;
  // size of spherical set
  unsigned short int nsgf_seta;
  unsigned short int nsgf_setb;
  // start of decontracted set, ie. pab and hab
  int first_coseta;
  int first_cosetb;
  // size of decontracted set, ie. pab and hab
  int ncoseta;
  int ncosetb;
  // strides of the sphi transformation matrices
  int maxcoa;
  int maxcob;
  // pointers matrices
  T *pab_block;
  T *sphia;
  T *sphib;
  // integrate
  T *hab_block;
  T *forces_a;
  T *forces_b;
};

/*******************************************************************************
 * \brief data needed for collocate and integrate kernels
 ******************************************************************************/
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

/*******************************************************************************
 * \brief Factorial function, e.g. fac(5) = 5! = 120.
 * \author Ole Schuett
 ******************************************************************************/
__device__ __inline__ double fac(const int i) {
  static const double table[] = {
      0.10000000000000000000E+01, 0.10000000000000000000E+01,
      0.20000000000000000000E+01, 0.60000000000000000000E+01,
      0.24000000000000000000E+02, 0.12000000000000000000E+03,
      0.72000000000000000000E+03, 0.50400000000000000000E+04,
      0.40320000000000000000E+05, 0.36288000000000000000E+06,
      0.36288000000000000000E+07, 0.39916800000000000000E+08,
      0.47900160000000000000E+09, 0.62270208000000000000E+10,
      0.87178291200000000000E+11, 0.13076743680000000000E+13,
      0.20922789888000000000E+14, 0.35568742809600000000E+15,
      0.64023737057280000000E+16, 0.12164510040883200000E+18,
      0.24329020081766400000E+19, 0.51090942171709440000E+20,
      0.11240007277776076800E+22, 0.25852016738884976640E+23,
      0.62044840173323943936E+24, 0.15511210043330985984E+26,
      0.40329146112660563558E+27, 0.10888869450418352161E+29,
      0.30488834461171386050E+30, 0.88417619937397019545E+31,
      0.26525285981219105864E+33};
  return table[i];
}

/*******************************************************************************
 * \brief Number of Cartesian orbitals up to given angular momentum quantum.
 * \author Ole Schuett
 ******************************************************************************/
__host__ __device__ __inline__ int ncoset(const int l) {
  static const int table[] = {1,  // l=0
                              4,  // l=1
                              10, // l=2 ...
                              20,  35,  56,  84,  120, 165, 220,  286,
                              364, 455, 560, 680, 816, 969, 1140, 1330};
  return table[l];
}

/*******************************************************************************
 * \brief Maps three angular momentum components to a single zero based index.
 ******************************************************************************/
__host__ __device__ __inline__ int coset(int lx, int ly, int lz) {
  const int l = lx + ly + lz;
  if (l == 0) {
    return 0;
  } else {
    return ncoset(l - 1) + ((l - lx) * (l - lx + 1)) / 2 + lz;
  }
}

/*******************************************************************************
 * \brief Increase i'th component of given orbital angular momentum.
 ******************************************************************************/
__device__ __inline__ orbital up(const int i, const orbital &a) {
  orbital b = a;
  b.l[i] += 1;
  return b;
}

/*******************************************************************************
 * \brief Decrease i'th component of given orbital angular momentum.
 ******************************************************************************/
__inline__ __device__ orbital down(const int i, const orbital &a) {
  orbital b = a;
  b.l[i] = max(0, a.l[i] - 1);
  return b;
}

/*******************************************************************************
 * \brief Return coset index of given orbital angular momentum.
 ******************************************************************************/
__inline__ __device__ int idx(const orbital a) {
  return coset(a.l[0], a.l[1], a.l[2]);
}

__device__ __inline__ double power(const double x, const int expo) {
  double tmp = 1.0;
  for (int i = 1; i <= expo; i++)
    tmp *= x;
  return tmp;
}

/*******************************************************************************
 * \brief Adds given value to matrix element cab[idx(b)][idx(a)].
 ******************************************************************************/
template <typename T = double>
__device__ __inline__ void prep_term(const orbital a, const orbital b,
                                     const T value, const int n, T *cab) {
  atomicAdd(&cab[idx(b) * n + idx(a)], value);
}

/*******************************************************************************
 * \brief Initializes the device's constant memory.
 * \author Ole Schuett
 ******************************************************************************/
inline static void init_constant_memory() {
  static bool initialized = false;
  if (initialized) {
    return; // constant memory has to be initialized only once
  }

  // Inverse coset mapping
  orbital coset_inv_host[1330];
  for (int lx = 0; lx <= 18; lx++) {
    for (int ly = 0; ly <= 18 - lx; ly++) {
      for (int lz = 0; lz <= 18 - lx - ly; lz++) {
        const int i = coset(lx, ly, lz);
        coset_inv_host[i] = {{lx, ly, lz}};
      }
    }
  }

  offloadMemcpyToSymbol(coset_inv, &coset_inv_host, sizeof(coset_inv_host));

  // Binomial coefficient
  int binomial_coef_host[19][19] = {
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 9, 36, 84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0},
      {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 0, 0, 0, 0, 0, 0, 0},
      {1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 0, 0, 0, 0, 0,
       0},
      {1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1, 0, 0,
       0, 0, 0},
      {1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1,
       0, 0, 0, 0},
      {1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455,
       105, 15, 1, 0, 0, 0},
      {1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820,
       560, 120, 16, 1, 0, 0},
      {1, 17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376,
       6188, 2380, 680, 136, 17, 1, 0},
      {1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, 31824,
       18564, 8568, 3060, 816, 153, 18, 1}};
  offloadMemcpyToSymbol(binomial_coef, &binomial_coef_host[0][0],
                        sizeof(binomial_coef_host));
  initialized = true;
}

// calculate the global index of a thread block
__inline__ __device__ unsigned int block_index() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

// Calculating the global index in the grid of any given device thread
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

__inline__ __device__ float3 compute_coordinates(const float *__restrict__ dh_,
                                                 const float x, const float y,
                                                 const float z) {

  float3 r3;
  // I make no distinction between orthorhombic and non orthorhombic
  // cases

  r3.x = z * dh_[6] + y * dh_[3] + x * dh_[0];
  r3.y = z * dh_[7] + y * dh_[4] + x * dh_[1];
  r3.z = z * dh_[8] + y * dh_[5] + x * dh_[2];
  return r3;
}

/*******************************************************************************
 * \brief Computes the polynomial expansion coefficients:
 *        (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
 ******************************************************************************/
template <typename T>
__inline__ __device__ void compute_alpha(const smem_task<T> &task,
                                         T *__restrict__ alpha) {
  // strides for accessing alpha
  const int s3 = (task.lp + 1);
  const int s2 = (task.la_max + 1) * s3;
  const int s1 = (task.lb_max + 1) * s2;
  const int tid = thread_global_index();

  for (int i = tid; i < 3 * s1; i += blockDim.x * blockDim.y * blockDim.z)
    alpha[i] = 0.0;

  __syncthreads();

  for (int idir = threadIdx.z; idir < 3; idir += blockDim.z) {
    const T drpa = task.rp[idir] - task.ra[idir];
    const T drpb = task.rp[idir] - task.rb[idir];
    for (int la = threadIdx.y; la <= task.la_max; la += blockDim.y) {
      for (int lb = threadIdx.x; lb <= task.lb_max; lb += blockDim.x) {
        T a = 1.0;
        for (int k = 0; k <= la; k++) {
          T b = 1.0;
          const int base = idir * s1 + lb * s2 + la * s3;
          for (int l = 0; l <= lb; l++) {
            alpha[base + la - l + lb - k] +=
                a * b * binomial_coef[la][k] * binomial_coef[lb][l];
            b *= drpb;
          }
          a *= drpa;
        }
      }
    }
  }
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

__host__ __inline__ __device__ void
convert_to_lattice_coordinates(const float *dh_inv_,
                               const float3 *__restrict__ const rp,
                               float3 *__restrict__ rp_c) {
  rp_c->x = dh_inv_[0] * rp->x + dh_inv_[3] * rp->y + dh_inv_[6] * rp->z;
  rp_c->y = dh_inv_[1] * rp->x + dh_inv_[4] * rp->y + dh_inv_[7] * rp->z;
  rp_c->z = dh_inv_[2] * rp->x + dh_inv_[5] * rp->y + dh_inv_[8] * rp->z;
}

__host__ __inline__ __device__ void
convert_from_lattice_coordinates_to_cartesian(
    const float *__restrict__ dh_, const float3 *__restrict__ const rp,
    float3 *__restrict__ rp_c) {
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

    // the cube is actually slightly bigger than the sphere of radius r. that's
    // why we need to discretize it to get the cube size "right".

    // disc_radius >= radius always. somehow despite the fact that we compile
    // things with a c++ compiler on the host side, we need to use fmin instead
    // of std::min since std::min is not allowed on the device side

    // We assume no specific form for the orthogonal matrix. (can be diaognal or
    // completely full, the only constraint is that the tree vectors are
    // orthogonal)

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

    /* This potentially makes the cube bigger than necessary because disc_radius
       is derived with the smallest lattice parameter. It is not an issue when
       the lattice is cubic but could lead to more calculations than necessary
       when the lattice is orthorhombic */
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
    /* the interval is not symmetrical for some curious reasons. it should go
     * from [-L..L+1] so the number of points is multiple of two */
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

__inline__ __device__ void compute_window_size(const int3 grid_size,
                                               const int border_mask,
                                               const int3 &border_width,
                                               int3 &window_size,
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

/*******************************************************************************
 * \brief Wraps a cube-local coordinate into the periodic full grid along one
 *        axis. Single point of entry shared by collocate_kernel and
 *        integrate_kernel.
 ******************************************************************************/
__inline__ __device__ int wrap_grid_index(const int idx, const int full_size) {
  int idx2 = idx % full_size;
  if (idx2 < 0)
    idx2 += full_size;
  return idx2;
}

/*******************************************************************************
 * \brief Shifts task.cube_center into the local grid's coordinate system and,
 *        for distributed grids, derives the border window. Single point of
 *        entry shared by collocate_kernel and integrate_kernel so the two
 *        cannot drift apart on which axis maps to which.
 ******************************************************************************/
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

/*******************************************************************************
 * \brief Transforms coefficients C_ab into C_xyz.
 ******************************************************************************/
template <typename T>
__device__ __inline__ static void
cab_to_cxyz(const smem_task<T> &task, const T *__restrict__ alpha,
            const T *__restrict__ cab, T *__restrict__ cxyz) {

  //   *** initialise the coefficient matrix, we transform the sum
  //
  // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
  //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya
  //         (z-a_z)**lza
  //
  // into
  //
  // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
  //
  // where p is center of the product gaussian, and lp = la_max + lb_max
  // (current implementation is l**7)

  // strides for accessing alpha
  const int s3 = (task.lp + 1);
  const int s2 = (task.la_max + 1) * s3;
  const int s1 = (task.lb_max + 1) * s2;

  // TODO: Maybe we can transpose alpha to index it directly with ico and jco.
  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);

  for (int i = tid; i < ncoset(task.lp);
       i += blockDim.x * blockDim.y * blockDim.z) {
    auto &co = coset_inv[i];
    T reg = 0.0; // accumulate into a register
    for (int jco = 0; jco < ncoset(task.lb_max); jco++) {
      const auto &b = coset_inv[jco];
      for (int ico = 0; ico < ncoset(task.la_max); ico++) {
        const auto &a = coset_inv[ico];
        const T p = alpha[0 * s1 + b.l[0] * s2 + a.l[0] * s3 + co.l[0]] *
                    alpha[1 * s1 + b.l[1] * s2 + a.l[1] * s3 + co.l[1]] *
                    alpha[2 * s1 + b.l[2] * s2 + a.l[2] * s3 + co.l[2]];
        reg += p * cab[jco * task.n1 + ico]; // collocate
      }
    }

    cxyz[i] = task.prefactor * reg;
  }
  __syncthreads(); // because of concurrent writes to cxyz / cab
}

/*******************************************************************************
 * \brief Transforms coefficients C_xyz into C_ab.
 ******************************************************************************/
template <typename T>
__device__ __inline__ static void
cxyz_to_cab(const smem_task<T> &task, const T *__restrict__ alpha,
            const T *__restrict__ cxyz, T *__restrict__ cab) {

  //   *** initialise the coefficient matrix, we transform the sum
  //
  // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
  //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya
  //         (z-a_z)**lza
  //
  // into
  //
  // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
  //
  // where p is center of the product gaussian, and lp = la_max + lb_max
  // (current implementation is l**7)

  // strides for accessing alpha
  const int s3 = (task.lp + 1);
  const int s2 = (task.la_max + 1) * s3;
  const int s1 = (task.lb_max + 1) * s2;

  const int tid =
      threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  for (int jco = tid / 8; jco < ncoset(task.lb_max); jco += 8) {
    const orbital b = coset_inv[jco];
    for (int ico = tid % 8; ico < ncoset(task.la_max); ico += 8) {
      const auto &a = coset_inv[ico];
      T reg = 0.0; // accumulate into a register
      for (int ic = 0; ic < ncoset(task.lp); ic++) {
        const auto &co = coset_inv[ic];
        const T p = alpha[b.l[0] * s2 + a.l[0] * s3 + co.l[0]] *
                    alpha[s1 + b.l[1] * s2 + a.l[1] * s3 + co.l[1]] *
                    alpha[2 * s1 + b.l[2] * s2 + a.l[2] * s3 + co.l[2]];

        reg += p * cxyz[ic]; // integrate
      }
      cab[jco * task.n1 + ico] =
          task.prefactor * reg; // partial loop coverage -> zero it
    }
  }
}

/*******************************************************************************
 * \brief Copies a task from global to shared memory.
 ******************************************************************************/

/* Collocate and integrate do not care about the sphi etc... computing the
 * coefficients do not care about the exponentials (the parameters are still
 * needed), so simplify the amount of information to save shared memory (it is
 * crucial on AMD GPU to max out occupancy) */
template <typename T, typename T3>
__device__ __inline__ void
fill_smem_task_reduced(const kernel_params &dev, const int task_id,
                       smem_task_reduced<T, T3> &task) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const auto &glb_task = dev.tasks[task_id];
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
  __syncthreads();
}

/*******************************************************************************
 * \brief Copies a task from global to shared memory and does precomputations.
 ******************************************************************************/

/*  computing the coefficients do not care about many of the exponentials
 * parameters, etc.. so */
template <typename T>
__device__ __inline__ void fill_smem_task_coef(const kernel_params &dev,
                                               const int task_id,
                                               smem_task<T> &task) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const auto &glb_task = dev.tasks[task_id];
    const int iatom = glb_task.iatom;
    const int jatom = glb_task.jatom;
    task.zeta = glb_task.zeta;
    task.zetb = glb_task.zetb;

    for (int i = 0; i < 3; i++) {
      task.rab[i] = glb_task.rab[i];
      task.ra[i] = glb_task.ra[i];
      task.rb[i] = task.ra[i] + task.rab[i];
      task.rp[i] =
          task.ra[i] + task.rab[i] * task.zetb / (task.zeta + task.zetb);
    }

    task.prefactor = glb_task.prefactor;
    task.off_diag_twice = glb_task.off_diag_twice;

    // angular momentum range of basis set
    const int la_max_basis = glb_task.la_max;
    const int lb_max_basis = glb_task.lb_max;
    const int la_min_basis = glb_task.la_min;
    const int lb_min_basis = glb_task.lb_min;

    // angular momentum range for the actual collocate/integrate opteration.
    task.la_max = la_max_basis + dev.la_max_diff;
    task.lb_max = lb_max_basis + dev.lb_max_diff;
    task.la_min = max(la_min_basis + (int)dev.la_min_diff, 0);
    task.lb_min = max(lb_min_basis + (int)dev.lb_min_diff, 0);
    task.lp = task.la_max + task.lb_max;

    // start of decontracted set, ie. pab and hab
    task.first_coseta = (la_min_basis > 0) ? ncoset(la_min_basis - 1) : 0;
    task.first_cosetb = (lb_min_basis > 0) ? ncoset(lb_min_basis - 1) : 0;

    // size of decontracted set, ie. pab and hab
    task.ncoseta = ncoset(la_max_basis);
    task.ncosetb = ncoset(lb_max_basis);

    // size of the cab matrix
    task.n1 = ncoset(task.la_max);
    task.n2 = ncoset(task.lb_max);

    // size of entire spherical basis
    task.nsgfa = glb_task.nsgfa; // ibasis.nsgf;
    task.nsgfb = glb_task.nsgfb;

    // size of spherical set
    task.nsgf_seta = glb_task.nsgf_seta;
    task.nsgf_setb = glb_task.nsgf_setb;

    // strides of the sphi transformation matrices
    task.maxcoa = glb_task.maxcoa;
    task.maxcob = glb_task.maxcob;

    // transformations from contracted spherical to primitive cartesian basis
    task.sphia = &dev.sphi_dev[glb_task.ikind][glb_task.sgfa * task.maxcoa +
                                               glb_task.ipgf * task.ncoseta];
    task.sphib = &dev.sphi_dev[glb_task.jkind][glb_task.sgfb * task.maxcob +
                                               glb_task.jpgf * task.ncosetb];

    // Locate current matrix block within the buffer.
    const int block_offset = dev.block_offsets[glb_task.block_num];
    task.block_transposed = glb_task.block_transposed;
    task.pab_block =
        dev.buffers_dev.pab_block + block_offset + glb_task.subblock_offset;

    if (dev.buffers_dev.hab_block != nullptr) {
      task.hab_block =
          dev.buffers_dev.hab_block + block_offset + glb_task.subblock_offset;
      if (dev.buffers_dev.forces != nullptr) {
        task.forces_a = &dev.buffers_dev.forces[3 * iatom];
        task.forces_b = &dev.buffers_dev.forces[3 * jatom];
      }
    }
  }
  __syncthreads();
}

/*******************************************************************************
 * \brief Size (in doubles) of the work arrays of the two step spherical <->
 *        cartesian transformations of calculate_coefficients_two_step and
 *        compute_hab_two_step: [max_nsgf_set][ld] and [ld][ld] with
 *        ld = ncoset(lmax).
 ******************************************************************************/
inline int sphi_work_size(const int lmax, const int max_nsgf_set) {
  const int ld = ncoset(lmax);
  return max_nsgf_set * ld + ld * ld;
}

// shared memory budget of the two step kernels (default limit without opting
// in on NVIDIA hardware). Above, the fused kernels are used.
constexpr size_t sphi_two_step_max_smem = 48 * 1024;

class smem_parameters {
private:
  int la_max_{-1};
  int lb_max_{-1};
  int smem_per_block_{0};
  int alpha_size_{-1};
  int cab_size_{-1};
  int lp_max_{-1};
  ldiffs_value ldiffs_;
  int lp_diff_{-1};

public:
  smem_parameters(const ldiffs_value ldiffs, const int lmax) {
    ldiffs_ = ldiffs;
    lp_diff_ = ldiffs.la_max_diff + ldiffs.lb_max_diff;
    la_max_ = lmax + ldiffs.la_max_diff;
    lb_max_ = lmax + ldiffs.lb_max_diff;
    lp_max_ = la_max_ + lb_max_;

    // NB: cab is allocated in global memory not shared memory. Each block has
    // its own cab space

    cab_size_ = (rocm_backend::ncoset(la_max_) * rocm_backend::ncoset(lb_max_));
    alpha_size_ = 3 * (lb_max_ + 1) * (la_max_ + 1) * (lp_max_ + 1);
    smem_per_block_ = std::max(alpha_size_, 64) * sizeof(double);

    if (smem_per_block_ > 64 * 1024) {
      fprintf(stderr,
              "ERROR: Not enough shared memory in grid_gpu_collocate.\n");
      fprintf(stderr, "alpha_len: %i, ", alpha_size_);
      fprintf(stderr, "total smem_per_block: %f kB\n\n",
              smem_per_block_ / 1024.0);
      abort();
    }
  }

  ~smem_parameters() {};

  // copy and move are trivial

  inline int alpha_size() const { return alpha_size_; }

  inline int cab_size() const { return cab_size_; }

  inline int smem_per_block() const { return smem_per_block_; }

  inline int lp_diff() const { return lp_diff_; }

  inline ldiffs_value ldiffs() const { return ldiffs_; }

  inline int lp_max() const { return lp_max_; }

  inline int cxyz_size() const { return ncoset(lp_max_); }
};
template <typename T>
__inline__ __device__ T *allocate_workspace(const kernel_params &dev_) {
  unsigned int offset = dev_.cab_block_offset_dev[block_index()];
  return (T *)(dev_.buffers_dev.cab + offset);
}

template <typename T, typename T3>
__device__ __inline__ T
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
__device__ __inline__ void
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

} // namespace rocm_backend
#endif
/* ===== end of src/grid/gpu/grid_gpu_internal_header.h ===== */

/* ===== begin of src/grid/gpu/grid_gpu_prepare_pab.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * NOTE : derived from the reference and GPU backends
 * Authors :
 - Dr Mathieu Taillefumier (ETH Zurich / CSCS)
 - Advanced Micro Devices, Inc.
*/

#ifndef GRID_GPU_PROCESS_PAB_H
#define GRID_GPU_PROCESS_PAB_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

namespace rocm_backend {
/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_AB, ie. identity transformation.
 ******************************************************************************/
template <typename T>
__device__ __inline__ void prepare_pab_AB(const orbital a, const orbital b,
                                          const T pab_val, const int n,
                                          T *cab) {

  // simply copy pab to cab
  prep_term<T>(a, b, pab_val, n, cab);
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_DADB.
 ******************************************************************************/
template <typename T>
__device__ __inline__ void
prepare_pab_DADB(const orbital a, const orbital b, const T zeta, const T zetb,
                 const T pab_val, const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with 0.5 * (nabla pgf_a) . (nabla pgf_b)
  // (ddx pgf_a ) (ddx pgf_b) = (lax pgf_{a-1x} - 2*zeta*pgf_{a+1x})*(lbx
  // pgf_{b-1x} - 2*zetb*pgf_{b+1x})

  for (int i = 0; i < 3; i++) {
    prep_term<T>(down(i, a), down(i, b), 0.5 * a.l[i] * b.l[i] * pab_val, n,
                 cab);
    prep_term<T>(down(i, a), up(i, b), -1.0 * a.l[i] * zetb * pab_val, n, cab);
    prep_term<T>(up(i, a), down(i, b), -1.0 * zeta * b.l[i] * pab_val, n, cab);
    prep_term<T>(up(i, a), up(i, b), 2.0 * zeta * zetb * pab_val, n, cab);
  }
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_ADBmDAB_{X,Y,Z}.
 ******************************************************************************/
template <typename T>
__device__ void prepare_pab_ADBmDAB(const int idir, const orbital a,
                                    const orbital b, const T zeta, const T zetb,
                                    const T pab_val, const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with
  //    pgf_a (nabla_{idir} pgf_b) - (nabla_{idir} pgf_a) pgf_b
  // ( pgf_a ) (ddx pgf_b) - (ddx pgf_a)( pgf_b ) =
  //          pgf_a *(lbx pgf_{b-1x} - 2*zetb*pgf_{b+1x}) -
  //                   (lax pgf_{a-1x} - 2*zeta*pgf_{a+1x}) pgf_b

  prep_term<T>(a, down(idir, b), +b.l[idir] * pab_val, n, cab);
  prep_term<T>(a, up(idir, b), -2.0 * zetb * pab_val, n, cab);
  prep_term<T>(down(idir, a), b, -a.l[idir] * pab_val, n, cab);
  prep_term<T>(up(idir, a), b, +2.0 * zeta * pab_val, n, cab);
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_ARDBmDARB_{X,Y,Z}{X,Y,Z}.
 ******************************************************************************/
template <typename T>
__device__ void prepare_pab_ARDBmDARB(const int idir, const int ir,
                                      const orbital a, const orbital b,
                                      const T zeta, const T zetb,
                                      const T pab_val, const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with
  // pgf_a (r-Rb)_{ir} (nabla_{idir} pgf_b) - (nabla_{idir} pgf_a) (r-Rb)_{ir}
  // pgf_b ( pgf_a )(r-Rb)_{ir} (ddx pgf_b) - (ddx pgf_a) (r-Rb)_{ir} ( pgf_b )
  // =
  //                        pgf_a *(lbx pgf_{b-1x+1ir} - 2*zetb*pgf_{b+1x+1ir})
  //                        -
  //                       (lax pgf_{a-1x} - 2*zeta*pgf_{a+1x}) pgf_{b+1ir}

  prep_term<T>(a, down(idir, up(ir, b)), b.l[idir] * pab_val, n, cab);
  prep_term<T>(a, up(idir, up(ir, b)), -2.0 * zetb * pab_val, n, cab);
  prep_term<T>(down(idir, a), up(ir, b), -a.l[idir] * pab_val, n, cab);
  prep_term<T>(up(idir, a), up(ir, b), +2.0 * zeta * pab_val, n, cab);
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_DABpADB_{X,Y,Z}.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ void prepare_pab_DABpADB(const int idir, const orbital a,
                                    const orbital b, const T zeta, const T zetb,
                                    const T pab_val, const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with
  //    pgf_a (nabla_{idir} pgf_b) + (nabla_{idir} pgf_a) pgf_b
  // ( pgf_a ) (ddx pgf_b) + (ddx pgf_a)( pgf_b ) =
  //          pgf_a *(lbx pgf_{b-1x} + 2*zetb*pgf_{b+1x}) +
  //                   (lax pgf_{a-1x} + 2*zeta*pgf_{a+1x}) pgf_b

  prep_term<T>(a, down(idir, b), b.l[idir] * pab_val, n, cab);
  prep_term<T>(a, up(idir, b), -2.0 * zetb * pab_val, n, cab);
  prep_term<T>(down(idir, a), b, a.l[idir] * pab_val, n, cab);
  prep_term<T>(up(idir, a), b, -2.0 * zeta * pab_val, n, cab);
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_{DX,DY,DZ}.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ void
prepare_pab_Di(const int ider, const orbital a, const orbital b, const T zeta,
               const T zetb, const T pab_val, const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with
  //   d_{ider1} pgf_a d_{ider1} pgf_b
  // dx pgf_a dx pgf_b =
  //        (lax pgf_{a-1x})*(lbx pgf_{b-1x}) - 2*zetb*lax*pgf_{a-1x}*pgf_{b+1x}
  //        -
  //         lbx pgf_{b-1x}*2*zeta*pgf_{a+1x}+ 4*zeta*zetab*pgf_{a+1x}pgf_{b+1x}

  prep_term<T>(down(ider, a), down(ider, b), a.l[ider] * b.l[ider] * pab_val, n,
               cab);
  prep_term<T>(down(ider, a), up(ider, b), -2.0 * a.l[ider] * zetb * pab_val, n,
               cab);
  prep_term<T>(up(ider, a), down(ider, b), -2.0 * zeta * b.l[ider] * pab_val, n,
               cab);
  prep_term<T>(up(ider, a), up(ider, b), +4.0 * zeta * zetb * pab_val, n, cab);
}

/*******************************************************************************
 * \brief Helper for grid_prepare_pab_DiDj.
 ******************************************************************************/
template <typename T>
__device__ __inline__ void oneterm_dijdij(const int idir, const T func_a,
                                          const orbital a, const orbital b,
                                          const T zetb, const int n, T *cab) {

  int i1, i2;
  if (idir == 0) {
    i1 = 0;
    i2 = 1;
  } else if (idir == 1) {
    i1 = 1;
    i2 = 2;
  } else if (idir == 2) {
    i1 = 2;
    i2 = 0;
  } else {
    return; // error
  }

  prep_term<T>(a, down(i1, down(i2, b)), b.l[i1] * b.l[i2] * func_a, n, cab);
  prep_term<T>(a, up(i1, down(i2, b)), -2.0 * zetb * b.l[i2] * func_a, n, cab);
  prep_term<T>(a, down(i1, up(i2, b)), -2.0 * zetb * b.l[i1] * func_a, n, cab);
  prep_term<T>(a, up(i1, up(i2, b)), +4.0 * zetb * zetb * func_a, n, cab);
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_{DXDY,DYDZ,DZDX}
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ void
prepare_pab_DiDj(const int ider1, const int ider2, const orbital a,
                 const orbital b, const T zeta, const T zetb, const T pab_val,
                 const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with
  //   d_{ider1} pgf_a d_{ider1} pgf_b

  const T func_a1 = a.l[ider1] * a.l[ider2] * pab_val;
  oneterm_dijdij<T>(ider1, func_a1, down(ider1, down(ider2, a)), b, zetb, n,
                    cab);

  const T func_a2 = -2.0 * zeta * a.l[ider2] * pab_val;
  oneterm_dijdij<T>(ider1, func_a2, up(ider1, down(ider2, a)), b, zetb, n, cab);

  const T func_a3 = -2.0 * zeta * a.l[ider1] * pab_val;
  oneterm_dijdij<T>(ider1, func_a3, down(ider1, up(ider2, a)), b, zetb, n, cab);

  const T func_a4 = 4.0 * zeta * zeta * pab_val;
  oneterm_dijdij<T>(ider1, func_a4, up(ider1, up(ider2, a)), b, zetb, n, cab);
}

/*******************************************************************************
 * \brief Helper for grid_prepare_pab_Di2.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ void oneterm_diidii(const int idir, const T func_a, const orbital a,
                               const orbital b, const T zetb, const int n,
                               T *cab) {

  prep_term<T>(a, down(idir, down(idir, b)),
               b.l[idir] * (b.l[idir] - 1) * func_a, n, cab);
  prep_term<T>(a, b, -2.0 * zetb * (2 * b.l[idir] + 1) * func_a, n, cab);
  prep_term<T>(a, up(idir, up(idir, b)), +4.0 * zetb * zetb * func_a, n, cab);
}

/*******************************************************************************
 * \brief Implementation of function GRID_FUNC_{DXDX,DYDY,DZDZ}
 ******************************************************************************/
template <typename T>
__device__ __inline__ void
prepare_pab_Di2(const int ider, const orbital a, const orbital b, const T zeta,
                const T zetb, const T pab_val, const int n, T *cab) {

  // creates cab such that mapping it with pgf_a pgf_b
  // is equivalent to mapping pab with
  //   dd_{ider1} pgf_a dd_{ider1} pgf_b

  const T func_a1 = a.l[ider] * (a.l[ider] - 1) * pab_val;
  oneterm_diidii<T>(ider, func_a1, down(ider, down(ider, a)), b, zetb, n, cab);

  const T func_a2 = -2.0 * zeta * (2 * a.l[ider] + 1) * pab_val;
  oneterm_diidii<T>(ider, func_a2, a, b, zetb, n, cab);

  const T func_a3 = 4.0 * zeta * zeta * pab_val;
  oneterm_diidii<T>(ider, func_a3, up(ider, up(ider, a)), b, zetb, n, cab);
}

/*******************************************************************************
 * \brief Transforms a given element of the density matrix according to func.
 * \param func          Transformation function to apply, one of GRID_FUNC_*.
 * \param {a,b}         Orbital angular momenta.
 * \param zet_{a,b}     Gaussian exponents.
 * \param pab_val       Input matrix element of pab.
 * \param n             Leading dimensions of output matrix cab.
 * \param cab           Output matrix.
 ******************************************************************************/
template <typename T>
__device__ __inline__ void
prepare_pab(const enum grid_func func, const orbital a, const orbital b,
            const T zeta, const T zetb, const T pab_val, const int n, T *cab) {

  // This switch statment will be in an inner loop but only with few iterations.
  switch (func) {
  case GRID_FUNC_AB:
    prepare_pab_AB<T>(a, b, pab_val, n, cab);
    break;
  case GRID_FUNC_DADB:
    prepare_pab_DADB<T>(a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ADBmDAB_X:
    prepare_pab_ADBmDAB<T>(0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ADBmDAB_Y:
    prepare_pab_ADBmDAB<T>(1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ADBmDAB_Z:
    prepare_pab_ADBmDAB<T>(2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_XX:
    prepare_pab_ARDBmDARB<T>(0, 0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_XY:
    prepare_pab_ARDBmDARB<T>(0, 1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_XZ:
    prepare_pab_ARDBmDARB<T>(0, 2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_YX:
    prepare_pab_ARDBmDARB<T>(1, 0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_YY:
    prepare_pab_ARDBmDARB<T>(1, 1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_YZ:
    prepare_pab_ARDBmDARB<T>(1, 2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_ZX:
    prepare_pab_ARDBmDARB<T>(2, 0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_ZY:
    prepare_pab_ARDBmDARB<T>(2, 1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_ARDBmDARB_ZZ:
    prepare_pab_ARDBmDARB<T>(2, 2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DABpADB_X:
    prepare_pab_DABpADB<T>(0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DABpADB_Y:
    prepare_pab_DABpADB<T>(1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DABpADB_Z:
    prepare_pab_DABpADB<T>(2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DX:
    prepare_pab_Di<T>(0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DY:
    prepare_pab_Di<T>(1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DZ:
    prepare_pab_Di<T>(2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DXDY:
    prepare_pab_DiDj<T>(0, 1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DYDZ:
    prepare_pab_DiDj<T>(1, 2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DZDX:
    prepare_pab_DiDj<T>(2, 0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DXDX:
    prepare_pab_Di2<T>(0, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DYDY:
    prepare_pab_Di2<T>(1, a, b, zeta, zetb, pab_val, n, cab);
    break;
  case GRID_FUNC_DZDZ:
    prepare_pab_Di2<T>(2, a, b, zeta, zetb, pab_val, n, cab);
    break;
  default:
    break; // Error: Unknown ga_gb_function - do nothing.
  }
}

/*******************************************************************************
 * \brief Returns difference in angular momentum range for given func.
 * \author Ole Schuett
 ******************************************************************************/
inline ldiffs_value prepare_get_ldiffs(const enum grid_func func) {
  ldiffs_value ldiffs;

  switch (func) {
  case GRID_FUNC_AB:
    ldiffs.la_max_diff = 0;
    ldiffs.la_min_diff = 0;
    ldiffs.lb_max_diff = 0;
    ldiffs.lb_min_diff = 0;
    break;
  case GRID_FUNC_DADB:
  case GRID_FUNC_ADBmDAB_X:
  case GRID_FUNC_ADBmDAB_Y:
  case GRID_FUNC_ADBmDAB_Z:
  case GRID_FUNC_DABpADB_X:
  case GRID_FUNC_DABpADB_Y:
  case GRID_FUNC_DABpADB_Z:
    ldiffs.la_max_diff = +1;
    ldiffs.la_min_diff = -1;
    ldiffs.lb_max_diff = +1;
    ldiffs.lb_min_diff = -1;
    break;
  case GRID_FUNC_ARDBmDARB_XX:
  case GRID_FUNC_ARDBmDARB_XY:
  case GRID_FUNC_ARDBmDARB_XZ:
  case GRID_FUNC_ARDBmDARB_YX:
  case GRID_FUNC_ARDBmDARB_YY:
  case GRID_FUNC_ARDBmDARB_YZ:
  case GRID_FUNC_ARDBmDARB_ZX:
  case GRID_FUNC_ARDBmDARB_ZY:
  case GRID_FUNC_ARDBmDARB_ZZ:
    ldiffs.la_max_diff = +1;
    ldiffs.la_min_diff = -1;
    ldiffs.lb_max_diff = +2; // this is legit
    ldiffs.lb_min_diff = -1;
    break;
  case GRID_FUNC_DX:
  case GRID_FUNC_DY:
  case GRID_FUNC_DZ:
    ldiffs.la_max_diff = +1;
    ldiffs.la_min_diff = -1;
    ldiffs.lb_max_diff = +1;
    ldiffs.lb_min_diff = -1;
    break;
  case GRID_FUNC_DXDY:
  case GRID_FUNC_DYDZ:
  case GRID_FUNC_DZDX:
  case GRID_FUNC_DXDX:
  case GRID_FUNC_DYDY:
  case GRID_FUNC_DZDZ:
    ldiffs.la_max_diff = +2;
    ldiffs.la_min_diff = -2;
    ldiffs.lb_max_diff = +2;
    ldiffs.lb_min_diff = -2;
    break;
  default:
    fprintf(stderr, "Error: Unknown ga_gb_function %i.\n", func);
    abort();
  }

  return ldiffs;
}
} // namespace rocm_backend
#endif
/* ===== end of src/grid/gpu/grid_gpu_prepare_pab.h ===== */

/* ===== begin of src/grid/gpu/grid_gpu_process_vab.h (verbatim, CP2K revision c05efa7fa7) ===== */
/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * NOTE : derived from the reference and GPU backends
 * Authors :
 - Dr Mathieu Taillefumier (ETH Zurich / CSCS)
 - Advanced Micro Devices, Inc.
*/

#ifndef GRID_GPU_PROCESS_VAB_H
#define GRID_GPU_PROCESS_VAB_H


/* taken from ../common/grid_process_vab.h but added template parameter */
namespace rocm_backend {
/*******************************************************************************
 * \brief Returns matrix element cab[idx(b)][idx(a)].
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ T get_term(const orbital &a, const orbital &b,
                                 const int n, const T *cab) {
  return cab[idx(b) * n + idx(a)];
}

/*******************************************************************************
 * \brief Returns i'th component of force on atom a for compute_tau=false.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ T get_force_a_normal(const orbital &a, const orbital &b,
                                           const int i, const T zeta,
                                           const int n, const T *cab) {
  const T aip1 = get_term(up(i, a), b, n, cab);
  const T aim1 = get_term(down(i, a), b, n, cab);
  return 2.0 * zeta * aip1 - a.l[i] * aim1;
}

/*******************************************************************************
 * \brief Returns i'th component of force on atom a.
 * \author Ole Schuett
 ******************************************************************************/
template <bool compute_tau, typename T>
__device__ __inline__ double
get_force_a(const orbital &a, const orbital &b, const int i, const T zeta,
            const T zetb, const int n, const T *cab) {
  if (!compute_tau) {
    return get_force_a_normal(a, b, i, zeta, n, cab);
  } else {
    T force = 0.0;
    for (int k = 0; k < 3; k++) {
      auto a_up = up(k, a);
      auto a_down = down(k, a);
      auto b_up = up(k, b);
      auto b_down = down(k, b);
      force += 0.5 * a.l[k] * b.l[k] *
               get_force_a_normal(a_down, b_down, i, zeta, n, cab);
      force -=
          zeta * b.l[k] * get_force_a_normal(a_up, b_down, i, zeta, n, cab);
      force -=
          a.l[k] * zetb * get_force_a_normal(a_down, b_up, i, zeta, n, cab);
      force +=
          2.0 * zeta * zetb * get_force_a_normal(a_up, b_up, i, zeta, n, cab);
    }
    return force;
  }
}

/*******************************************************************************
 * \brief Returns i'th component of force on atom b for compute_tau=false.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ double
get_force_b_normal(const orbital &a, const orbital &b, const int i,
                   const T zetb, const T rab[3], const int n, const T *cab) {
  const T axpm0 = get_term(a, b, n, cab);
  const T aip1 = get_term(up(i, a), b, n, cab);
  const T bim1 = get_term(a, down(i, b), n, cab);
  return 2.0 * zetb * (aip1 - rab[i] * axpm0) - b.l[i] * bim1;
}

/*******************************************************************************
 * \brief Returns i'th component of force on atom b.
 * \author Ole Schuett
 ******************************************************************************/
template <bool compute_tau, typename T>
__device__ __inline__ T get_force_b(const orbital &a, const orbital &b,
                                    const int i, const T zeta, const T zetb,
                                    const T rab[3], const int n, const T *cab) {
  if (!compute_tau) {
    return get_force_b_normal(a, b, i, zetb, rab, n, cab);
  } else {
    T force = 0.0;
    for (int k = 0; k < 3; k++) {
      const auto a_up = up(k, a);
      const auto a_down = down(k, a);
      const auto b_up = up(k, b);
      const auto b_down = down(k, b);
      force += 0.5 * a.l[k] * b.l[k] *
               get_force_b_normal(a_down, b_down, i, zetb, rab, n, cab);
      force -= zeta * b.l[k] *
               get_force_b_normal(a_up, b_down, i, zetb, rab, n, cab);
      force -= a.l[k] * zetb *
               get_force_b_normal(a_down, b_up, i, zetb, rab, n, cab);
      force += 2.0 * zeta * zetb *
               get_force_b_normal(a_up, b_up, i, zetb, rab, n, cab);
    }
    return force;
  }
}

/*******************************************************************************
 * \brief Returns element i,j of virial on atom a for compute_tau=false.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ double
get_virial_a_normal(const orbital &a, const orbital &b, const int i,
                    const int j, const T zeta, const int n, const T *cab) {
  return 2.0 * zeta * get_term(up(i, up(j, a)), b, n, cab) -
         a.l[j] * get_term(up(i, down(j, a)), b, n, cab);
}

/*******************************************************************************
 * \brief Returns element i,j of virial on atom a.
 * \author Ole Schuett
 ******************************************************************************/
template <bool compute_tau, typename T>
__device__ __inline__ T get_virial_a(const orbital &a, const orbital &b,
                                     const int i, const int j, const T zeta,
                                     const T zetb, const int n, const T *cab) {

  if (!compute_tau) {
    return get_virial_a_normal(a, b, i, j, zeta, n, cab);
  } else {
    T virial = 0.0;
    for (int k = 0; k < 3; k++) {
      const auto a_up = up(k, a);
      const auto a_down = down(k, a);
      const auto b_up = up(k, b);
      const auto b_down = down(k, b);
      virial += 0.5 * a.l[k] * b.l[k] *
                get_virial_a_normal(a_down, b_down, i, j, zeta, n, cab);
      virial -=
          zeta * b.l[k] * get_virial_a_normal(a_up, b_down, i, j, zeta, n, cab);
      virial -=
          a.l[k] * zetb * get_virial_a_normal(a_down, b_up, i, j, zeta, n, cab);
      virial += 2.0 * zeta * zetb *
                get_virial_a_normal(a_up, b_up, i, j, zeta, n, cab);
    }
    return virial;
  }
}

/*******************************************************************************
 * \brief Returns element i,j of virial on atom b for compute_tau=false.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T>
__device__ __inline__ double
get_virial_b_normal(const orbital &a, const orbital &b, const int i,
                    const int j, const T zetb, const T rab[3], const int n,
                    const T *cab) {

  return 2.0 * zetb *
             (get_term(up(i, up(j, a)), b, n, cab) -
              get_term(up(i, a), b, n, cab) * rab[j] -
              get_term(up(j, a), b, n, cab) * rab[i] +
              get_term(a, b, n, cab) * rab[j] * rab[i]) -
         b.l[j] * get_term(a, up(i, down(j, b)), n, cab);
}

/*******************************************************************************
 * \brief Returns element i,j of virial on atom b.
 * \author Ole Schuett
 ******************************************************************************/
template <bool compute_tau, typename T>
__device__ __inline__ double
get_virial_b(const orbital &a, const orbital &b, const int i, const int j,
             const T zeta, const T zetb, const T rab[3], const int n,
             const T *cab) {

  if (!compute_tau) {
    return get_virial_b_normal(a, b, i, j, zetb, rab, n, cab);
  } else {
    T virial = 0.0;
    for (int k = 0; k < 3; k++) {
      const auto a_up = up(k, a);
      const auto a_down = down(k, a);
      const auto b_up = up(k, b);
      const auto b_down = down(k, b);
      virial += 0.5 * a.l[k] * b.l[k] *
                get_virial_b_normal(a_down, b_down, i, j, zetb, rab, n, cab);
      virial -= zeta * b.l[k] *
                get_virial_b_normal(a_up, b_down, i, j, zetb, rab, n, cab);
      virial -= a.l[k] * zetb *
                get_virial_b_normal(a_down, b_up, i, j, zetb, rab, n, cab);
      virial += 2.0 * zeta * zetb *
                get_virial_b_normal(a_up, b_up, i, j, zetb, rab, n, cab);
    }
    return virial;
  }
}

/*******************************************************************************
 * \brief Returns element i,j of hab matrix.
 * \author Ole Schuett
 ******************************************************************************/
template <bool compute_tau, typename T>
__device__ __inline__ T get_hab(const orbital &a, const orbital &b,
                                const T zeta, const T zetb, const int n,
                                const T *cab) {
  if (!compute_tau) {
    return get_term(a, b, n, cab);
  } else {
    T hab = 0.0;
    for (int k = 0; k < 3; k++) {
      const auto a_up = up(k, a);
      const auto a_down = down(k, a);
      const auto b_up = up(k, b);
      const auto b_down = down(k, b);
      hab += 0.5 * a.l[k] * b.l[k] * get_term(a_down, b_down, n, cab);
      hab -= zeta * b.l[k] * get_term(a_up, b_down, n, cab);
      hab -= a.l[k] * zetb * get_term(a_down, b_up, n, cab);
      hab += 2.0 * zeta * zetb * get_term(a_up, b_up, n, cab);
    }
    return hab;
  }
}

/*******************************************************************************
 * \brief Returns difference in angular momentum range for given flags.
 ******************************************************************************/
inline ldiffs_value process_get_ldiffs(bool calculate_forces,
                                       bool calculate_virial,
                                       bool compute_tau) {
  ldiffs_value ldiffs;

  ldiffs.la_max_diff = 0;
  ldiffs.lb_max_diff = 0;
  ldiffs.la_min_diff = 0;
  ldiffs.lb_min_diff = 0;

  if (calculate_forces || calculate_virial) {
    ldiffs.la_max_diff += 1; // for deriv. of gaussian, unimportant which one
    ldiffs.la_min_diff -= 1;
    ldiffs.lb_min_diff -= 1;
  }

  if (calculate_virial) {
    ldiffs.la_max_diff += 1;
    ldiffs.lb_max_diff += 1;
  }

  if (compute_tau) {
    ldiffs.la_max_diff += 1;
    ldiffs.lb_max_diff += 1;
    ldiffs.la_min_diff -= 1;
    ldiffs.lb_min_diff -= 1;
  }

  return ldiffs;
}
}; // namespace rocm_backend
#endif
/* ===== end of src/grid/gpu/grid_gpu_process_vab.h ===== */

namespace rocm_backend {

/*******************************************************************************
 * Verbatim from grid_gpu_collocate.cu
 ******************************************************************************/
template <typename T, bool IS_FUNC_AB>
__device__ __inline__ void block_to_cab(const kernel_params &params,
                                        const smem_task<T> &task, T *cab) {

  // The spherical index runs over angular momentum and then over contractions.
  // The cartesian index runs over exponents and then over angular momentum.

  // This is a T matrix product. Since the pab block can be quite large the
  // two products are fused to conserve shared memory.
  const int tid = thread_global_index();

  for (int jco = task.first_cosetb + tid / 16; jco < task.ncosetb; jco += 4) {
    for (int ico = task.first_coseta + (tid % 16); ico < task.ncoseta;
         ico += 16) {
      T pab_val = 0.0;
      for (int i = 0; i < task.nsgf_setb; i++) {
        const T sphib = task.sphib[i * task.maxcob + jco];
        T tmp_val = 0.0;
        for (int j = 0; j < task.nsgf_seta; j++) {
          T block_val;
          if (task.block_transposed) {
            block_val =
                task.pab_block[j * task.nsgfb + i] * task.off_diag_twice;
          } else {
            block_val =
                task.pab_block[i * task.nsgfa + j] * task.off_diag_twice;
          }
          tmp_val += block_val * task.sphia[j * task.maxcoa + ico];
        }
        pab_val += tmp_val * sphib;
      }

      if (IS_FUNC_AB) {
        cab[jco * task.ncoseta + ico] += pab_val;
      } else {
        const auto a = coset_inv[ico];
        const auto b = coset_inv[jco];
        // remains the atomicAdd used in prepare_pab. Are they needed ?
        prepare_pab(params.func, a, b, task.zeta, task.zetb, pab_val, task.n1,
                    cab);
      }
    }
  }
}

template <typename T, bool IS_FUNC_AB>
__global__
__launch_bounds__(64) void calculate_coefficients(const kernel_params dev_) {
  // Copy task from global to shared memory and precompute some stuff.
  extern __shared__ T shared_memory[];
  const int number_of_tasks = dev_.num_tasks_per_block_dev[block_index()];

  if (number_of_tasks == 0)
    return;

  T *smem_alpha = &shared_memory[0];
  const int tid = thread_global_index();
  const int offset = dev_.sorted_blocks_offset_dev[block_index()];
  T *smem_cab = reinterpret_cast<double *>(
      __builtin_assume_aligned(allocate_workspace<T>(dev_), 32));

  for (int tk = 0; tk < number_of_tasks; tk++) {
    __shared__ smem_task<T> task;
    const int task_id = dev_.task_sorted_by_blocks_dev[offset + tk];
    if (dev_.tasks[task_id].skip_task)
      continue;
    fill_smem_task_coef(dev_, task_id, task);

    T *__restrict__ coef_ =
        &dev_.buffers_dev.coef[dev_.tasks[task_id].coef_offset];

    compute_alpha(task, smem_alpha);

    for (int z = tid; z < task.n1 * task.n2;
         z += blockDim.x * blockDim.y * blockDim.z)
      smem_cab[z] = 0.0;
    __syncthreads();

    block_to_cab<T, IS_FUNC_AB>(dev_, task, smem_cab);
    __syncthreads();

    cab_to_cxyz(task, smem_alpha, smem_cab, coef_);
    __syncthreads();
  }
}

/*******************************************************************************
 * Verbatim from grid_gpu_integrate.cu
 ******************************************************************************/
template <typename T, bool COMPUTE_TAU>
__global__ __launch_bounds__(64) void compute_hab(const kernel_params dev_) {
  // Copy task from global to shared memory and precompute some stuff.
  extern __shared__ T shared_memory[];
  // T *smem_cab = &shared_memory[dev_.smem_cab_offset];
  const int number_of_tasks = dev_.num_tasks_per_block_dev[block_index()];

  if (number_of_tasks == 0)
    return;

  T *smem_alpha = &shared_memory[0];
  const int tid = thread_global_index();
  const int offset = dev_.sorted_blocks_offset_dev[block_index()];

  T *__restrict__ smem_cab = nullptr;

  smem_cab = allocate_workspace<T>(dev_);

  for (int tk = 0; tk < number_of_tasks; tk++) {
    __shared__ smem_task<T> task;
    const int task_id = dev_.task_sorted_by_blocks_dev[offset + tk];
    if (dev_.tasks[task_id].skip_task)
      continue;

    // all warps need to be synchronized here before modifying the task
    // information.
    __syncthreads();
    fill_smem_task_coef(dev_, task_id, task);

    T *__restrict__ coef_ = reinterpret_cast<T *>(__builtin_assume_aligned(
        &dev_.buffers_dev.coef[dev_.tasks[task_id].coef_offset], 32));

    __syncthreads();
    compute_alpha(task, smem_alpha);
    __syncthreads();
    cxyz_to_cab(task, smem_alpha, coef_, smem_cab);
    __syncthreads();

    for (int i = tid / 8; i < task.nsgf_setb; i += 8) {
      for (int j = tid % 8; j < task.nsgf_seta; j += 8) {
        T tmp = 0.0;
        for (int jco = task.first_cosetb; jco < task.ncosetb; jco++) {
          const T sphib = task.sphib[i * task.maxcob + jco];
          const auto &b = coset_inv[jco];
          for (int ico = task.first_coseta; ico < task.ncoseta; ico++) {
            const auto &a = coset_inv[ico];
            const T hab = get_hab<COMPUTE_TAU, T>(a, b, task.zeta, task.zetb,
                                                  task.n1, smem_cab);
            const T sphia_times_sphib =
                task.sphia[j * task.maxcoa + ico] * sphib;
            tmp += hab * sphia_times_sphib;
          }
        }
        if (task.block_transposed) {
          task.hab_block[j * task.nsgfb + i] += tmp;
        } else {
          task.hab_block[i * task.nsgfa + j] += tmp;
        }
      }
    }
  }
}

} // namespace rocm_backend

using namespace rocm_backend;

/*******************************************************************************
 * Serial CPU port of calculate_coefficients and compute_hab. It follows the
 * definitions (fill_smem_task_coef, compute_alpha, block_to_cab, cab_to_cxyz,
 * cxyz_to_cab, get_hab) with plain loops.
 ******************************************************************************/
struct host_data {
  std::vector<orbital> coset_inv;        // coset index -> (lx, ly, lz)
  std::vector<std::vector<double>> sphi; // per kind, [nsgf][maxco]
  std::vector<int> block_offsets;        // offset of each block in pab/hab
};

static double binomial(const int n, const int k) {
  double b = 1.0;
  for (int i = 1; i <= k; i++)
    b = b * (n - k + i) / i;
  return b;
}

static int host_coset(const int lx, const int ly, const int lz) {
  return coset(lx, ly, lz);
}

// (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
static void host_alpha(const task_info &t, const int la_max, const int lb_max,
                       std::vector<double> &alpha) {
  const int lp = la_max + lb_max;
  const int s3 = lp + 1;
  const int s2 = (la_max + 1) * s3;
  const int s1 = (lb_max + 1) * s2;
  alpha.assign(3 * s1, 0.0);
  for (int idir = 0; idir < 3; idir++) {
    const double ra = t.ra[idir];
    const double rb = ra + t.rab[idir];
    const double rp = ra + t.rab[idir] * t.zetb / (t.zeta + t.zetb);
    const double drpa = rp - ra, drpb = rp - rb;
    for (int la = 0; la <= la_max; la++) {
      for (int lb = 0; lb <= lb_max; lb++) {
        double a = 1.0;
        for (int k = 0; k <= la; k++) {
          double b = 1.0;
          for (int l = 0; l <= lb; l++) {
            alpha[idir * s1 + lb * s2 + la * s3 + la - k + lb - l] +=
                a * b * binomial(la, k) * binomial(lb, l);
            b *= drpb;
          }
          a *= drpa;
        }
      }
    }
  }
}

static double host_alpha_product(const std::vector<double> &alpha,
                                 const int la_max, const int lb_max,
                                 const orbital &a, const orbital &b,
                                 const orbital &co) {
  const int s3 = la_max + lb_max + 1;
  const int s2 = (la_max + 1) * s3;
  const int s1 = (lb_max + 1) * s2;
  double p = 1.0;
  for (int d = 0; d < 3; d++)
    p *= alpha[d * s1 + b.l[d] * s2 + a.l[d] * s3 + co.l[d]];
  return p;
}

static double host_term(const orbital &a, const orbital &b, const int n,
                        const std::vector<double> &cab) {
  return cab[host_coset(b.l[0], b.l[1], b.l[2]) * n +
             host_coset(a.l[0], a.l[1], a.l[2])];
}

static double host_hab(const bool tau, const orbital &a, const orbital &b,
                       const double zeta, const double zetb, const int n,
                       const std::vector<double> &cab) {
  if (!tau)
    return host_term(a, b, n, cab);
  double hab = 0.0;
  for (int k = 0; k < 3; k++) {
    orbital a_up = a, a_down = a, b_up = b, b_down = b;
    a_up.l[k] += 1;
    b_up.l[k] += 1;
    a_down.l[k] = std::max(0, a.l[k] - 1);
    b_down.l[k] = std::max(0, b.l[k] - 1);
    hab += 0.5 * a.l[k] * b.l[k] * host_term(a_down, b_down, n, cab);
    hab -= zeta * b.l[k] * host_term(a_up, b_down, n, cab);
    hab -= a.l[k] * zetb * host_term(a_down, b_up, n, cab);
    hab += 2.0 * zeta * zetb * host_term(a_up, b_up, n, cab);
  }
  return hab;
}

// collocate: pab -> cab -> cxyz, GRID_FUNC_AB only (no angular momentum shift)
static void calculate_coefficients_reference(
    const host_data &h, const std::vector<task_info> &tasks,
    const std::vector<double> &pab, std::vector<double> &coef) {
  std::vector<double> alpha, cab, tmp;
  for (const task_info &t : tasks) {
    if (t.skip_task)
      continue;
    const int ncoa = ncoset(t.la_max), ncob = ncoset(t.lb_max);
    const int first_coa = (t.la_min > 0) ? ncoset(t.la_min - 1) : 0;
    const int first_cob = (t.lb_min > 0) ? ncoset(t.lb_min - 1) : 0;
    const double *sphia = &h.sphi[t.ikind][t.sgfa * t.maxcoa + t.ipgf * ncoa];
    const double *sphib = &h.sphi[t.jkind][t.sgfb * t.maxcob + t.jpgf * ncob];
    const double *block =
        &pab[h.block_offsets[t.block_num] + t.subblock_offset];

    // cab[jco][ico] = sum_ij sphib[i][jco] pab[i][j] sphia[j][ico], done as
    // two matrix products
    tmp.assign(t.nsgf_setb * ncoa, 0.0);
    for (int i = 0; i < t.nsgf_setb; i++) {
      for (int j = 0; j < t.nsgf_seta; j++) {
        const double p = (t.block_transposed ? block[j * t.nsgfb + i]
                                             : block[i * t.nsgfa + j]) *
                         t.off_diag_twice;
        for (int ico = first_coa; ico < ncoa; ico++)
          tmp[i * ncoa + ico] += p * sphia[j * t.maxcoa + ico];
      }
    }
    cab.assign(ncoa * ncob, 0.0);
    for (int i = 0; i < t.nsgf_setb; i++)
      for (int jco = first_cob; jco < ncob; jco++)
        for (int ico = first_coa; ico < ncoa; ico++)
          cab[jco * ncoa + ico] +=
              sphib[i * t.maxcob + jco] * tmp[i * ncoa + ico];

    host_alpha(t, t.la_max, t.lb_max, alpha);
    for (int ic = 0; ic < ncoset(t.la_max + t.lb_max); ic++) {
      double sum = 0.0;
      for (int jco = 0; jco < ncob; jco++)
        for (int ico = 0; ico < ncoa; ico++)
          sum += host_alpha_product(alpha, t.la_max, t.lb_max, h.coset_inv[ico],
                                    h.coset_inv[jco], h.coset_inv[ic]) *
                 cab[jco * ncoa + ico];
      coef[t.coef_offset + ic] = t.prefactor * sum;
    }
  }
}

// integrate: cxyz -> cab -> hab, without forces
static void compute_hab_reference(const host_data &h,
                                  const std::vector<task_info> &tasks,
                                  const ldiffs_value &ldiffs, const bool tau,
                                  const std::vector<double> &coef,
                                  std::vector<double> &hab) {
  std::vector<double> alpha, cab, tmp;
  for (const task_info &t : tasks) {
    if (t.skip_task)
      continue;
    // angular momentum range of the operation
    const int la_max = t.la_max + ldiffs.la_max_diff;
    const int lb_max = t.lb_max + ldiffs.lb_max_diff;
    const int n1 = ncoset(la_max), n2 = ncoset(lb_max);
    // decontracted set of the basis
    const int ncoa = ncoset(t.la_max), ncob = ncoset(t.lb_max);
    const int first_coa = (t.la_min > 0) ? ncoset(t.la_min - 1) : 0;
    const int first_cob = (t.lb_min > 0) ? ncoset(t.lb_min - 1) : 0;
    const double *sphia = &h.sphi[t.ikind][t.sgfa * t.maxcoa + t.ipgf * ncoa];
    const double *sphib = &h.sphi[t.jkind][t.sgfb * t.maxcob + t.jpgf * ncob];
    double *block = &hab[h.block_offsets[t.block_num] + t.subblock_offset];

    host_alpha(t, la_max, lb_max, alpha);
    cab.assign(n1 * n2, 0.0);
    for (int jco = 0; jco < n2; jco++) {
      for (int ico = 0; ico < n1; ico++) {
        double sum = 0.0;
        for (int ic = 0; ic < ncoset(la_max + lb_max); ic++)
          sum += host_alpha_product(alpha, la_max, lb_max, h.coset_inv[ico],
                                    h.coset_inv[jco], h.coset_inv[ic]) *
                 coef[t.coef_offset + ic];
        cab[jco * n1 + ico] = t.prefactor * sum;
      }
    }

    // hab[i][j] += sum_{jco,ico} sphib[i][jco] H[jco][ico] sphia[j][ico],
    // done as two matrix products
    std::vector<double> hmat(ncob * ncoa, 0.0);
    for (int jco = first_cob; jco < ncob; jco++)
      for (int ico = first_coa; ico < ncoa; ico++)
        hmat[jco * ncoa + ico] = host_hab(
            tau, h.coset_inv[ico], h.coset_inv[jco], t.zeta, t.zetb, n1, cab);
    tmp.assign(t.nsgf_setb * ncoa, 0.0);
    for (int i = 0; i < t.nsgf_setb; i++)
      for (int jco = first_cob; jco < ncob; jco++)
        for (int ico = first_coa; ico < ncoa; ico++)
          tmp[i * ncoa + ico] +=
              sphib[i * t.maxcob + jco] * hmat[jco * ncoa + ico];
    for (int i = 0; i < t.nsgf_setb; i++) {
      for (int j = 0; j < t.nsgf_seta; j++) {
        double sum = 0.0;
        for (int ico = first_coa; ico < ncoa; ico++)
          sum += tmp[i * ncoa + ico] * sphia[j * t.maxcoa + ico];
        if (t.block_transposed)
          block[j * t.nsgfb + i] += sum;
        else
          block[i * t.nsgfa + j] += sum;
      }
    }
  }
}

/*******************************************************************************
 * Driver
 ******************************************************************************/
int main(int argc, char **argv) {
  int nblocks = 2000;
  int lmax = 3;
  int npgf = 6;
  int nctr = 2;
  bool random_l = false;
  bool tau = false;
  int nrep = 10;
  unsigned seed = 42;
  int mode = 2;

  int opt;
  while ((opt = getopt(argc, argv, "b:l:p:c:R:t:r:s:m:h")) != -1) {
    switch (opt) {
    case 'b':
      nblocks = atoi(optarg);
      break;
    case 'l':
      lmax = atoi(optarg);
      break;
    case 'p':
      npgf = atoi(optarg);
      break;
    case 'c':
      nctr = atoi(optarg);
      break;
    case 'R':
      random_l = atoi(optarg) != 0;
      break;
    case 't':
      tau = atoi(optarg) != 0;
      break;
    case 'r':
      nrep = atoi(optarg);
      break;
    case 's':
      seed = atoi(optarg);
      break;
    case 'm':
      mode = atoi(optarg);
      break;
    default:
      fprintf(stderr,
              "usage: %s [-b nblocks] [-l lmax] [-p npgf] [-c nctr] [-R 0|1]"
              " [-t 0|1] [-r nrep] [-s seed] [-m 0|1|2]\n",
              argv[0]);
      return 1;
    }
  }
  if (lmax < 0 || lmax > 6) {
    fprintf(stderr, "lmax must be in [0, 6]\n");
    return 1;
  }

  init_constant_memory();

  host_data h;
  h.coset_inv.resize(ncoset(18));
  for (int lx = 0; lx <= 18; lx++)
    for (int ly = 0; ly <= 18 - lx; ly++)
      for (int lz = 0; lz <= 18 - lx - ly; lz++)
        h.coset_inv[coset(lx, ly, lz)] = {{lx, ly, lz}};
  h.sphi.resize(lmax + 1);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);

  // one kind per angular momentum l = 0..lmax: a single set with npgf
  // primitives and nctr contractions per l, i.e. nsgf = nctr (l + 1)^2
  // spherical functions and maxco = npgf ncoset(l) primitive cartesians.
  std::vector<int> nsgf_kind(lmax + 1), maxco_kind(lmax + 1);
  std::vector<double *> sphi_dev_host(lmax + 1);
  for (int l = 0; l <= lmax; l++) {
    nsgf_kind[l] = nctr * (l + 1) * (l + 1);
    maxco_kind[l] = npgf * ncoset(l);
    std::vector<double> &sphi = h.sphi[l];
    sphi.resize((size_t)nsgf_kind[l] * maxco_kind[l]);
    for (auto &v : sphi)
      v = 2.0 * unif(gen) - 1.0;
    offloadMalloc((void **)&sphi_dev_host[l], sizeof(double) * sphi.size());
    offloadMemcpyHtoD(sphi_dev_host[l], sphi.data(),
                      sizeof(double) * sphi.size());
  }
  double **sphi_dev;
  offloadMalloc((void **)&sphi_dev, sizeof(double *) * (lmax + 1));
  offloadMemcpyHtoD(sphi_dev, sphi_dev_host.data(),
                    sizeof(double *) * (lmax + 1));

  // angular momentum differences of the two operations
  const ldiffs_value ldiffs_col = prepare_get_ldiffs(GRID_FUNC_AB);
  const ldiffs_value ldiffs_int = process_get_ldiffs(false, false, tau);

  // matrix blocks, each with all npgf x npgf primitive pairs as tasks
  std::vector<task_info> tasks;
  std::vector<int> block_offsets(nblocks), num_tasks(nblocks),
      sorted_offset(nblocks), cab_offset(nblocks), sorted_tasks;
  size_t block_size_total = 0, coef_size = 0;
  int max_nsgf_set = 0;
  const int max_diff = std::max(ldiffs_int.la_max_diff, 0);
  const int cab_size = ncoset(lmax + max_diff) * ncoset(lmax + max_diff);
  for (int b = 0; b < nblocks; b++) {
    const int la = random_l ? (int)(unif(gen) * (lmax + 1)) % (lmax + 1) : lmax;
    const int lb = random_l ? (int)(unif(gen) * (lmax + 1)) % (lmax + 1) : lmax;
    const int nsgfa = nsgf_kind[la], nsgfb = nsgf_kind[lb];
    max_nsgf_set = std::max(max_nsgf_set, std::max(nsgfa, nsgfb));
    block_offsets[b] = block_size_total;
    block_size_total += (size_t)nsgfa * nsgfb;
    sorted_offset[b] = tasks.size();
    num_tasks[b] = npgf * npgf;
    cab_offset[b] = b * cab_size;
    const bool transposed = unif(gen) < 0.5;
    const double off_diag_twice = (unif(gen) < 0.2) ? 1.0 : 2.0;
    double ra[3], rab[3];
    for (int d = 0; d < 3; d++) {
      ra[d] = 10.0 * unif(gen);
      rab[d] = 4.0 * unif(gen) - 2.0;
    }
    for (int ipgf = 0; ipgf < npgf; ipgf++) {
      for (int jpgf = 0; jpgf < npgf; jpgf++) {
        task_info t;
        memset(&t, 0, sizeof(task_info));
        t.iatom = 2 * b;
        t.jatom = 2 * b + 1;
        t.ikind = la;
        t.jkind = lb;
        t.ipgf = ipgf;
        t.jpgf = jpgf;
        t.block_num = b;
        t.zeta = 0.1 * pow(3.0, ipgf) * (0.5 + unif(gen));
        t.zetb = 0.1 * pow(3.0, jpgf) * (0.5 + unif(gen));
        t.zetp = t.zeta + t.zetb;
        for (int d = 0; d < 3; d++) {
          t.ra[d] = ra[d];
          t.rab[d] = rab[d];
          t.rb[d] = ra[d] + rab[d];
          t.rp[d] = ra[d] + rab[d] * t.zetb / t.zetp;
        }
        t.prefactor = unif(gen);
        t.off_diag_twice = off_diag_twice;
        t.la_max = la;
        t.lb_max = lb;
        t.la_min = 0;
        t.lb_min = 0;
        t.first_coseta = 0;
        t.first_cosetb = 0;
        t.ncoseta = ncoset(la);
        t.ncosetb = ncoset(lb);
        t.nsgfa = nsgfa;
        t.nsgfb = nsgfb;
        t.nsgf_seta = nsgfa;
        t.nsgf_setb = nsgfb;
        t.maxcoa = maxco_kind[la];
        t.maxcob = maxco_kind[lb];
        t.sgfa = 0;
        t.sgfb = 0;
        t.subblock_offset = 0;
        t.block_transposed = transposed;
        t.skip_task = false;
        // large enough for both operations
        const int lp = la + lb + 2 * std::max(ldiffs_int.la_max_diff, 0);
        t.lp_max = lp;
        t.coef_offset = coef_size;
        coef_size += ((ncoset(lp) + 3) / 4) * 4;
        sorted_tasks.push_back(tasks.size());
        tasks.push_back(t);
      }
    }
  }
  const int ntasks = tasks.size();
  h.block_offsets = block_offsets;

  std::vector<double> pab(block_size_total), coef_in(coef_size);
  for (auto &v : pab)
    v = 2.0 * unif(gen) - 1.0;
  for (auto &v : coef_in)
    v = 2.0 * unif(gen) - 1.0;

  // device buffers
  auto to_dev = [](const auto &vec) {
    using V = typename std::decay_t<decltype(vec)>::value_type;
    V *ptr;
    offloadMalloc((void **)&ptr, sizeof(V) * std::max<size_t>(vec.size(), 1));
    offloadMemcpyHtoD(ptr, vec.data(), sizeof(V) * vec.size());
    return ptr;
  };
  task_info *tasks_dev = to_dev(tasks);
  int *block_offsets_dev = to_dev(block_offsets);
  int *num_tasks_dev = to_dev(num_tasks);
  int *sorted_offset_dev = to_dev(sorted_offset);
  int *sorted_tasks_dev = to_dev(sorted_tasks);
  int *cab_offset_dev = to_dev(cab_offset);
  double *pab_dev = to_dev(pab);
  double *coef_in_dev = to_dev(coef_in);
  double *coef_dev, *hab_dev, *cab_dev;
  offloadMalloc((void **)&coef_dev, sizeof(double) * coef_size);
  offloadMalloc((void **)&hab_dev, sizeof(double) * block_size_total);
  offloadMalloc((void **)&cab_dev, sizeof(double) * (size_t)nblocks * cab_size);

  kernel_params params;
  params.tasks = tasks_dev;
  params.block_offsets = block_offsets_dev;
  params.sphi_dev = sphi_dev;
  params.ntasks = ntasks;
  params.task_sorted_by_blocks_dev = sorted_tasks_dev;
  params.sorted_blocks_offset_dev = sorted_offset_dev;
  params.num_tasks_per_block_dev = num_tasks_dev;
  params.cab_block_offset_dev = cab_offset_dev;
  params.buffers_dev.pab_block = pab_dev;
  params.buffers_dev.cab = cab_dev;
  params.func = GRID_FUNC_AB;

  printf("nblocks %d, ntasks %d, lmax %d%s, npgf %d, nsgf_set %d, tau %d\n",
         nblocks, ntasks, lmax, random_l ? " (random per block)" : "", npgf,
         max_nsgf_set, tau);

  const dim3 threads_per_block(4, 4, 4);
  bool all_passed = true;

  auto time_it = [&](auto &&launch, auto &&reset) {
    // the reset is a memset of the hab blocks, small compared to the kernels
    offloadDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < nrep; rep++) {
      reset();
      launch();
    }
    offloadDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() /
           std::max(nrep, 1);
  };

  auto compare = [&](const char *name, const std::vector<double> &ref,
                     const std::vector<double> &out, const double ms) {
    double max_diff = 0.0, norm = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
      norm = std::max(norm, fabs(ref[i]));
      max_diff = std::max(max_diff, fabs(ref[i] - out[i]));
    }
    const double rel = max_diff / std::max(norm, 1e-300);
    const bool passed = (rel < 1e-12) && (norm > 0.0);
    all_passed = all_passed && passed;
    printf("%-24s %9.3f ms, max rel diff %.3e (max |ref| %.3e) -> %s\n", name,
           ms, rel, norm, passed ? "PASSED" : "FAILED");
  };

  // ---------------------------------------------------------------- collocate
  if (mode == 0 || mode == 2) {
    smem_parameters smem_params(ldiffs_col, lmax);
    params.la_max_diff = ldiffs_col.la_max_diff;
    params.lb_max_diff = ldiffs_col.lb_max_diff;
    params.la_min_diff = ldiffs_col.la_min_diff;
    params.lb_min_diff = ldiffs_col.lb_min_diff;
    params.buffers_dev.coef = coef_dev;
    params.buffers_dev.hab_block = nullptr;
    const size_t smem_ref = smem_params.smem_per_block();
    auto reset = [&]() {};
    auto run_ref = [&]() {
      calculate_coefficients<double, true>
          <<<nblocks, threads_per_block, smem_ref>>>(params);
    };
    std::vector<double> ref(coef_size, 0.0), out(coef_size);
    offloadMemset(coef_dev, 0, sizeof(double) * coef_size);
    run_ref();
    OFFLOAD_CHECK(offloadGetLastError());
    offloadMemcpyDtoH(out.data(), coef_dev, sizeof(double) * coef_size);
    calculate_coefficients_reference(h, tasks, pab, ref);
    compare("calculate_coefficients", ref, out, time_it(run_ref, reset));
  }

  // ---------------------------------------------------------------- integrate
  if (mode == 1 || mode == 2) {
    smem_parameters smem_params(ldiffs_int, lmax);
    params.la_max_diff = ldiffs_int.la_max_diff;
    params.lb_max_diff = ldiffs_int.lb_max_diff;
    params.la_min_diff = ldiffs_int.la_min_diff;
    params.lb_min_diff = ldiffs_int.lb_min_diff;
    params.buffers_dev.coef = coef_in_dev;
    params.buffers_dev.hab_block = hab_dev;
    const size_t smem_ref = smem_params.smem_per_block();
    auto reset = [&]() {
      offloadMemsetAsync(hab_dev, 0, sizeof(double) * block_size_total, 0);
    };
    auto run_ref = [&]() {
      if (tau)
        compute_hab<double, true>
            <<<nblocks, threads_per_block, smem_ref>>>(params);
      else
        compute_hab<double, false>
            <<<nblocks, threads_per_block, smem_ref>>>(params);
    };
    std::vector<double> ref(block_size_total, 0.0), out(block_size_total);
    reset();
    run_ref();
    OFFLOAD_CHECK(offloadGetLastError());
    offloadMemcpyDtoH(out.data(), hab_dev, sizeof(double) * block_size_total);
    compute_hab_reference(h, tasks, ldiffs_int, tau, coef_in, ref);
    compare(tau ? "compute_hab<tau>" : "compute_hab", ref, out,
            time_it(run_ref, reset));
  }

  return all_passed ? 0 : 1;
}
