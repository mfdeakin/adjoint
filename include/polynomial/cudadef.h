
#ifndef _CUDADEF_H_
#define _CUDADEF_H_

#ifdef __CUDACC__

#include <cuda.h>

#ifndef NDEBUG
#define NDEBUG
#endif

#define CUDA_CALLABLE __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_SHARED __shared__
#else
#define CUDA_CALLABLE
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_SHARED
#endif

#endif
