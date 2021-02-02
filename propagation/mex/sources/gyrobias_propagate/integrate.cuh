#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>

#include "setup.cuh"

__global__ void flip_shift(const cuDoubleComplex* X, cuDoubleComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F);
__global__ void addup_F(cuDoubleComplex* dF, const int nTot);
__global__ void add_F(cuDoubleComplex* dF, const cuDoubleComplex* F, const int nTot);
__global__ void get_c(double* c, const int i, const int j, const double* G, const Size_F* size_F);
__global__ void get_biasRW_small(cuDoubleComplex* dF_temp, const cuDoubleComplex* Fold, const double* c, const int i, const int j,const  Size_F* size_F);
__global__ void get_biasRW_large(cuDoubleComplex* dF_temp, const cuDoubleComplex* Fold, const double* c, const int i, const int j, const int ijk_k, const Size_F* size_F);
__global__ void integrate_Fnew(cuDoubleComplex* Fnew, const cuDoubleComplex* Fold, const cuDoubleComplex* dF, const double dt, const int nTot);

__host__ void modify_F(const cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce, Size_F* size_F);
__host__ void permute_F(cuDoubleComplex* F, bool R_first, const Size_F* size_F);
__host__ void modify_u(cuDoubleComplex* u, cuDoubleComplex* u_modify, Size_F* size_F);

__host__ void get_dF_small(cuDoubleComplex* dF, const cuDoubleComplex* F, const cuDoubleComplex* X, const cuDoubleComplex* u,
	const double* G1, const double* G2, const Size_F* size_F, const Size_F* size_F_dev);
__host__ void get_dF_large(cuDoubleComplex* dF, cuDoubleComplex* F, const cuDoubleComplex* X, const cuDoubleComplex* u,
    const double* G1, const double* G2, const Size_F* size_F, const Size_F* size_F_dev);

