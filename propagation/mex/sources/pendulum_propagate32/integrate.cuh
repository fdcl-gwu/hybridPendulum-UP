#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>

#include "setup.cuh"

__global__ void flip_shift(const cuComplex* X, cuComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F);
__global__ void addup_F(cuComplex* dF, const int nTot);
__global__ void add_F(cuComplex* dF, const cuComplex* F, const int nTot);
__global__ void mulImg_FR(cuComplex* dF, const float c, const int nR);
__global__ void add_FMR_small(cuComplex* dF, const cuComplex* FMR, const int ind_cumR, const Size_F* size_F);
__global__ void add_FMR_large(cuComplex* dF, const cuComplex* FMR, const int ind_cumR, const Size_F* size_F);
__global__ void mulImg_FTot_small(cuComplex* dF, const float* c, const int dim, const Size_F* size_F);
__global__ void mulImg_FTot_large(cuComplex* dF, const float* c, const int dim, const int k, const Size_F* size_F);
__global__ void integrate_Fnew(cuComplex* Fnew, const cuComplex* Fold, const cuComplex* dF, const float dt, const int nTot);

__host__ void modify_F(const cuComplex* F, cuComplex* F_modify, bool reduce, Size_F* size_F);
__host__ void permute_F(cuComplex* F, bool R_first, const Size_F* size_F);
__host__ void modify_u(const cuComplex* u, cuComplex* u_modify, Size_F* size_F);

__host__ void deriv_x(float* c, const int n, const int B, const float L);
__host__ void get_dF_small(cuComplex* dF, const cuComplex* F, const cuComplex* X, const cuComplex* OJO, const cuComplex* MR,
	const float* L, const cuComplex* u, const float* const* CG, const Size_F* size_F, const Size_F* size_F_dev);
__host__ void get_dF_large(cuComplex* dF, cuComplex* F, const cuComplex* X, const cuComplex* OJO, const cuComplex* MR,
    const float* L, const cuComplex* u, const float* const* CG, const Size_F* size_F, const Size_F* size_F_dev);

