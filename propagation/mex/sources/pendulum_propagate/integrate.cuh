#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>

#include "setup.cuh"

__global__ void flip_shift(const cuDoubleComplex* X, cuDoubleComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F);
__global__ void addup_F(cuDoubleComplex* dF, const Size_F* size_F);
__global__ void add_F(cuDoubleComplex* dF, const cuDoubleComplex* F, const Size_F* size_F);
__global__ void mulImg_FR(cuDoubleComplex* dF, const double c, const Size_F* size_F);
__global__ void add_FMR(cuDoubleComplex* dF, const cuDoubleComplex* FMR, const int ind_cumR, const Size_F* size_F);
__global__ void mulImg_FTot(cuDoubleComplex* dF, const double* c, const int dim, const Size_F* size_F);
__global__ void integrate_Fnew(cuDoubleComplex* Fnew, cuDoubleComplex* Fold, const cuDoubleComplex* dF, const double dt, const Size_F* size_F);

__host__ void deriv_x(double* c, const int n, const int B, const double L);
__host__ void get_dF(cuDoubleComplex* dF, const cuDoubleComplex* F, const cuDoubleComplex* X, const cuDoubleComplex* OJO, const cuDoubleComplex* MR,
	const double* L, const cuDoubleComplex* u, const double* const* CG, const Size_F* size_F, const Size_F* size_F_dev);
