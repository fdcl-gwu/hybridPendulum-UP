#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>

#include "setup.cuh"

__global__ void flip_shift(const myComplex* X, myComplex* X_ijk, const int is, const int js, const Size_F* size_F);
__global__ void addup_F(myComplex* dF, const int nTot);
__global__ void add_F(myComplex* dF, const myComplex* F, const int nTot);
__global__ void mulImg_FR(myComplex* dF, const myReal c, const int nR);
__global__ void add_FMR(myComplex* dF, const myComplex* FMR, const int ind_cumR, const Size_F* size_F);
__global__ void mulImg_FTot(myComplex* dF, const myReal* c, const int dim, const Size_F* size_F);
__global__ void get_c(myReal* c, const int i, const int j, const myReal* L, const myReal* G, const Size_F* size_F);
__global__ void get_biasRW(myComplex* dF_temp, const myComplex* Fold, const myReal* c, const int i, const int j, const Size_F* size_F);
__global__ void integrate_Fnew(myComplex* Fnew, const myComplex* Fold, const myComplex* dF, const myReal dt, const int nTot);

__host__ void modify_F(const myComplex* F, myComplex* F_modify, bool reduce, Size_F* size_F);
__host__ void modify_u(const myComplex* u, myComplex* u_modify, Size_F* size_F);

__host__ void deriv_x(myReal* c, const int n, const int B, const myReal L);
__host__ void get_dF(myComplex* dF, const myComplex* F, const myComplex* X, const myComplex* MR, const myReal* b, const myReal* G,
	const myReal* L, const myComplex* u, const myReal* const* CG, const Size_F* size_F, const Size_F* size_F_dev);

