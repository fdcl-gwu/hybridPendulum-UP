#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>

constexpr double PI = 3.141592653589793;

struct Size_F {
	int BR;
	int Bx;
	int lmax;

	int nR;
	int nx;
	int nTot;
	int nR_compact;
	int nTot_compact;

	int const_2Bx;
	int const_2Bxs;
	int const_2lp1;
	int const_lp1;
	int const_2lp1s;

	int l_cum0;
	int l_cum1;
	int l_cum2;
	int l_cum3;
	int l_cum4;
};

__global__ void flip_shift(const cuDoubleComplex* X, cuDoubleComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F);
__global__ void addup_F(cuDoubleComplex* dF, Size_F* size_F);
__global__ void add_F(cuDoubleComplex* dF, const cuDoubleComplex* F, const Size_F* size_F);
__global__ void get_c(double* c, const int i, const int j, const double* G, const Size_F* size_F);
__global__ void get_biasRW(cuDoubleComplex* dF_temp, const cuDoubleComplex* Fold, const double* c, const int i, const int j, const Size_F* size_F);
__global__ void integrate_Fnew(cuDoubleComplex* Fnew, const cuDoubleComplex* Fold, const cuDoubleComplex* dF, const double dt, const Size_F* size_F);

__host__ void modify_F(cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce, Size_F* size_F);
__host__ void modify_u(cuDoubleComplex* u, cuDoubleComplex* u_modify, Size_F* size_F);

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);
__host__ void cublasErrorHandle(const cublasStatus_t& err);

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx);
