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

	// split settings
	int ns;
	int nR_split;
	int nR_remainder;

	int nTot_splitx;
	int nTot_splitx2;
};

__global__ void flip_shift(const cuDoubleComplex* X, cuDoubleComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F);
__global__ void addup_F(cuDoubleComplex* dF, Size_F* size_F);
__global__ void add_F(cuDoubleComplex* dF, const cuDoubleComplex* F, const Size_F* size_F);
__global__ void mulImg_FR(cuDoubleComplex* dF, const double c, const int nR, const Size_F* size_F);
__global__ void kron_FMR(double* FMR_real, double* FMR_imag, const cuDoubleComplex* F, const cuDoubleComplex* MR,
	const int ind_F_cumR, const int ind_MR_cumR, const bool iswrapped, const int c2l1p1, const int c2l2p1, const Size_F* size_F);
__global__ void add_FMR(cuDoubleComplex* dF, const double* FMR_real, const double* FMR_imag, const int ind_cumR, const int c2lp1s, const Size_F* size_F);
__global__ void mulImg_FTot(cuDoubleComplex* dF, const double* c, const int dim, const int k, const Size_F* size_F);
__global__ void integrate_Fnew(cuDoubleComplex* Fnew, const cuDoubleComplex* Fold, const cuDoubleComplex* dF, const double dt, const Size_F* size_F);

__host__ void modify_F(const cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce, Size_F* size_F);
__host__ void permute_F(cuDoubleComplex* F, bool R_first, Size_F* size_F);
__host__ void modify_u(const cuDoubleComplex* u, cuDoubleComplex* u_modify, Size_F* size_F);

__host__ void deriv_x(double* c, const int n, const int B, const double L);
__host__ void get_blocksize(dim3* gridsize, dim3* blocksize, bool* iswrapped, const int l1, const int l2, const Size_F* size_F);

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);
__host__ void cublasErrorHandle(const cublasStatus_t& err);

__host__ void cutensor_initialize(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	void* Fold_dev, void* X_dev, void* dF_dev, const int nR_split, const Size_F* size_F);

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx);
