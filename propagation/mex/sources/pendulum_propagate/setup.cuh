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

__host__ void modify_F(const cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce, Size_F* size_F);
__host__ void modify_u(const cuDoubleComplex* u, cuDoubleComplex* u_modify, Size_F* size_F);

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);
__host__ void cublasErrorHandle(const cublasStatus_t& err);

__host__ void cutensor_initConv(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const cuDoubleComplex* Fold_dev, const cuDoubleComplex* X_ijk_dev, const cuDoubleComplex* dF_temp_dev, const Size_F* size_F);
__host__ void cutensor_initFMR(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const cuDoubleComplex* Fold_dev, const cuDoubleComplex* MR_dev, const cuDoubleComplex* FMR_dev, const int l, const Size_F* size_F);

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx);
