#ifndef SETUP
#define SETUP

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>
#include <cufft.h>

#include "mex.h"

#define FP32 false
#if FP32
	typedef float myReal;
	typedef cuComplex myComplex;

	#define mycutensor_datatype CUDA_C_32F
	#define mycutensor_computetype CUTENSOR_COMPUTE_32F
	#define mycublasgemmStridedBatched cublasCgemm3mStridedBatched
	#define mycuCadd cuCaddf
	#define make_myComplex make_cuComplex

	#define mymxGetComplex mxGetComplexSingles
	#define mymxGetReal mxGetSingles
	#define mymxRealClass mxSINGLE_CLASS
#else
	typedef double myReal;
	typedef cufftDoubleReal myfftReal;
	typedef cuDoubleComplex myComplex;
	typedef cufftDoubleComplex myfftComplex;

	#define myfftForwardType_R CUFFT_D2Z
	#define myfftForwardType_x CUFFT_Z2Z
	#define myfftBackwardType_R CUFFT_Z2D
	#define myfftBackwardType_x CUFFT_Z2Z
	#define myfftForwardExec_R cufftExecD2Z
	#define myfftForwardExec_x cufftExecZ2Z
	#define myfftBackwardExec_R cufftExecZ2D
	#define myfftBackwardExec_x cufftExecZ2Z

	#define mycutensor_Complextype CUDA_C_64F
	#define mycutensor_Realtype CUDA_R_64F
	#define mycutensor_computetype CUTENSOR_COMPUTE_64F
	#define mycublasgemmStridedBatched cublasZgemmStridedBatched
	#define mycuCadd cuCadd
	#define make_myComplex make_cuDoubleComplex

	#define mymxGetComplex mxGetComplexDoubles
	#define mymxGetReal mxGetDoubles
	#define mymxRealClass mxDOUBLE_CLASS
#endif

constexpr myReal PI = 3.141592653589793;

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
	int const_2lp1;
	int const_lp1;
	int const_2lp1s;

	int l_cum0;
	int l_cum1;
	int l_cum2;
	int l_cum3;
};

struct Size_f {
	int BR;
	int Bx;

	int nR;
	int nx;
	int nTot;

	int const_2Bx;
	int const_2BR;
	int const_2Bxs;
	int const_2BRs;
};

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);
__host__ void cublasErrorHandle(const cublasStatus_t& err);
__host__ void cufftErrorHandle(const cufftResult_t& err);

__host__ void cutensor_initConv(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const void* Fold_dev, const void* X_dev, const void* dF_dev, const Size_F* size_F);

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx);
__host__ void init_Size_f(Size_f* size_f, const int BR, const int Bx);

#endif // !setup

