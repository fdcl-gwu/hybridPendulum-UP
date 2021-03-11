#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>
#include <cublas_v2.h>

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
	typedef cuDoubleComplex myComplex;

	#define mycutensor_datatype CUDA_C_64F
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

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);
__host__ void cublasErrorHandle(const cublasStatus_t& err);

__host__ void cutensor_initConv(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const void* Fold_dev, const void* X_dev, const void* dF_dev, const Size_F* size_F);
__host__ void cutensor_initFMR(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const void* Fold_dev, const void* MR_dev, const void* FMR_dev, const int l, const Size_F* size_F);

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx);

