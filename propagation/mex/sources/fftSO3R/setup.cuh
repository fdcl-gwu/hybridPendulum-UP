#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuComplex.h"
#include "cufft.h"
#include "cutensor.h"

#include <stdio.h>

// only works with FP64, cutensor won't work with this setting for FP32
#define FP32 false
#if FP32
	typedef float myReal;
	typedef cufftReal myfftReal;
	typedef cuComplex myComplex;
	typedef cufftComplex myfftComplex;

	#define myfftForwardType_R CUFFT_R2C
	#define myfftForwardType_x CUFFT_C2C
	#define myfftBackwardType_R CUFFT_C2R
	#define myfftBackwardType_x CUFFT_C2C
	#define myfftForwardExec_R cufftExecR2C
	#define myfftForwardExec_x cufftExecC2C
	#define myfftBackwardExec_R cufftExecC2R
	#define myfftBackwardExec_x cufftExecC2C

	#define mycutensor_Complextype CUDA_C_32F
	#define mycutensor_Realtype CUDA_R_32F
	#define mycutensor_computetype CUTENSOR_COMPUTE_32F
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
	#define make_myComplex make_cuDoubleComplex

	#define mymxGetComplex mxGetComplexDoubles
	#define mymxGetReal mxGetDoubles
	#define mymxRealClass mxDOUBLE_CLASS
#endif

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
__host__ void cufftErrorHandle(const cufftResult_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);

__host__ void init_Size_F(Size_F* size_F, const int BR, const int Bx, const int ndims);
__host__ void init_Size_f(Size_f* size_f, const int BR, const int Bx, const int ndims);

__global__ void supplement_R(myComplex* F, const Size_f* size_f);
__global__ void shiftflip(myComplex* F1, const int dim, const Size_f* size_f);
__global__ void mul_dw(myReal* dw, const myReal* d, const myReal* w, const Size_F* size_F);
__global__ void mul_dl(myReal* dl, const myReal* d, const Size_F* size_F);

