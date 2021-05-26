#ifndef SETUP
#define SETUP

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cublas_v2.h"
#include "cutensor.h"

#define FP32 false
#if FP32

#else
    typedef double myReal;
       
    #define mymxGetReal mxGetDoubles
    #define mymxRealClass mxDOUBLE_CLASS

    #define mysqrt sqrt
    #define myexp exp
    #define myasin asin
    #define mysin sin
    #define mycos cos
    #define mytan tan

    #define mycudaRealType CUDA_R_64F
    #define mycutensor_computetype CUTENSOR_COMPUTE_64F
    
    #define mycublasdot cublasDdot
#endif

constexpr myReal PI = 3.141592653589793;

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

void cudaErrorHandle(const cudaError_t& err);
void cublasErrorHandle(const cublasStatus_t& err);
void cutensorErrorHandle(const cutensorStatus_t& err);

void init_Size_f(Size_f* size_f, const int BR, const int Bx);

#endif // !setup
