#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>

constexpr int B = 10;
constexpr int lmax = B-1;

constexpr int nR = (2*lmax+1) * (2*lmax+1) * (lmax+1);
constexpr int nx = (2*B) * (2*B) * (2*B);
constexpr int nTot = nR * nx;

constexpr int const_2B = 2*B;
constexpr int const_4Bs = (2*B) * (2*B);

__global__ void flip_shift(cuDoubleComplex* X, cuDoubleComplex* X_ijk, int is, int js, int ks);
__global__ void derivate(cuDoubleComplex* temp, cuDoubleComplex* u, cuDoubleComplex* Fnew, int i, int j, int k);
__global__ void add_dF(cuDoubleComplex* Fnew, cuDoubleComplex* Fold, double dt);

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);
