#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuComplex.h>
#include <cutensor.h>

struct Size_F {
	int BR;
	int Bx;
	int lmax;

	int nR;
	int nx;
	int nTot;

	int const_2Bx;
	int const_2Bxs;
	int const_2lp1;
	int const_lp1;
	int const_2lp1s;
};

__global__ void flip_shift(cuDoubleComplex* X, cuDoubleComplex* X_ijk, int is, int js, int ks, Size_F* size_F);
__global__ void derivate(cuDoubleComplex* temp, cuDoubleComplex* u, cuDoubleComplex* Fnew, int i, int j, int k, Size_F* size_F);
__global__ void add_dF(cuDoubleComplex* Fnew, cuDoubleComplex* Fold, double dt, Size_F* size_F);

__host__ void cudaErrorHandle(const cudaError_t& err);
__host__ void cutensorErrorHandle(const cutensorStatus_t& err);

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx);
