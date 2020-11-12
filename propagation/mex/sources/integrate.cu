#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

__global__ void flip_shift(cuDoubleComplex* X, cuDoubleComplex* X_ijk, int is, int js, int ks)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (i < const_2B && j < const_2B && k < const_2B) {
		int iout = is-i;
		if (iout < 0)
			iout += const_2B;
		else if (iout >= const_2B)
			iout -= const_2B;

		int jout = js-j;
		if (jout < 0)
			jout += const_2B;
		else if (jout >= const_2B)
			jout -= const_2B;

		int kout = ks-k;
		if (kout < 0)
			kout += const_2B;
		else if (kout >= const_2B)
			kout -= const_2B;

		int X_ind = i + j*const_2B + k*const_4Bs;
		int X_ijk_ind = iout + jout*const_2B + kout*const_4Bs;

		for (int m = 0; m < 3; m++)
			X_ijk[X_ijk_ind + m*nx] = X[X_ind + m*nx];
	}
}

__global__ void derivate(cuDoubleComplex* temp, cuDoubleComplex* u, cuDoubleComplex* Fnew, int i, int j, int k)
{
	int l = blockIdx.x;
	int m_p_lmax = threadIdx.x;
	int n_p_lmax = threadIdx.y;

	long ind_Fnew = m_p_lmax + n_p_lmax*(2*lmax+1) + l*(2*lmax+1)*(2*lmax+1) + (i + j*const_2B + k*const_4Bs)*nR;

	if (m_p_lmax-lmax >= -l && m_p_lmax-lmax <= l && n_p_lmax-lmax >= -l && n_p_lmax-lmax <= l) {
		for (int ii = -l+lmax; ii <= l+lmax; ii++) {
			for (int p = 0; p < 3; p++) {
				int ind_temp = m_p_lmax + ii*(2*lmax+1) + l*(2*lmax+1)*(2*lmax+1) + p*nR;
				int ind_u = n_p_lmax + ii*(2*lmax+1) + l*(2*lmax+1)*(2*lmax+1) + p*nR;

				Fnew[ind_Fnew] = cuCsub(Fnew[ind_Fnew], cuCmul(temp[ind_temp], u[ind_u]));
			}
		}
	}
}

__host__ void cudaErrorHandle(const cudaError_t& err)
{
	if (err != cudaSuccess) {
		std::cout << "Cuda Error: " << cudaGetErrorString(err) << std::endl;
	}
}

__host__ void cutensorErrorHandle(const cutensorStatus_t& err)
{
	if (err != CUTENSOR_STATUS_SUCCESS) {
		std::cout << "cuTensor Error: " << cutensorGetErrorString(err) << std::endl;
	}
}

