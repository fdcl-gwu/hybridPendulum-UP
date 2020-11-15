#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

__global__ void flip_shift(cuDoubleComplex* X, cuDoubleComplex* X_ijk, int is, int js, int ks, Size_F* size_F)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int k = blockIdx.z*blockDim.z + threadIdx.z;

	if (i < size_F[0].const_2Bx && j < size_F[0].const_2Bx && k < size_F[0].const_2Bx) {
		int iout = is-i;
		if (iout < 0)
			iout += size_F[0].const_2Bx;
		else if (iout >= size_F[0].const_2Bx)
			iout -= size_F[0].const_2Bx;

		int jout = js-j;
		if (jout < 0)
			jout += size_F[0].const_2Bx;
		else if (jout >= size_F[0].const_2Bx)
			jout -= size_F[0].const_2Bx;

		int kout = ks-k;
		if (kout < 0)
			kout += size_F[0].const_2Bx;
		else if (kout >= size_F[0].const_2Bx)
			kout -= size_F[0].const_2Bx;

		int X_ind = i + j*size_F[0].const_2Bx + k*size_F[0].const_2Bxs;
		int X_ijk_ind = iout + jout*size_F[0].const_2Bx + kout*size_F[0].const_2Bxs;

		for (int m = 0; m < 3; m++)
			X_ijk[X_ijk_ind + m*size_F[0].nx] = X[X_ind + m*size_F[0].nx];
	}
}

__global__ void derivate(cuDoubleComplex* temp, cuDoubleComplex* u, cuDoubleComplex* Fnew, int i, int j, int k, Size_F* size_F)
{
	int l = blockIdx.x;
	int m_p_lmax = threadIdx.x;
	int n_p_lmax = threadIdx.y;

	long ind_Fnew = m_p_lmax + n_p_lmax*size_F[0].const_2lp1 + l*size_F[0].const_2lp1s
		+ (i + j*size_F[0].const_2Bx + k*size_F[0].const_2Bxs)*size_F[0].nR;

	if (m_p_lmax-size_F[0].lmax >= -l && m_p_lmax-size_F[0].lmax <= l && n_p_lmax-size_F[0].lmax >= -l && n_p_lmax-size_F[0].lmax <= l) {
		for (int ii = -l+size_F[0].lmax; ii <= l+size_F[0].lmax; ii++) {
			for (int p = 0; p < 3; p++) {
				int ind_temp = m_p_lmax + ii*size_F[0].const_2lp1 + l*size_F[0].const_2lp1s + p*size_F[0].nR;
				int ind_u = n_p_lmax + ii*size_F[0].const_2lp1 + l*size_F[0].const_2lp1s + p*size_F[0].nR;

				Fnew[ind_Fnew] = cuCsub(Fnew[ind_Fnew], cuCmul(temp[ind_temp], u[ind_u]));
			}
		}
	}
}

__global__ void add_dF(cuDoubleComplex* Fnew, cuDoubleComplex* Fold, double dt, Size_F* size_F)
{
	long i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < size_F[0].nTot) {
		Fnew[i].x = Fold[i].x + dt*Fnew[i].x;
		Fnew[i].y = Fold[i].y + dt*Fnew[i].y;
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

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx)
{
	size_F->BR = BR;
	size_F->Bx = Bx;
	size_F->lmax = BR-1;

	size_F->nR = (2*size_F->lmax+1) * (2*size_F->lmax+1) * (size_F->lmax+1);
	size_F->nx = (2*Bx) * (2*Bx) * (2*Bx);
	size_F->nTot = size_F->nR * size_F->nx;

	size_F->const_2Bx = 2*Bx;
	size_F->const_2Bxs = (2*Bx) * (2*Bx);
	size_F->const_2lp1 = 2*size_F->lmax+1;
	size_F->const_lp1 = size_F->lmax+1;
	size_F->const_2lp1s = (2*size_F->lmax+1) * (2*size_F->lmax+1);
}

