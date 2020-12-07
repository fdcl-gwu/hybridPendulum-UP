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

__global__ void addup_F(cuDoubleComplex* Fnew, cuDoubleComplex* Fold, double dt, Size_F* size_F)
{
	int ind1 = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind1 < size_F[0].nR_compact) {
		ind1 += blockIdx.y*size_F[0].nR_compact;

		int ind2 = ind1 + size_F[0].nTot_compact;
		int ind3 = ind2 + size_F[0].nTot_compact;

		Fnew[ind1] = cuCadd(Fnew[ind1], Fnew[ind2]);
		Fnew[ind1] = cuCadd(Fnew[ind1], Fnew[ind3]);

		Fnew[ind1].x = Fold[ind1].x + dt*Fnew[ind1].x;
		Fnew[ind1].y = Fold[ind1].y + dt*Fnew[ind1].y;
	}
}

__global__ void add(cuDoubleComplex*F1, cuDoubleComplex* F2, Size_F* size_F)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind < size_F[0].nTot_compact)
		F1[ind] = cuCadd(F1[ind], F2[ind]);
}

__global__ void get_c(double* c, int i, int j, double* G, double dt, Size_F* size_F)
{
	if (i == j) {
		int ix = threadIdx.x;
		if (ix < size_F[0].Bx)
			c[ix] = -PI*PI * ix*ix * G[i+3*j] * dt;
		else
			c[ix] = -PI*PI * (ix-size_F[0].const_2Bx)*(ix-size_F[0].const_2Bx) * G[i+3*j] * dt;
	} else {
		int ix = threadIdx.x;
		int jx = threadIdx.y;

		double c1;
		if (ix < size_F[0].Bx)
			c1 = PI * ix;
		else if (ix == size_F[0].Bx)
			c1 = 0;
		else
			c1 = PI * (ix-size_F[0].const_2Bx);

		double c2;
		if (jx < size_F[0].Bx)
			c2 = PI * jx;
		else if (jx == size_F[0].Bx)
			c2 = 0;
		else
			c2 = PI * (jx-size_F[0].const_2Bx);

		int indc = ix + jx*size_F[0].const_2Bx;
		c[indc] = -c1*c2*G[i+3*j]*dt;
	}
}

__global__ void add_biasRW(cuDoubleComplex* dF, cuDoubleComplex* Fold, double* c, int i, int j, Size_F* size_F)
{
	int indR = threadIdx.x + blockIdx.x*blockDim.x;
	if (indR < size_F[0].nR_compact) {
		unsigned int indx = blockIdx.y;
		int ijk[3];

		ijk[2] = (int) indx / size_F[0].const_2Bxs;
		int ijx = indx % size_F[0].const_2Bxs;
		ijk[1] = (int) ijx / size_F[0].const_2Bx;
		ijk[0] = ijx % size_F[0].const_2Bx;

		int ind = indR + indx*size_F[0].nR_compact;

		if (i==j) {
			dF[ind].x = Fold[ind].x * c[ijk[i]];
			dF[ind].y = Fold[ind].y * c[ijk[i]];
		} else {
			int indc = ijk[i] + ijk[j]*size_F[0].const_2Bx;
			dF[ind].x = Fold[ind].x * c[indc];
			dF[ind].y = Fold[ind].y * c[indc];
		}
	}
}

__host__ void modify_F(cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce,Size_F* size_F)
{
	if (reduce) {
		int ind_F_reduced = 0;
		for (int k = 0; k < size_F[0].const_2Bx; k++) {
			for (int j = 0; j < size_F[0].const_2Bx; j++) {
				for (int i = 0; i < size_F[0].const_2Bx; i++) {
					for (int l = 0; l <= size_F[0].lmax; l++) {
						for (int m = -l; m <= l; m++) {
							for (int n = -l; n <= l; n++) {
								int ind_F = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + 
									l*size_F[0].l_cum1 + i*size_F[0].l_cum2 + j*size_F[0].l_cum3 + k*size_F[0].l_cum4;
								F_modify[ind_F_reduced] = F[ind_F];

								ind_F_reduced++;
							}
						}
					}
				}
			}
		}
	} else {
		int ind_F_reduced = 0;
		for (int k = 0; k < size_F[0].const_2Bx; k++) {
			for (int j = 0; j < size_F[0].const_2Bx; j++) {
				for (int i = 0; i < size_F[0].const_2Bx; i++) {
					for (int l = 0; l <= size_F[0].lmax; l++) {
						for (int m = -l; m <= l; m++) {
							for (int n = -l; n <= l; n++) {
								int ind_F = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + 
									l*size_F[0].l_cum1 + i*size_F[0].l_cum2 + j*size_F[0].l_cum3 + k*size_F[0].l_cum4;
								F_modify[ind_F] = F[ind_F_reduced];

								ind_F_reduced++;
							}
						}
					}
				}
			}
		}
	}
}

__host__ void modify_u(cuDoubleComplex* u, cuDoubleComplex* u_modify, Size_F* size_F)
{
	int ind_u_reduced = 0;
	for (int i = 0; i < 3; i++) {
		for (int l = 0; l <= size_F[0].lmax; l++) {
			for (int m = -l; m <= l; m++) {
				for (int n = -l; n <= l; n++) {
					int ind_u = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + l*size_F[0].l_cum1 + i*size_F[0].l_cum2;
					u_modify[ind_u_reduced] = u[ind_u];

					ind_u_reduced++;
				}
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

__host__ void cublasErrorHandle(const cublasStatus_t& err)
{
	if (err != CUBLAS_STATUS_SUCCESS) {
		std::cout << "cuBlas Error: " << err << std::endl;
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

	size_F->nR_compact = (size_F->lmax+1) * (2*size_F->lmax+1) * (2*size_F->lmax+3) / 3;
	size_F->nTot_compact = size_F->nR_compact * size_F->nx;

	size_F->const_2Bx = 2*Bx;
	size_F->const_2Bxs = (2*Bx) * (2*Bx);
	size_F->const_2lp1 = 2*size_F->lmax+1;
	size_F->const_lp1 = size_F->lmax+1;
	size_F->const_2lp1s = (2*size_F->lmax+1) * (2*size_F->lmax+1);

	size_F->l_cum0 = size_F->const_2lp1;
	size_F->l_cum1 = size_F->l_cum0*size_F->const_2lp1;
	size_F->l_cum2 = size_F->l_cum1*size_F->const_lp1;
	size_F->l_cum3 = size_F->l_cum2*size_F->const_2Bx;
	size_F->l_cum4 = size_F->l_cum3*size_F->const_2Bx;
}

