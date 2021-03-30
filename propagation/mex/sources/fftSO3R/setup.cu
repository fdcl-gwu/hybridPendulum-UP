#include "setup.cuh"

__host__ void cudaErrorHandle(const cudaError_t& err)
{
	if (err != cudaSuccess) {
		printf("cuda Error: ");
		printf(cudaGetErrorString(err));
		printf("\n");
	}
}

__host__ void cufftErrorHandle(const cufftResult_t& err)
{
	if (err != CUFFT_SUCCESS) {
		printf("cufft Error %i\n", err);
	}
}

__host__ void cutensorErrorHandle(const cutensorStatus_t& err)
{
	if (err != CUTENSOR_STATUS_SUCCESS) {
		printf("cuTensor Error: ");
		printf(cutensorGetErrorString(err));
		printf("\n");
	}
}

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx, int ndims)
{
	size_F->BR = BR;
	size_F->Bx = Bx;
	size_F->lmax = BR-1;

	size_F->nR = (2*size_F->lmax+1) * (2*size_F->lmax+1) * (size_F->lmax+1);
	if (ndims == 5) {
		size_F->nx = (2*Bx) * (2*Bx);
	} else {
		size_F->nx = (2*Bx) * (2*Bx) * (2*Bx);
	}
	
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

__host__ void init_Size_f(Size_f* size_f, const int BR, const int Bx, const int ndims)
{
	size_f->BR = BR;
	size_f->Bx = Bx;

	size_f->nR = (2*BR) * (2*BR) * (2*BR);
	if (ndims == 5) {
		size_f->nx = (2*Bx) * (2*Bx);
	} else {
		size_f->nx = (2*Bx) * (2*Bx) * (2*Bx);
	}
	size_f->nTot = size_f->nR * size_f->nx;

	size_f->const_2BR = 2*BR;
	size_f->const_2Bx = 2*Bx;
	size_f->const_2BRs = (2*BR) * (2*Bx);
	size_f->const_2Bxs = (2*Bx) * (2*Bx);
}

__global__ void supplement_R(myComplex* F, const Size_f* size_f)
{
	unsigned int i = threadIdx.x + 1 + size_f->BR;
	unsigned int j = blockIdx.x;

	unsigned int indf_t = i + blockIdx.y*size_f->const_2BR + j*size_f->const_2BRs + blockIdx.z*size_f->nR;
	unsigned int indf_s;
	if (j == 0) {
		indf_s = size_f->const_2BR-i + blockIdx.y*size_f->const_2BR + j*size_f->const_2BRs + blockIdx.z*size_f->nR;
	} else {
		indf_s = size_f->const_2BR-i + blockIdx.y*size_f->const_2BR + (size_f->const_2BR-j)*size_f->const_2BRs + blockIdx.z*size_f->nR;
	}
	
	F[indf_t].x = F[indf_s].x;
	F[indf_t].y = -F[indf_s].y;
}

__global__ void shiftflip(myComplex* F1, const int dim, const Size_f* size_f)
{
	extern __shared__ myComplex F1_temp[];

	unsigned int ind;
	if (dim == 1) {
		ind = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->nR;
	} else if (dim == 3) {
		ind = blockIdx.x + threadIdx.x*size_f->const_2BRs + blockIdx.y*size_f->nR;
	}

	F1_temp[threadIdx.x] = F1[ind];
	__syncthreads();

	unsigned int ind_sf;
	unsigned int ind_sf_dim;
	if (threadIdx.x < size_f->BR) {
		ind_sf_dim = size_f->BR - threadIdx.x - 1;
	} else {
		ind_sf_dim = 3*size_f->BR - threadIdx.x - 1;
	}

	if (dim == 1) {
		ind_sf = ind_sf_dim + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->nR;
	} else if (dim == 3) {
		ind_sf = blockIdx.x + ind_sf_dim*size_f->const_2BRs + blockIdx.y*size_f->nR;
	}
	F1[ind_sf] = F1_temp[threadIdx.x];
}

__global__ void mul_dw(myReal* dw, const myReal* d, const myReal* w, const Size_F* size_F)
{
	unsigned int ind_d = threadIdx.x + blockIdx.x*size_F->const_2lp1 + blockIdx.y*size_F->const_2lp1s + blockIdx.z*size_F->nR;
	unsigned int ind_w = blockIdx.z;

	dw[ind_d] = d[ind_d] * w[ind_w];
}

__global__ void mul_dl(myReal* dl, const myReal* d, const Size_F* size_F)
{
	unsigned int ind_d = threadIdx.x + blockIdx.x*size_F->const_2lp1 + blockIdx.y*size_F->const_2lp1s + blockIdx.z*size_F->nR;
	dl[ind_d] = d[ind_d] * (2*blockIdx.y+1);
}
