#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#undef printf

__global__ void flip_shift(const myComplex* X, myComplex* X_ij, const int is, const int js, const Size_F* size_F)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if (i < size_F[0].const_2Bx && j < size_F[0].const_2Bx) {
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

		int X_ind = i + j*size_F[0].const_2Bx;
		int X_ij_ind = iout + jout*size_F[0].const_2Bx;

		for (int ip = 0; ip < 2; ip++)
			X_ij[X_ij_ind + ip*size_F[0].nx] = X[X_ind + ip*size_F[0].nx];
	}
}

__global__ void addup_F(myComplex* dF, const int nTot)
{
	int ind1 = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind1 < nTot) {
		int ind2 = ind1 + nTot;
		dF[ind1] = mycuCadd(dF[ind1], dF[ind2]);
	}
}

__global__ void add_F(myComplex* dF, const myComplex* dF_temp, const int nTot)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind < nTot)
		dF[ind] = mycuCadd(dF[ind], dF_temp[ind]);
}

__global__ void mulImg_FR(myComplex* dF, const myReal c, const int nR)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind < nR) {
		myReal y = dF[ind].y;
		dF[ind].y = dF[ind].x * c;
		dF[ind].x = -y * c;
	}
}

__global__ void add_FMR(myComplex* dF, const myComplex* FMR, const int ind_cumR, const Size_F* size_F)
{
	int ind_dF = ind_cumR + (threadIdx.x + threadIdx.y*size_F->const_2Bx)*size_F->nR_compact + blockIdx.x*size_F->nTot_compact;
	int ind_FMR = threadIdx.x + threadIdx.y*size_F->const_2Bx + blockIdx.x*size_F->nx;

	dF[ind_dF] = mycuCadd(dF[ind_dF], FMR[ind_FMR]);
}

__global__ void mul_fmR(myReal* f, const myReal* mR, const int dim, const Size_f* size_f)
{
	int ind_F = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind_F < size_f->nR) {
		int ind_mR = dim + ind_F*2;
		ind_F += blockIdx.y*size_f->nR;
		
		f[ind_F] = f[ind_F] * mR[ind_mR];
	}
}

__global__ void mulImg_FTot(myComplex* dF, const myReal* c, const int dim, const Size_F* size_F)
{
	int ind_R = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind_R < size_F->nR_compact) {
		unsigned int ij[2] = {blockIdx.y, blockIdx.z};
		int ind_dF = ind_R + (ij[0] + ij[1]*size_F->const_2Bx)*size_F->nR_compact;

		myReal y = dF[ind_dF].y;
		dF[ind_dF].y = dF[ind_dF].x * c[ij[dim]];
		dF[ind_dF].x = -y * c[ij[dim]];
	}
}

__global__ void get_c(myReal* c, const int i, const int j, const myReal* L, const myReal* G, const Size_F* size_F)
{
	if (i == j) {
		int ix = threadIdx.x;
		if (ix < size_F[0].Bx)
			c[ix] = -4*PI*PI * ix*ix * G[i+2*j] / (L[0]*L[0]);
		else
			c[ix] = -4*PI*PI * (ix-size_F[0].const_2Bx)*(ix-size_F[0].const_2Bx) * G[i+2*j] / (L[0]*L[0]);
	} else {
		int ix = threadIdx.x;
		int jx = blockIdx.x;

		myReal c1;
		if (ix < size_F[0].Bx)
			c1 = 2*PI * ix / L[0];
		else if (ix == size_F[0].Bx)
			c1 = 0;
		else
			c1 = 2*PI * (ix-size_F[0].const_2Bx) / L[0];

		myReal c2;
		if (jx < size_F[0].Bx)
			c2 = 2*PI * jx / L[0];
		else if (jx == size_F[0].Bx)
			c2 = 0;
		else
			c2 = 2*PI * (jx-size_F[0].const_2Bx) / L[0];

		int indc = ix + jx*size_F[0].const_2Bx;
		c[indc] = -c1*c2 * G[i+2*j];
	}
}

__global__ void get_biasRW(myComplex* dF_temp, const myComplex* Fold, const myReal* c, const int i, const int j, const Size_F* size_F)
{
	int indR = threadIdx.x + blockIdx.x*blockDim.x;
	if (indR < size_F[0].nR_compact) {
		unsigned int ij[2] = {blockIdx.y, blockIdx.z};

		int ind = indR + (ij[0] + ij[1]*size_F->const_2Bx)*size_F[0].nR_compact;

		if (i==j) {
			dF_temp[ind].x = Fold[ind].x * c[ij[i]];
			dF_temp[ind].y = Fold[ind].y * c[ij[i]];
		} else {
			int indc = ij[i] + ij[j]*size_F[0].const_2Bx;
			dF_temp[ind].x = Fold[ind].x * c[indc];
			dF_temp[ind].y = Fold[ind].y * c[indc];
		}
	}
}

__global__ void integrate_Fnew(myComplex* Fnew, const myComplex* Fold, const myComplex* dF, const myReal dt, const int nTot)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind < nTot)
	{
		Fnew[ind].x = Fold[ind].x + dt*dF[ind].x;
		Fnew[ind].y = Fold[ind].y + dt*dF[ind].y;
	}
}

__host__ void modify_F(const myComplex* F, myComplex* F_modify, bool reduce, Size_F* size_F)
{
	if (reduce) {
		int ind_F_reduced = 0;
		for (int j = 0; j < size_F[0].const_2Bx; j++) {
			for (int i = 0; i < size_F[0].const_2Bx; i++) {
				for (int l = 0; l <= size_F[0].lmax; l++) {
					for (int m = -l; m <= l; m++) {
						for (int n = -l; n <= l; n++) {
							int ind_F = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + 
								l*size_F[0].l_cum1 + i*size_F[0].l_cum2 + j*size_F[0].l_cum3;
							F_modify[ind_F_reduced] = F[ind_F];

							ind_F_reduced++;
						}
					}
				}
			}
		}
	} else {
		int ind_F_reduced = 0;
		for (int j = 0; j < size_F[0].const_2Bx; j++) {
			for (int i = 0; i < size_F[0].const_2Bx; i++) {
				for (int l = 0; l <= size_F[0].lmax; l++) {
					for (int m = -l; m <= l; m++) {
						for (int n = -l; n <= l; n++) {
							int ind_F = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + 
								l*size_F[0].l_cum1 + i*size_F[0].l_cum2 + j*size_F[0].l_cum3;
							F_modify[ind_F] = F[ind_F_reduced];

							ind_F_reduced++;
						}
					}
				}
			}
		}
	}
}

__host__ void modify_u(const myComplex* u, myComplex* u_modify, Size_F* size_F)
{
	int ind_u_reduced = 0;
	for (int ip = 0; ip < 2; ip++) {
		for (int l = 0; l <= size_F[0].lmax; l++) {
			for (int m = -l; m <= l; m++) {
				for (int n = -l; n <= l; n++) {
					int ind_u = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + l*size_F[0].l_cum1 + ip*size_F[0].l_cum2;
					u_modify[ind_u_reduced] = u[ind_u];

					ind_u_reduced++;
				}
			}
		}
	}
}

__host__ void deriv_x(myReal* c, const int n, const int B, const myReal L)
{
	if (n < B)
		*c = 2*PI*n/L;
	else if (n == B)
		*c = 0;
	else
		*c = 2*PI*(n-2*B)/L;
}

__host__ void get_dF(myComplex* dF, const myComplex* F, const myReal* f, const myComplex* X, const myReal* mR, const myReal* b, const myReal* G,
	const myReal* L, const myComplex* u, const myReal* dw_dev, const Size_F* size_F, const Size_F* size_F_dev, const Size_f* size_f, const Size_f* size_f_dev)
{
	////////////////////////////
	// circular_convolution X //
	////////////////////////////

	// X_ijk = flip(flip(flip(X,1),2),3)
	// X_ijk = circshift(X_ijk,1,i)
	// X_ijk = circshift(X_ijk,2,j)
	// X_ijk = circshift(X_ijk,3,k)
	// dF{r,i,j,k,p} = F{r,m,n,l}.*X_ijk{m,n,l,p}
	// dF(indmn,indmn,l,i,j,k,p) = -dF(indmn,indmn,l,i,j,k,p)*u(indmn,indmn,l,p)'
	// dF = sum(dF,'p')

	// set up arrays
	myComplex* F_dev;
	cudaErrorHandle(cudaMalloc(&F_dev, size_F->nTot_compact*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(F_dev, F, size_F->nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* X_dev;
	cudaErrorHandle(cudaMalloc(&X_dev, 2*size_F->nx*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(X_dev, X, 2*size_F->nx*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* X_ij_dev;
	cudaErrorHandle(cudaMalloc(&X_ij_dev, 2*size_F->nx*sizeof(myComplex)));

	myComplex* dF2_dev;
	cudaErrorHandle(cudaMalloc(&dF2_dev, 2*size_F->nTot_compact*sizeof(myComplex)));

	myComplex* dF2_temp_dev;
	cudaErrorHandle(cudaMalloc(&dF2_temp_dev, 2*size_F->nTot_compact*sizeof(myComplex)));

	myComplex* dF_temp_dev;
	cudaErrorHandle(cudaMalloc(&dF_temp_dev, 2*size_F->nR_compact*sizeof(myComplex)));

	myComplex* u_dev;
	cudaErrorHandle(cudaMalloc(&u_dev, 2*size_F->nR_compact*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(u_dev, u, 2*size_F->nR_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* dF_dev;
	cudaErrorHandle(cudaMalloc(&dF_dev, size_F->nTot_compact*sizeof(myComplex)));

	// set up cublas
	cublasHandle_t handle_cublas;
	cublasCreate(&handle_cublas);

	myComplex alpha_cublas = make_myComplex(1,0);
	myComplex beta_cublas = make_myComplex(0,0);

	// set up cutensor
	cutensorHandle_t handle_cutensor;
	cutensorInit(&handle_cutensor);

	cutensorContractionPlan_t plan_conv;
	size_t worksize_conv;

	cutensor_initConv(&handle_cutensor, &plan_conv, &worksize_conv, F_dev, X_ij_dev, dF_temp_dev, size_F);

	void* work = nullptr;
	if (worksize_conv > 0)
		cudaErrorHandle(cudaMalloc(&work, worksize_conv));

	myComplex alpha_cutensor = make_myComplex(0-(myReal)1/size_F->nx,0);
	myComplex beta_cutensor = make_myComplex(0,0);

	// set up blocksize and gridsize
	dim3 blocksize_16(16, 16, 1);
	int gridnum_16 = (int) size_F->const_2Bx/16 + 1;
	dim3 gridsize_16(gridnum_16, gridnum_16, 1);

	dim3 blocksize_512_nTot(512, 1, 1);
	dim3 gridsize_512_nTot((int)size_F->nTot_compact/512+1, 1, 1);

	// calculate
	for (int i = 0; i < size_F->const_2Bx; i++) {
		for (int j = 0; j < size_F->const_2Bx; j++) {
			flip_shift <<<gridsize_16, blocksize_16>>> (X_dev, X_ij_dev, i, j, size_F_dev);
			cudaErrorHandle(cudaGetLastError());

			cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv, (void*)&alpha_cutensor, F_dev, X_ij_dev,
				(void*)&beta_cutensor, dF_temp_dev, dF_temp_dev, work, worksize_conv, 0));

			for (int ip = 0; ip < 2; ip++) {
				myComplex* dF2_dev_ijn = dF2_dev + i*size_F->nR_compact + 
					j*(size_F->nR_compact*size_F->const_2Bx) + ip*size_F->nTot_compact;
				myComplex* dF_temp_dev_n = dF_temp_dev + ip*size_F->nR_compact;

				cudaErrorHandle(cudaMemcpy(dF2_dev_ijn, dF_temp_dev_n, size_F->nR_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));
			}
		}
	}

	for (int ip = 0; ip < 2; ip++) {
		for (int l = 0; l <= size_F->lmax; l++)
		{
			int ind_dF = l*(2*l-1)*(2*l+1)/3 + ip*size_F->nTot_compact;
			long long int stride_Fnew = size_F->nR_compact;

			int ind_u = l*(2*l-1)*(2*l+1)/3 + ip*size_F->nR_compact;
			long long int stride_u = 0;

			cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
				&alpha_cublas, dF2_dev+ind_dF, 2*l+1, stride_Fnew,
				u_dev+ind_u, 2*l+1, stride_u,
				&beta_cublas, dF2_temp_dev+ind_dF, 2*l+1, stride_Fnew, size_F->nx));
		}
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF2_temp_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	cudaErrorHandle(cudaMemcpy(dF_dev, dF2_temp_dev, size_F->nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));

	// free memory
	cudaErrorHandle(cudaFree(u_dev));
	cudaErrorHandle(cudaFree(dF2_temp_dev));

	//////////////////////////////
	// circular convolutions bX //
	//////////////////////////////

	// bX_ijk = flip(flip(flip(-b*X,1),2),3)
	// bX_ijk = circshift(bX_ijk,1,i)
	// bX_ijk = circshift(bX_ijk,2,j)
	// bX_ijk = circshift(bX_ijk,3,k)
	// dF{r,i,j,k,p} = Fold{r,m,n,l}.*bX_ijk{m,n,l,p}
	// dF{r,i,j,k,p} = dF{r,i,j,k,p}*c(p)
	// dF = sum(dF,'p')

	// set up blocksize and gridsize
	dim3 blocksize_512_nR(512, 1, 1);
	dim3 gridsize_512_nR((int)size_F->nR_compact/512+1, 1, 1);

	// calculate
	for (int i = 0; i < size_F->const_2Bx; i++) {
		for (int j = 0; j < size_F->const_2Bx; j++) {
			flip_shift <<<gridsize_16, blocksize_16>>> (X_dev, X_ij_dev, i, j, size_F_dev);
			cudaErrorHandle(cudaGetLastError());

			cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv, (void*)&alpha_cutensor, F_dev, X_ij_dev,
				(void*)&beta_cutensor, dF_temp_dev, dF_temp_dev, work, worksize_conv, 0));

			myReal c[2];
			deriv_x(c, i, size_F->Bx, *L);
			deriv_x(c+1, j, size_F->Bx, *L);

			mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev, -c[0]*b[0], size_F->nR_compact);
			cudaErrorHandle(cudaGetLastError());
			mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev+size_F->nR_compact, -c[1]*b[1], size_F->nR_compact);
			cudaErrorHandle(cudaGetLastError());

			for (int ip = 0; ip < 2; ip++) {
				myComplex* dF2_dev_ijp = dF2_dev + i*size_F->nR_compact + 
					j*(size_F->nR_compact*size_F->const_2Bx) + ip*size_F->nTot_compact;
				myComplex* dF_temp_dev_p = dF_temp_dev + ip*size_F->nR_compact;

				cudaErrorHandle(cudaMemcpy(dF2_dev_ijp, dF_temp_dev_p, size_F->nR_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));
			}
		}
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF2_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF2_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	// free memory
	cudaErrorHandle(cudaFree(X_dev));
	cudaErrorHandle(cudaFree(X_ij_dev));
	cudaErrorHandle(cudaFree(dF_temp_dev));
	if (worksize_conv > 0)
		cudaErrorHandle(cudaFree(work));

	/////////////////
	// multiply mR //
	/////////////////

	// set up arrays
	myReal* fmR_dev;
	cudaErrorHandle(cudaMalloc(&fmR_dev, size_f->nTot*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(fmR_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

	myReal* mR_dev;
	cudaErrorHandle(cudaMalloc(&mR_dev, 2*size_f->nR*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(mR_dev, mR, 2*size_f->nR*sizeof(myReal), cudaMemcpyHostToDevice));

	// get c
	myReal* c = new myReal[size_F->const_2Bx];
	for (int i = 0; i < size_F->const_2Bx; i++) {
		deriv_x(&c[i], i, size_F->Bx, *L);
		c[i] = -c[i];
	}

	myReal* c_dev;
	cudaErrorHandle(cudaMalloc(&c_dev, size_F->const_2Bx*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(c_dev, c, size_F->const_2Bx*sizeof(myReal), cudaMemcpyHostToDevice));

	// set up blocksize and gridsize
	dim3 blocksize_512_nRf_nx(512, 1, 1);
	dim3 gridsize_512_nRf_nx((int)size_f->nR/512+1, size_f->nx, 1);

	dim3 blocksize_deriv(512,1,1);
	dim3 gridsize_deriv((int)size_F->nR_compact/512+1, size_F->const_2Bx, size_F->const_2Bx);

	// calculate
	mul_fmR <<<gridsize_512_nRf_nx, blocksize_512_nRf_nx>>> (fmR_dev, mR_dev, 0, size_f_dev);
	fftSO3R_forward(dF2_dev, fmR_dev, dw_dev, size_F, size_F_dev, size_f, size_f_dev);

	cudaErrorHandle(cudaMemcpy(fmR_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));
	mul_fmR <<<gridsize_512_nRf_nx, blocksize_512_nRf_nx>>> (fmR_dev, mR_dev, 1, size_f_dev);
	fftSO3R_forward(dF2_dev+size_F->nTot_compact, fmR_dev, dw_dev, size_F, size_F_dev, size_f, size_f_dev);

	for (int ip = 0; ip < 2; ip++) {
		mulImg_FTot <<<gridsize_deriv, blocksize_deriv>>> (dF2_dev+ip*size_F->nTot_compact, c_dev, ip, size_F_dev);
		cudaErrorHandle(cudaGetLastError());
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF2_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF2_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	// free memory
	cudaErrorHandle(cudaFree(fmR_dev));
	cudaErrorHandle(cudaFree(mR_dev));
	cudaErrorHandle(cudaFree(c_dev));

	delete[] c;

	///////////////////////
	// random walk noise //
	///////////////////////

	// set up arrays
	cudaErrorHandle(cudaMalloc(&c_dev, size_F->nx*sizeof(myReal)));

	myReal* G_dev;
	cudaErrorHandle(cudaMalloc(&G_dev, 9*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(G_dev, G, 9*sizeof(myReal), cudaMemcpyHostToDevice));

	myReal* L_dev;
	cudaErrorHandle(cudaMalloc(&L_dev, sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(L_dev, L, sizeof(myReal), cudaMemcpyHostToDevice));

	// calculate
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			if (i == j) {
				get_c <<<1, size_F->const_2Bx>>> (c_dev, i, j, L_dev, G_dev, size_F_dev);
				cudaErrorHandle(cudaGetLastError());
			}
			else {
				get_c <<<size_F->const_2Bx, size_F->const_2Bx>>> (c_dev, i, j, L_dev, G_dev, size_F_dev);
				cudaErrorHandle(cudaGetLastError());
			}

			get_biasRW <<<gridsize_deriv, blocksize_deriv>>> (dF2_dev, F_dev, c_dev, i, j, size_F_dev);
			cudaErrorHandle(cudaGetLastError());

			add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF2_dev, size_F->nTot_compact);
			cudaErrorHandle(cudaGetLastError());
		}
	}

	// free memory
	cudaErrorHandle(cudaFree(c_dev));
	cudaErrorHandle(cudaFree(G_dev));
	cudaErrorHandle(cudaFree(L_dev));
	cudaErrorHandle(cudaFree(F_dev));
	cudaErrorHandle(cudaFree(dF2_dev));

	// return
	cudaErrorHandle(cudaMemcpy(dF, dF_dev, size_F->nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));

	cudaErrorHandle(cudaFree(dF_dev));
}

