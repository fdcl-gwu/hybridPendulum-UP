#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

__global__ void flip_shift(const myComplex* X, myComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F)
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

__global__ void addup_F(myComplex* dF, const int nTot)
{
	int ind1 = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind1 < nTot) {
		int ind2 = ind1 + nTot;
		int ind3 = ind2 + nTot;

		dF[ind1] = mycuCadd(dF[ind1], dF[ind2]);
		dF[ind1] = mycuCadd(dF[ind1], dF[ind3]);
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

__global__ void add_FMR_small(myComplex* dF, const myComplex* FMR, const int ind_cumR, const Size_F* size_F)
{
	int ind_dF = ind_cumR + (threadIdx.x + threadIdx.y*size_F->const_2Bx + blockIdx.x*size_F->const_2Bxs)*size_F->nR_compact + blockIdx.y*size_F->nTot_compact;
	int ind_FMR = threadIdx.x + threadIdx.y*size_F->const_2Bx + blockIdx.x*size_F->const_2Bxs + blockIdx.y*size_F->nx;

	dF[ind_dF] = mycuCadd(dF[ind_dF], FMR[ind_FMR]);
}

__global__ void add_FMR_large(myComplex* dF, const myComplex* FMR, const int ind_cumR, const Size_F* size_F)
{
	int ind_dF = ind_cumR + (threadIdx.x + blockIdx.x*size_F->const_2Bx)*size_F->nR_compact + threadIdx.y*size_F->nTot_splitx;
	int ind_FMR = threadIdx.x + blockIdx.x*size_F->const_2Bx + threadIdx.y*size_F->const_2Bxs;

	dF[ind_dF] = mycuCadd(dF[ind_dF], FMR[ind_FMR]);
}

__global__ void mulImg_FTot_small(myComplex* dF, const myReal* c, const int dim, const Size_F* size_F)
{
	int ind_R = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind_R < size_F->nR_compact) {
		int ijk[3] = {};
		ijk[0] = blockIdx.y;

		if (dim != 0) {
			ijk[2] = (int) blockIdx.z / size_F->const_2Bx;
			ijk[1] = blockIdx.z % size_F->const_2Bx;
		}

		int ind_dF = ind_R + (ijk[0] + blockIdx.z*size_F->const_2Bx)*size_F->nR_compact;

		myReal y = dF[ind_dF].y;
		dF[ind_dF].y = dF[ind_dF].x * c[ijk[dim]];
		dF[ind_dF].x = -y * c[ijk[dim]];
	}
}

__global__ void mulImg_FTot_large(myComplex* dF, const myReal* c, const int dim, const int k, const Size_F* size_F)
{
	int ind_R = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind_R < size_F->nR_compact) {
		int ijk[3] = {};
		ijk[0] = blockIdx.y;
		ijk[1] = blockIdx.z;
		ijk[2] = k;

		int ind_dF = ind_R + (ijk[0] + ijk[1]*size_F->const_2Bx)*size_F->nR_compact;

		myReal y = dF[ind_dF].y;
		dF[ind_dF].y = dF[ind_dF].x * c[ijk[dim]];
		dF[ind_dF].x = -y * c[ijk[dim]];
	}
}

__global__ void get_c(myReal* c, const int i, const int j, const myReal* L, const myReal* G, const Size_F* size_F)
{
	if (i == j) {
		int ix = threadIdx.x;
		if (ix < size_F[0].Bx)
			c[ix] = -4*PI*PI * ix*ix * G[i+3*j] / (L[0]*L[0]);
		else
			c[ix] = -4*PI*PI * (ix-size_F[0].const_2Bx)*(ix-size_F[0].const_2Bx) * G[i+3*j] / (L[0]*L[0]);
	} else {
		int ix = threadIdx.x;
		int jx = threadIdx.y;

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
		c[indc] = -c1*c2 * G[i+3*j];
	}
}

__global__ void get_biasRW_small(myComplex* dF_temp, const myComplex* Fold, const myReal* c, const int i, const int j, const Size_F* size_F)
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
			dF_temp[ind].x = Fold[ind].x * c[ijk[i]];
			dF_temp[ind].y = Fold[ind].y * c[ijk[i]];
		} else {
			int indc = ijk[i] + ijk[j]*size_F[0].const_2Bx;
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

__host__ void modify_F(const myComplex* F, myComplex* F_modify, bool reduce,Size_F* size_F)
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

__host__ void permute_F(myComplex* F, bool R_first, const Size_F* size_F)
{
	myComplex* Fp = new myComplex[size_F->nTot_compact];
	if (R_first) {
		for (int iR = 0; iR < size_F->nR_compact; iR++) {
			for (int i = 0; i < size_F->const_2Bx; i++) {
				for (int j = 0; j < size_F->const_2Bx; j++) {
					for (int k = 0; k < size_F->const_2Bx; k++) {
						int ind_F = i + j*size_F->const_2Bx + k*size_F->const_2Bxs + iR*size_F->nx;
						int ind_Fp = iR + (i + j*size_F->const_2Bx + k*size_F->const_2Bxs) * size_F->nR_compact;

						Fp[ind_Fp] = F[ind_F];
					}
				}
			}
		}
	} else {
		for (int iR = 0; iR < size_F->nR_compact; iR++) {
			for (int i = 0; i < size_F->const_2Bx; i++) {
				for (int j = 0; j < size_F->const_2Bx; j++) {
					for (int k = 0; k < size_F->const_2Bx; k++) {
						int ind_F = iR + (i + j*size_F->const_2Bx + k*size_F->const_2Bxs) * size_F->nR_compact;
						int ind_Fp = i + j*size_F->const_2Bx + k*size_F->const_2Bxs + iR*size_F->nx;

						Fp[ind_Fp] = F[ind_F];
					}
				}
			}
		}
	}

	memcpy(F, Fp, size_F->nTot_compact * sizeof(myComplex));
	delete[] Fp;
}

__host__ void modify_u(const myComplex* u, myComplex* u_modify, Size_F* size_F)
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

__host__ void deriv_x(myReal* c, const int n, const int B, const myReal L)
{
	if (n < B)
		*c = 2*PI*n/L;
	else if (n == B)
		*c = 0;
	else
		*c = 2*PI*(n-2*B)/L;
}

__host__ void get_dF_small(myComplex* dF, const myComplex* F, const myComplex* X, const myComplex* OJO, const myComplex* MR, const myReal* b,
	const myReal* G, const myReal* L, const myComplex* u, const myReal* const* CG, const Size_F* size_F, const Size_F* size_F_dev)
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
	cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F->nx*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F->nx*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* X_ijk_dev;
	cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F->nx*sizeof(myComplex)));

	myComplex* dF3_dev;
	cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F->nTot_compact*sizeof(myComplex)));

	myComplex* dF_temp_dev;
	cudaErrorHandle(cudaMalloc(&dF_temp_dev, 3*size_F->nR_compact*sizeof(myComplex)));

	myComplex* u_dev;
	cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F->nR_compact*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(u_dev, u, 3*size_F->nR_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

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

	cutensor_initConv(&handle_cutensor, &plan_conv, &worksize_conv, F_dev, X_ijk_dev, dF_temp_dev, size_F->nR_compact, true, size_F);

	void* work = nullptr;
	if (worksize_conv > 0)
		cudaErrorHandle(cudaMalloc(&work, worksize_conv));

	myComplex alpha_cutensor = make_myComplex(0-(myReal)1/size_F->nx,0);
	myComplex beta_cutensor = make_myComplex(0,0);

	// set up blocksize and gridsize
	dim3 blocksize_8(8, 8, 8);
	int gridnum_8 = (int) size_F->const_2Bx/8 + 1;
	dim3 gridsize_8(gridnum_8, gridnum_8, gridnum_8);

	dim3 blocksize_512_nTot(512, 1, 1);
	dim3 gridsize_512_nTot((int)size_F->nTot_compact/512+1, 1, 1);

	// calculate
	for (int i = 0; i < size_F->const_2Bx; i++) {
		for (int j = 0; j < size_F->const_2Bx; j++) {
			for (int k = 0; k < size_F->const_2Bx; k++) {
				flip_shift <<<gridsize_8, blocksize_8>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
				cudaErrorHandle(cudaGetLastError());

				cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv, (void*)&alpha_cutensor, F_dev, X_ijk_dev,
					(void*)&beta_cutensor, dF_temp_dev, dF_temp_dev, work, worksize_conv, 0));

				for (int n = 0; n < 3; n++) {
					myComplex* dF3_dev_ijkn = dF3_dev + i*size_F->nR_compact + 
						j*(size_F->nR_compact*size_F->const_2Bx) + k*(size_F->nR_compact*size_F->const_2Bxs) + n*size_F->nTot_compact;
					myComplex* dF_temp_dev_n = dF_temp_dev + n*size_F->nR_compact;

					cudaErrorHandle(cudaMemcpy(dF3_dev_ijkn, dF_temp_dev_n, size_F->nR_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));
				}
			}
		}
	}

	for (int ip = 0; ip < 3; ip++) {
		for (int l = 0; l <= size_F->lmax; l++)
		{
			int ind_dF = l*(2*l-1)*(2*l+1)/3 + ip*size_F->nTot_compact;
			long long int stride_Fnew = size_F->nR_compact;

			int ind_u = l*(2*l-1)*(2*l+1)/3 + ip*size_F->nR_compact;
			long long int stride_u = 0;

			cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
				&alpha_cublas, dF3_dev+ind_dF, 2*l+1, stride_Fnew,
				u_dev+ind_u, 2*l+1, stride_u,
				&beta_cublas, dF3_dev+ind_dF, 2*l+1, stride_Fnew, size_F->nx));
		}
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	cudaErrorHandle(cudaMemcpy(dF_dev, dF3_dev, size_F->nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));

	// free memory
	cudaErrorHandle(cudaFree(u_dev));

	//////////////////////////////
	// circular convolution OJO //
	//////////////////////////////

	// OJO_ijk = flip(flip(flip(OJO,1),2),3)
	// OJO_ijk = circshift(OJO_ijk,1,i)
	// OJO_ijk = circshift(OJO_ijk,2,j)
	// OJO_ijk = circshift(OJO_ijk,3,k)
	// dF{r,i,j,k,p} = Fold{r,m,n,l}.*OJO_ijk{m,n,l,p}
	// dF{r,i,j,k,p} = dF{r,i,j,k,p}*c(p)
	// dF = sum(dF,'p')

	// set up arrays
	myComplex* OJO_dev;
	cudaErrorHandle(cudaMalloc(&OJO_dev, 3*size_F->nx*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(OJO_dev, OJO, 3*size_F->nx*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* OJO_ijk_dev;
	cudaErrorHandle(cudaMalloc(&OJO_ijk_dev, 3*size_F->nx*sizeof(myComplex)));

	// set up blocksize and gridsize
	dim3 blocksize_512_nR(512, 1, 1);
	dim3 gridsize_512_nR((int)size_F->nR_compact/512+1, 1, 1);

	// calculate
	for (int i = 0; i < size_F->const_2Bx; i++) {
		for (int j = 0; j < size_F->const_2Bx; j++) {
			for (int k = 0; k < size_F->const_2Bx; k++) {
				flip_shift <<<gridsize_8, blocksize_8>>> (OJO_dev, OJO_ijk_dev, i, j, k, size_F_dev);
				cudaErrorHandle(cudaGetLastError());

				cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv, (void*)&alpha_cutensor, F_dev, OJO_ijk_dev,
					(void*)&beta_cutensor, dF_temp_dev, dF_temp_dev, work, worksize_conv, 0));

				myReal c[3];
				deriv_x(c, i, size_F->Bx, *L);
				deriv_x(c+1, j, size_F->Bx, *L);
				deriv_x(c+2, k, size_F->Bx, *L);

				mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev, c[0], size_F->nR_compact);
				cudaErrorHandle(cudaGetLastError());
				mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev+size_F->nR_compact, c[1], size_F->nR_compact);
				cudaErrorHandle(cudaGetLastError());
				mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev+2*size_F->nR_compact, c[2], size_F->nR_compact);
				cudaErrorHandle(cudaGetLastError());

				for (int ip = 0; ip < 3; ip++) {
					myComplex* dF3_dev_ijkp = dF3_dev + i*size_F->nR_compact + 
						j*(size_F->nR_compact*size_F->const_2Bx) + k*(size_F->nR_compact*size_F->const_2Bxs) + ip*size_F->nTot_compact;
					myComplex* dF_temp_dev_p = dF_temp_dev + ip*size_F->nR_compact;

					cudaErrorHandle(cudaMemcpy(dF3_dev_ijkp, dF_temp_dev_p, size_F->nR_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));
				}
			}
		}
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	// free memory
	cudaErrorHandle(cudaFree(OJO_dev));
	cudaErrorHandle(cudaFree(OJO_ijk_dev));

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

	// calculate
	for (int i = 0; i < size_F->const_2Bx; i++) {
		for (int j = 0; j < size_F->const_2Bx; j++) {
			for (int k = 0; k < size_F->const_2Bx; k++) {
				flip_shift <<<gridsize_8, blocksize_8>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
				cudaErrorHandle(cudaGetLastError());

				cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv, (void*)&alpha_cutensor, F_dev, X_ijk_dev,
					(void*)&beta_cutensor, dF_temp_dev, dF_temp_dev, work, worksize_conv, 0));

				myReal c[3];
				deriv_x(c, i, size_F->Bx, *L);
				deriv_x(c+1, j, size_F->Bx, *L);
				deriv_x(c+2, k, size_F->Bx, *L);

				mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev, -c[0]*b[0], size_F->nR_compact);
				cudaErrorHandle(cudaGetLastError());
				mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev+size_F->nR_compact, -c[1]*b[1], size_F->nR_compact);
				cudaErrorHandle(cudaGetLastError());
				mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF_temp_dev+2*size_F->nR_compact, -c[2]*b[2], size_F->nR_compact);
				cudaErrorHandle(cudaGetLastError());

				for (int ip = 0; ip < 3; ip++) {
					myComplex* dF3_dev_ijkp = dF3_dev + i*size_F->nR_compact + 
						j*(size_F->nR_compact*size_F->const_2Bx) + k*(size_F->nR_compact*size_F->const_2Bxs) + ip*size_F->nTot_compact;
					myComplex* dF_temp_dev_p = dF_temp_dev + ip*size_F->nR_compact;

					cudaErrorHandle(cudaMemcpy(dF3_dev_ijkp, dF_temp_dev_p, size_F->nR_compact*sizeof(myComplex), cudaMemcpyDeviceToDevice));
				}
			}
		}
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	// free memory
	cudaErrorHandle(cudaFree(X_dev));
	cudaErrorHandle(cudaFree(X_ijk_dev));
	cudaErrorHandle(cudaFree(dF_temp_dev));
	if (worksize_conv > 0)
		cudaErrorHandle(cudaFree(work));

	///////////////////////
	// kronecker product //
	///////////////////////

	// set up arrays
	myComplex** CG_dev = new myComplex* [size_F->BR*size_F->BR];
	for (int l1 = 0; l1 <= size_F->lmax; l1++) {
		for (int l2 = 0; l2 <= size_F->lmax; l2++) {
			int m = (2*l1+1)*(2*l2+1);
			int ind_CG = l1+l2*size_F->BR;
			cudaErrorHandle(cudaMalloc(&CG_dev[ind_CG], m*m*sizeof(myComplex)));
			cudaErrorHandle(cudaMemset(CG_dev[ind_CG], 0, m*m*sizeof(myComplex)));

			myReal* CG_dev_d = (myReal*) CG_dev[ind_CG];
			cudaErrorHandle(cudaMemcpy2D(CG_dev_d, 2*sizeof(myReal), CG[ind_CG], sizeof(myReal), sizeof(myReal), m*m, cudaMemcpyHostToDevice));
		}
	}

	myComplex** F_strided = new myComplex* [size_F->BR];
	for (int l = 0; l <= size_F->lmax; l++) {
		int ind = l*(2*l-1)*(2*l+1)/3;
		int m = (2*l+1)*(2*l+1);
		cudaErrorHandle(cudaMalloc(&F_strided[l], m*size_F->nx*sizeof(myComplex)));
		cudaErrorHandle(cudaMemcpy2D(F_strided[l], m*sizeof(myComplex), F+ind, size_F->nR_compact*sizeof(myComplex),
			m*sizeof(myComplex), size_F->nx, cudaMemcpyHostToDevice));
	}

	myComplex* MR_dev;
	cudaErrorHandle(cudaMalloc(&MR_dev, 3*size_F->nR_compact*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(MR_dev, MR, 3*size_F->nR_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* FMR_dev;
	int m = (2*size_F->lmax+1) * (2*size_F->lmax+1);
	cudaErrorHandle(cudaMalloc(&FMR_dev, 3*m*sizeof(myComplex)));

	myComplex* FMR_temp_dev;
	cudaErrorHandle(cudaMalloc(&FMR_temp_dev, 3*size_F->nx*sizeof(myComplex)));

	cudaErrorHandle(cudaMemset(dF3_dev, 0, 3*size_F->nTot_compact*sizeof(myComplex)));

	// get c
	myReal* c = new myReal[size_F->const_2Bx];
	for (int i = 0; i < size_F->const_2Bx; i++) {
		deriv_x(&c[i], i, size_F->Bx, *L);
	}

	myReal* c_dev;
	cudaErrorHandle(cudaMalloc(&c_dev, size_F->const_2Bx*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(c_dev, c, size_F->const_2Bx*sizeof(myReal), cudaMemcpyHostToDevice));

	// set up cutensor
	cutensorContractionPlan_t* plan_FMR = new cutensorContractionPlan_t [size_F->BR];
	size_t* worksize_FMR = new size_t [size_F->BR];

	for (int l1 = 0; l1 <= size_F->lmax; l1++) {
		cutensor_initFMR(&handle_cutensor, &plan_FMR[l1], &worksize_FMR[l1], F_strided[l1], FMR_dev, FMR_temp_dev, l1, true, size_F);
	}

	size_t worksize_FMR_max = 0;
	for (int l = 0; l <= size_F->lmax; l++) {
		worksize_FMR_max = (worksize_FMR[l] > worksize_FMR_max) ? worksize_FMR[l] : worksize_FMR_max;
	}

	if (worksize_FMR_max > 0) {
		cudaErrorHandle(cudaMalloc(&work, worksize_FMR_max));
	}

	// set up blocksize and gridsize
	dim3 blocksize_addMFR(size_F->const_2Bx, size_F->const_2Bx, 1);
	dim3 gridsize_addMFR(size_F->const_2Bx, 3, 1);

	dim3 blocksize_deriv(512,1,1);
	dim3 gridsize_deriv((int)size_F->nR_compact/512+1, size_F->const_2Bx, size_F->const_2Bxs);

	// calculate
	for (int l = 0; l <= size_F->lmax; l++) {
		int ind_cumR = l*(2*l-1)*(2*l+1)/3;

		for (int l1 = 0; l1 <= size_F->lmax; l1++) {
			for (int l2 = 0; l2 <= size_F->lmax; l2++) {
				if (abs(l1-l2)<=l && l1+l2>=l) {
					int ind_MR = l2*(2*l2-1)*(2*l2+1)/3;
					int ind_CG = l1+l2*size_F->BR;
					int l12 = (2*l1+1)*(2*l2+1);

					alpha_cutensor.x = (myReal) -l12/(2*l+1);

					for (int m = -l; m <= l; m++) {
						int ind_CG_m = (l*l-(l1-l2)*(l1-l2)+m+l)*l12;

						for (int n = -l; n <= l; n++) {
							int ind_CG_n = (l*l-(l1-l2)*(l1-l2)+n+l)*l12;
							int ind_mnl = m+l + (n+l)*(2*l+1) + ind_cumR;

							cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2*l1+1, 2*l2+1, 2*l2+1,
								&alpha_cublas, CG_dev[ind_CG]+ind_CG_m, 2*l2+1, 0, MR_dev+ind_MR, 2*l2+1, size_F->nR_compact,
								&beta_cublas, FMR_temp_dev, 2*l1+1, (2*l1+1)*(2*l2+1), 3));

							cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, 2*l1+1, 2*l1+1, 2*l2+1,
								&alpha_cublas, FMR_temp_dev, 2*l1+1, (2*l1+1)*(2*l2+1), CG_dev[ind_CG]+ind_CG_n, 2*l2+1, 0,
								&beta_cublas, FMR_dev, 2*l1+1, (2*l1+1)*(2*l1+1), 3));

							cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_FMR[l1], &alpha_cutensor, F_strided[l1],
								FMR_dev, &beta_cutensor, FMR_temp_dev, FMR_temp_dev, work, worksize_FMR[l1], 0));

							add_FMR_small <<<gridsize_addMFR, blocksize_addMFR>>> (dF3_dev, FMR_temp_dev, ind_mnl, size_F_dev);
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		mulImg_FTot_small <<<gridsize_deriv, blocksize_deriv>>> (dF3_dev+i*size_F->nTot_compact, c_dev, i, size_F_dev);
		cudaErrorHandle(cudaGetLastError());
	}

	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF3_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	// free memory
	cudaErrorHandle(cudaFree(MR_dev));
	cudaErrorHandle(cudaFree(FMR_dev));
	cudaErrorHandle(cudaFree(FMR_temp_dev));
	cudaErrorHandle(cudaFree(c_dev));

	if (worksize_FMR_max > 0) {
		cudaErrorHandle(cudaFree(work));
	}

	for (int l1 = 0; l1 <= size_F->lmax; l1++) {
		for (int l2 = 0; l2 <= size_F->lmax; l2++) {
			int ind_CG = l1+l2*size_F->BR;
			cudaErrorHandle(cudaFree(CG_dev[ind_CG]));
		}
	}

	for (int l = 0; l <= size_F->lmax; l++) {
		cudaErrorHandle(cudaFree(F_strided[l]));
	}

	delete[] c;
	delete[] plan_FMR;
	delete[] worksize_FMR;
	delete[] CG_dev;
	delete[] F_strided;

	///////////////////////
	// random walk noise //
	///////////////////////

	// set up arrays
	cudaErrorHandle(cudaMalloc(&c_dev, size_F->const_2Bxs*sizeof(myReal)));

	myReal* G_dev;
	cudaErrorHandle(cudaMalloc(&G_dev, 9*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(G_dev, G, 9*sizeof(myReal), cudaMemcpyHostToDevice));

	myReal* L_dev;
	cudaErrorHandle(cudaMalloc(&L_dev, sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(L_dev, L, sizeof(myReal), cudaMemcpyHostToDevice));

	// set up block and grid sizes
	dim3 blocksize_512_nR_nx(512, 1, 1);
	dim3 gridsize_512_nR_nx((int)size_F->nR_compact/512+1, size_F->nx, 1);

	// calculate
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j) {
				dim3 blocksize_c(size_F->const_2Bx, 1, 1);
				get_c <<<1, blocksize_c>>> (c_dev, i, j, L_dev, G_dev, size_F_dev);
				cudaErrorHandle(cudaGetLastError());
			}
			else {
				dim3 blocksize_c(size_F->const_2Bx, size_F->const_2Bx, 1);
				get_c <<<1, blocksize_c>>> (c_dev, i, j, L_dev, G_dev, size_F_dev);
				cudaErrorHandle(cudaGetLastError());
			}

			get_biasRW_small <<<gridsize_512_nR_nx, blocksize_512_nR_nx>>> (dF3_dev, F_dev, c_dev, i, j, size_F_dev);
			cudaErrorHandle(cudaGetLastError());

			add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF3_dev, size_F->nTot_compact);
			cudaErrorHandle(cudaGetLastError());
		}
	}

	// free memory
	cudaErrorHandle(cudaFree(c_dev));
	cudaErrorHandle(cudaFree(G_dev));
	cudaErrorHandle(cudaFree(L_dev));
	cudaErrorHandle(cudaFree(F_dev));
	cudaErrorHandle(cudaFree(dF3_dev));

	// return
	cudaErrorHandle(cudaMemcpy(dF, dF_dev, size_F->nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));

	cudaErrorHandle(cudaFree(dF_dev));
}

__host__ void get_dF_large(myComplex* dF, myComplex* F, const myComplex* X, const myComplex* OJO, const myComplex* MR,
	const myReal* L, const myComplex* u, const myReal* const* CG, const Size_F* size_F, const Size_F* size_F_dev)
{
	////////////////////////////
	// circular_convolution X //
	////////////////////////////

	// X_ijk = flip(flip(flip(X,1),2),3)
	// X_ijk = circshift(X_ijk,1,i)
	// X_ijk = circshift(X_ijk,2,j)
	// X_ijk = circshift(X_ijk,3,k)
	// dF{r,i,j,k,p} = Fold{r,m,n,l}.*X_ijk{m,n,l,p}
	// dF(indmn,indmn,l,i,j,k,p) = -dF(indmn,indmn,l,i,j,k,p)*u(indmn,indmn,l,p)'
	// dF = sum(dF,'p')

	// set up GPU arrays
	myComplex* F_dev;
	if (size_F->nx*size_F->nR_split > size_F->nTot_splitx) {
		cudaErrorHandle(cudaMalloc(&F_dev, size_F->nx*size_F->nR_split*sizeof(myComplex)));
	} else {
		cudaErrorHandle(cudaMalloc(&F_dev, size_F->nTot_splitx*sizeof(myComplex)));
	}

	myComplex* X_dev;
	cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F->nx*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F->nx*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* X_ijk_dev;
	cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F->nx*sizeof(myComplex)));

	myComplex* dF3_dev;
	cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F->nTot_splitx*sizeof(myComplex)));

	myComplex* u_dev;
	cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F->nR_compact*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(u_dev, u, 3*size_F->nR_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

	// set up CPU arrays
	permute_F(F, false, size_F);

	myComplex* dF3 = new myComplex[3*size_F->nTot_compact];

	// set up cutensor
	cutensorHandle_t handle_cutensor;
	cutensorInit(&handle_cutensor);

	cutensorContractionPlan_t plan_conv[2];
	size_t worksize_conv[2] = {0,0};

	cutensor_initConv(&handle_cutensor, &plan_conv[0], &worksize_conv[0], F_dev, X_ijk_dev, dF3_dev, size_F->nR_split, false, size_F);
	cutensor_initConv(&handle_cutensor, &plan_conv[1], &worksize_conv[1], F_dev, X_ijk_dev, dF3_dev, size_F->nR_remainder, false, size_F);

	void* cutensor_workspace = nullptr;
	size_t worksize_max = worksize_conv[0]>worksize_conv[1] ? worksize_conv[0] : worksize_conv[1];
	if (worksize_max > 0) {
		cudaErrorHandle(cudaMalloc(&cutensor_workspace, worksize_max));
	}

	myComplex alpha_cutensor = make_myComplex(-(myReal)1/size_F->nx,0);
	myComplex beta_cutensor = make_myComplex(0,0);

	// set up cublas
	cublasHandle_t handle_cublas;
	cublasCreate(&handle_cublas);

	myComplex alpha_cublas = make_myComplex(1,0);
	myComplex beta_cublas = make_myComplex(0,0);

	// set up blocksize and gridsize
	dim3 blocksize_8(8, 8, 8);
	int gridnum_8 = (int)size_F->const_2Bx/8+1;
	dim3 gridsize_8(gridnum_8, gridnum_8, gridnum_8);

	dim3 blocksize_512_nTot(512, 1, 1);
	dim3 gridsize_512_nTot((int)size_F->nTot_splitx/512+1, 1, 1);

	// calculate
	for (int is = 0; is < size_F->ns; is++) {
		int nR_split;
		if (is == size_F->ns-1)
			nR_split = size_F->nR_remainder;
		else
			nR_split = size_F->nR_split;

		cudaErrorHandle(cudaMemcpy(F_dev, F+is*size_F->nx*size_F->nR_split, size_F->nx*nR_split*sizeof(myComplex), cudaMemcpyHostToDevice));

		for (int i = 0; i < size_F->const_2Bx; i++) {
			for (int j = 0; j < size_F->const_2Bx; j++) {
				for (int k = 0; k < size_F->const_2Bx; k++) {
					flip_shift <<<gridsize_8, blocksize_8>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
					cudaErrorHandle(cudaGetLastError());

					if (is == size_F->ns-1) {
						cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv[1], &alpha_cutensor, F_dev, X_ijk_dev,
							&beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize_conv[1], 0));
					} else {
						cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv[0], &alpha_cutensor, F_dev, X_ijk_dev,
							&beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize_conv[0], 0));
					}

					for (int ip = 0; ip < 3; ip++) {
						int ind_dF3 = is*size_F->nR_split + (i + j*size_F->const_2Bx + k*size_F->const_2Bxs)*size_F->nR_compact + ip*size_F->nTot_compact;
						cudaMemcpy(dF3+ind_dF3, dF3_dev+ip*nR_split, nR_split*sizeof(myComplex), cudaMemcpyDeviceToHost);
					}
				}
			}
		}
	}

	// set up CPU arrays
	permute_F(F, true, size_F);

	// multiply u
	for (int k = 0; k < size_F->const_2Bx; k++) {
		for (int ip = 0; ip < 3; ip++) {
			int ind_dF3 = k*size_F->nTot_splitx + ip*size_F->nTot_compact;
			int ind_dF3_dev = ip*size_F->nTot_splitx;

			cudaErrorHandle(cudaMemcpy(F_dev, dF3+ind_dF3, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

			for (int l = 0; l <= size_F->lmax; l++)
			{
				int ind_dF = l*(2*l-1)*(2*l+1)/3;
				long long int stride_dF = size_F->nR_compact;

				int ind_u = l*(2*l-1)*(2*l+1)/3 + ip*size_F->nR_compact;
				long long int stride_u = 0;

				cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
					&alpha_cublas, F_dev+ind_dF, 2*l+1, stride_dF,
					u_dev+ind_u, 2*l+1, stride_u,
					&beta_cublas, dF3_dev+ind_dF3_dev+ind_dF, 2*l+1, stride_dF, size_F->const_2Bxs));
			}
		}

		// addup dF
		addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_splitx);
		cudaErrorHandle(cudaGetLastError());

		int ind_dF = k*size_F->nTot_splitx;
		cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF3_dev, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
	}

	// free memory
	cudaErrorHandle(cudaFree(X_dev));
	cudaErrorHandle(cudaFree(X_ijk_dev));
	cudaErrorHandle(cudaFree(u_dev));

	//////////////////////////////
	// circular convolution OJO //
	//////////////////////////////

	// OJO_ijk = flip(flip(flip(OJO,1),2),3)
	// OJO_ijk = circshift(OJO_ijk,1,i)
	// OJO_ijk = circshift(OJO_ijk,2,j)
	// OJO_ijk = circshift(OJO_ijk,3,k)
	// dF{r,i,j,k,p} = Fold{r,m,n,l}.*OJO_ijk{m,n,l,p}
	// dF{r,i,j,k,p} = dF{r,i,j,k,p}*c(p)
	// dF = sum(dF,'p')

	// set up GPU arrays
	myComplex* OJO_dev;
	cudaErrorHandle(cudaMalloc(&OJO_dev, 3*size_F->nx*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(OJO_dev, OJO, 3*size_F->nx*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* OJO_ijk_dev;
	cudaErrorHandle(cudaMalloc(&OJO_ijk_dev, 3*size_F->nx*sizeof(myComplex)));

	// set up CPU arrays
	permute_F(F, false, size_F);

	// set up blocksize and gridsize
	dim3 blocksize_512_nR(512, 1, 1);
	dim3 gridsize_512_nR((int)size_F->nR_split/512+1, 1, 1);

	// calculate
	for (int is = 0; is < size_F->ns; is++) {
		int nR_split;
		if (is == size_F->ns-1) {
			nR_split = size_F->nR_remainder;
		} else {
			nR_split = size_F->nR_split;
		}

		gridsize_512_nR.x = (int)nR_split/512+1;

		cudaErrorHandle(cudaMemcpy(F_dev, F+is*size_F->nx*size_F->nR_split, size_F->nx*nR_split*sizeof(myComplex), cudaMemcpyHostToDevice));

		for (int i = 0; i < size_F->const_2Bx; i++) {
			for (int j = 0; j < size_F->const_2Bx; j++) {
				for (int k = 0; k < size_F->const_2Bx; k++) {
					flip_shift <<<gridsize_8, blocksize_8>>> (OJO_dev, OJO_ijk_dev, i, j, k, size_F_dev);
					cudaErrorHandle(cudaGetLastError());

					if (is == size_F->ns-1) {
						cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv[1], &alpha_cutensor, F_dev, OJO_ijk_dev,
							&beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize_conv[1], 0));
					} else {
						cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_conv[0], &alpha_cutensor, F_dev, OJO_ijk_dev,
							&beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize_conv[0], 0));
					}

					myReal c[3];
					deriv_x(c, i, size_F->Bx, *L);
					deriv_x(c+1, j, size_F->Bx, *L);
					deriv_x(c+2, k, size_F->Bx, *L);

					mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF3_dev, c[0], nR_split);
					cudaErrorHandle(cudaGetLastError());
					mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF3_dev+nR_split, c[1], nR_split);
					cudaErrorHandle(cudaGetLastError());
					mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF3_dev+2*nR_split, c[2], nR_split);
					cudaErrorHandle(cudaGetLastError());

					for (int ip = 0; ip < 3; ip++) {
						int ind_dF3 = is*size_F->nR_split + (i + j*size_F->const_2Bx + k*size_F->const_2Bxs)*size_F->nR_compact + ip*size_F->nTot_compact;
						cudaMemcpy(dF3+ind_dF3, dF3_dev+ip*nR_split, nR_split*sizeof(myComplex), cudaMemcpyDeviceToHost);
					}
				}
			}
		}
	}

	for (int k = 0; k < size_F->const_2Bx; k++) {
		for (int ip = 0; ip < 3; ip++) {
			int ind_dF3 = k*size_F->nTot_splitx + ip*size_F->nTot_compact;
			int ind_dF3_dev = ip*size_F->nTot_splitx;

			cudaErrorHandle(cudaMemcpy(dF3_dev+ind_dF3_dev, dF3+ind_dF3, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
		}

		addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_splitx);
		cudaErrorHandle(cudaGetLastError());

		int ind_dF = k*size_F->nTot_splitx;
		cudaErrorHandle(cudaMemcpy(dF3_dev+size_F->nTot_splitx, dF+ind_dF, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

		add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev+size_F->nTot_splitx, dF3_dev, size_F->nTot_splitx);
		cudaErrorHandle(cudaGetLastError());

		cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF3_dev+size_F->nTot_splitx, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
	}

	// free memory
	cudaErrorHandle(cudaFree(OJO_dev));
	cudaErrorHandle(cudaFree(OJO_ijk_dev));
	cudaErrorHandle(cudaFree(F_dev));

	if (worksize_max > 0) {
		cudaErrorHandle(cudaFree(cutensor_workspace));
	}

	///////////////////////
	// kronecker product //
	///////////////////////

	// set up GPU arrays
	myComplex** CG_dev = new myComplex* [size_F->BR*size_F->BR];
	for (int l1 = 0; l1 <= size_F->lmax; l1++) {
		for (int l2 = 0; l2 <= size_F->lmax; l2++) {
			int m = (2*l1+1)*(2*l2+1);
			int ind_CG = l1+l2*size_F->BR;
			cudaErrorHandle(cudaMalloc(&CG_dev[ind_CG], m*m*sizeof(myComplex)));
			cudaErrorHandle(cudaMemset(CG_dev[ind_CG], 0, m*m*sizeof(myComplex)));

			myReal* CG_dev_d = (myReal*) CG_dev[ind_CG];
			cudaErrorHandle(cudaMemcpy2D(CG_dev_d, 2*sizeof(myReal), CG[ind_CG], sizeof(myReal), sizeof(myReal), m*m, cudaMemcpyHostToDevice));
		}
	}

	myComplex* MR_dev;
	cudaErrorHandle(cudaMalloc(&MR_dev, 3*size_F->nR_compact*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(MR_dev, MR, 3*size_F->nR_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* FMR_dev;
	int m = (2*size_F->lmax+1) * (2*size_F->lmax+1);
	cudaErrorHandle(cudaMalloc(&FMR_dev, 3*m*sizeof(myComplex)));

	myComplex* FMR_temp_dev;
	cudaErrorHandle(cudaMalloc(&FMR_temp_dev, 3*size_F->const_2Bxs*sizeof(myComplex)));

	myComplex** F_strided = new myComplex* [size_F->BR];
	for (int l = 0; l <= size_F->lmax; l++) {
		int m = (2*l+1)*(2*l+1);
		cudaErrorHandle(cudaMalloc(&F_strided[l], m*size_F->const_2Bxs*sizeof(myComplex)));
	}

	// set up CPU arrays
	permute_F(F, true, size_F);

	// get c
	myReal* c = new myReal[size_F->const_2Bx];
	for (int i = 0; i < size_F->const_2Bx; i++) {
		deriv_x(&c[i], i, size_F->Bx, *L);
	}

	myReal* c_dev;
	cudaErrorHandle(cudaMalloc(&c_dev, size_F->const_2Bx*sizeof(myReal)));
	cudaErrorHandle(cudaMemcpy(c_dev, c, size_F->const_2Bx*sizeof(myReal), cudaMemcpyHostToDevice));

	// set up cutensor
	cutensorContractionPlan_t* plan_FMR = new cutensorContractionPlan_t [size_F->BR];
	size_t* worksize_FMR = new size_t [size_F->BR];

	for (int l1 = 0; l1 <= size_F->lmax; l1++) {
		cutensor_initFMR(&handle_cutensor, &plan_FMR[l1], &worksize_FMR[l1], F_strided[l1], FMR_dev, FMR_temp_dev, l1, false, size_F);
	}

	worksize_max = 0;
	for (int l = 0; l <= size_F->lmax; l++) {
		worksize_max = (worksize_FMR[l] > worksize_max) ? worksize_FMR[l] : worksize_max;
	}

	if (worksize_max > 0) {
		cudaErrorHandle(cudaMalloc(&cutensor_workspace, worksize_max));
	}

	// set up blocksize and gridsize
	dim3 blocksize_addMFR(size_F->const_2Bx, 3, 1);
	dim3 gridsize_addMFR(size_F->const_2Bx, 1, 1);

	dim3 blocksize_deriv(512,1,1);
	dim3 gridsize_deriv((int)size_F->nR_compact/512+1, size_F->const_2Bx, size_F->const_2Bx);

	// calculate
	for (int k = 0; k < size_F->const_2Bx; k++) {
		int ind_Fold = k*size_F->nTot_splitx;
		for (int l = 0; l <= size_F->lmax; l++) {
			int ind = l*(2*l-1)*(2*l+1)/3;
			int m = (2*l+1)*(2*l+1);
			cudaErrorHandle(cudaMemcpy2D(F_strided[l], m*sizeof(myComplex), F+ind_Fold+ind, size_F->nR_compact*sizeof(myComplex),
				m*sizeof(myComplex), size_F->const_2Bxs, cudaMemcpyHostToDevice));
		}

		cudaErrorHandle(cudaMemset(dF3_dev, 0, 3*size_F->nTot_splitx*sizeof(myComplex)));

		for (int l = 0; l <= size_F->lmax; l++) {
			int ind_cumR = l*(2*l-1)*(2*l+1)/3;

			for (int l1 = 0; l1 <= size_F->lmax; l1++) {
				for (int l2 = 0; l2 <= size_F->lmax; l2++) {
					if (abs(l1-l2)<=l && l1+l2>=l) {
						int ind_MR = l2*(2*l2-1)*(2*l2+1)/3;
						int ind_CG = l1+l2*size_F->BR;
						int l12 = (2*l1+1)*(2*l2+1);

						alpha_cutensor.x = (myReal) -l12/(2*l+1);

						for (int m = -l; m <= l; m++) {
							int ind_CG_m = (l*l-(l1-l2)*(l1-l2)+m+l)*l12;

							for (int n = -l; n <= l; n++) {
								int ind_CG_n = (l*l-(l1-l2)*(l1-l2)+n+l)*l12;
								int ind_mnl = m+l + (n+l)*(2*l+1) + ind_cumR;

								cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2*l1+1, 2*l2+1, 2*l2+1,
									&alpha_cublas, CG_dev[ind_CG]+ind_CG_m, 2*l2+1, 0, MR_dev+ind_MR, 2*l2+1, size_F->nR_compact,
									&beta_cublas, FMR_temp_dev, 2*l1+1, (2*l1+1)*(2*l2+1), 3));

								cublasErrorHandle(mycublasgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N, 2*l1+1, 2*l1+1, 2*l2+1,
									&alpha_cublas, FMR_temp_dev, 2*l1+1, (2*l1+1)*(2*l2+1), CG_dev[ind_CG]+ind_CG_n, 2*l2+1, 0,
									&beta_cublas, FMR_dev, 2*l1+1, (2*l1+1)*(2*l1+1), 3));

								cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan_FMR[l1], &alpha_cutensor, F_strided[l1],
									FMR_dev, &beta_cutensor, FMR_temp_dev, FMR_temp_dev, cutensor_workspace, worksize_FMR[l1], 0));

								add_FMR_large <<<gridsize_addMFR, blocksize_addMFR>>> (dF3_dev, FMR_temp_dev, ind_mnl, size_F_dev);
							}
						}
					}
				}
			}
		}

		for (int ip = 0; ip < 3; ip++) {
			int ind_dF3 = ind_Fold + ip*size_F->nTot_compact;
			int ind_dF3_dev = ip*size_F->nTot_splitx;
			cudaErrorHandle(cudaMemcpy(dF3+ind_dF3, dF3_dev+ind_dF3_dev, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
		}
	}

	for (int k = 0; k < size_F->const_2Bx; k++) {
		// take derivative about x
		for (int ip = 0; ip < 3; ip++) {
			int ind_dF3 = k*size_F->nTot_splitx + ip*size_F->nTot_compact;
			int ind_dF3_dev = ip*size_F->nTot_splitx;

			cudaErrorHandle(cudaMemcpy(dF3_dev+ind_dF3_dev, dF3+ind_dF3, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

			mulImg_FTot_large <<<gridsize_deriv, blocksize_deriv>>> (dF3_dev+ind_dF3_dev, c_dev, ip, k, size_F_dev);
			cudaErrorHandle(cudaGetLastError());
		}

		// addup dF
		addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_splitx);
		cudaErrorHandle(cudaGetLastError());

		int ind_dF = k*size_F->nTot_splitx;
		cudaErrorHandle(cudaMemcpy(dF3_dev+size_F->nTot_splitx, dF+ind_dF, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

		add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev+size_F->nTot_splitx, dF3_dev, size_F->nTot_splitx);
		cudaErrorHandle(cudaGetLastError());

		cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF3_dev+size_F->nTot_splitx, size_F->nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
	}

	// free memory
	cudaErrorHandle(cudaFree(dF3_dev));
	cudaErrorHandle(cudaFree(MR_dev));
	cudaErrorHandle(cudaFree(FMR_dev));
	cudaErrorHandle(cudaFree(FMR_temp_dev));
	cudaErrorHandle(cudaFree(c_dev));

	if (worksize_max > 0) {
		cudaErrorHandle(cudaFree(cutensor_workspace));
	}

	for (int l1 = 0; l1 <= size_F->lmax; l1++) {
		for (int l2 = 0; l2 <= size_F->lmax; l2++) {
			int ind_CG = l1+l2*size_F->BR;
			cudaErrorHandle(cudaFree(CG_dev[ind_CG]));
		}
	}

	for (int l = 0; l <= size_F->lmax; l++) {
		cudaErrorHandle(cudaFree(F_strided[l]));
	}

	delete[] c;
	delete[] CG_dev;
	delete[] F_strided;
	delete[] dF3;
	delete[] plan_FMR;
	delete[] worksize_FMR;
}

