#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

__global__ void flip_shift(const cuDoubleComplex* X, cuDoubleComplex* X_ijk, const int is, const int js, const int ks, const Size_F* size_F)
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

__global__ void addup_F(cuDoubleComplex* dF, const int nTot)
{
	int ind1 = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind1 < nTot) {
		int ind2 = ind1 + nTot;
		int ind3 = ind2 + nTot;

		dF[ind1] = cuCadd(dF[ind1], dF[ind2]);
		dF[ind1] = cuCadd(dF[ind1], dF[ind3]);
	}
}

__global__ void add_F(cuDoubleComplex*dF, const cuDoubleComplex* dF_temp, const int nTot)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind < nTot)
		dF[ind] = cuCadd(dF[ind], dF_temp[ind]);
}

__global__ void get_c(double* c, const int i, const int j, const double* G, const Size_F* size_F)
{
	if (i == j) {
		int ix = threadIdx.x;
		if (ix < size_F[0].Bx)
			c[ix] = -PI*PI * ix*ix * G[i+3*j];
		else
			c[ix] = -PI*PI * (ix-size_F[0].const_2Bx)*(ix-size_F[0].const_2Bx) * G[i+3*j];
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
		c[indc] = -c1*c2 * G[i+3*j];
	}
}

__global__ void get_biasRW_small(cuDoubleComplex* dF_temp, const cuDoubleComplex* Fold, const double* c, const int i, const int j,const  Size_F* size_F)
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

__global__ void get_biasRW_large(cuDoubleComplex* dF_temp, const cuDoubleComplex* Fold, const double* c, const int i, const int j, const int ijk_k, const Size_F* size_F)
{
	int indR = threadIdx.x + blockIdx.x*blockDim.x;
	if (indR < size_F[0].nR_compact) {
		int ijk[3] = {blockIdx.y, blockIdx.z, ijk_k};

		int ind = indR + (ijk[0] + ijk[1]*size_F->const_2Bx)*size_F->nR_compact;

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

__global__ void integrate_Fnew(cuDoubleComplex* Fnew, const cuDoubleComplex* Fold, const cuDoubleComplex* dF, const double dt, const int nTot)
{
	int ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind < nTot)
	{
		Fnew[ind].x = Fold[ind].x + dt*dF[ind].x;
		Fnew[ind].y = Fold[ind].y + dt*dF[ind].y;
	}
}

__host__ void modify_F(const cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce, Size_F* size_F)
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

__host__ void permute_F(cuDoubleComplex* F, bool R_first, const Size_F* size_F)
{
	cuDoubleComplex* Fp = new cuDoubleComplex[size_F->nTot_compact];
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

	memcpy(F, Fp, size_F->nTot_compact * sizeof(cuDoubleComplex));
	delete[] Fp;
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

__host__ void get_dF_small(cuDoubleComplex* dF, const cuDoubleComplex* F, const cuDoubleComplex* X, const cuDoubleComplex* u,
	const double* G1, const double* G2, const Size_F* size_F, const Size_F* size_F_dev)
{
	//////////////////////////
	// circular convolution //
	//////////////////////////

	// X_ijk = flip(flip(flip(X,1),2),3)
	// X_ijk = circshift(X_ijk,1,i)
	// X_ijk = circshift(X_ijk,2,j)
	// X_ijk = circshift(X_ijk,3,k)
	// dF{r,i,j,k,p} = Fold{r,m,n,l}.*X_ijk{m,n,l,p}
	// dF(indmn,indmn,l,i,j,k,p) = -dF(indmn,indmn,l,i,j,k,p)*u(indmn,indmn,l,p)'
	// dF = sum(dF,'p')

	// set up GPU arrays
	cuDoubleComplex* F_dev;
	cudaErrorHandle(cudaMalloc(&F_dev, size_F->nTot_compact*sizeof(cuDoubleComplex)));
	cudaErrorHandle(cudaMemcpy(F_dev, F, size_F->nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

	cuDoubleComplex* X_dev;
	cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F->nx*sizeof(cuDoubleComplex)));
	cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F->nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

	cuDoubleComplex* X_ijk_dev;
	cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F->nx*sizeof(cuDoubleComplex)));

	cuDoubleComplex* u_dev;
	cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F->nR_compact*sizeof(cuDoubleComplex)));
	cudaErrorHandle(cudaMemcpy(u_dev, u, 3*size_F->nR_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

	cuDoubleComplex* dF_dev;
	cudaErrorHandle(cudaMalloc(&dF_dev, 3*size_F->nTot_compact*sizeof(cuDoubleComplex)));

	cuDoubleComplex* dF_temp_dev;
	cudaErrorHandle(cudaMalloc(&dF_temp_dev, size_F->nTot_compact*sizeof(cuDoubleComplex)));

	// set up cutensor
	cutensorHandle_t handle_cutensor;
	cutensorInit(&handle_cutensor);

	size_t worksize = 0;
	cutensorContractionPlan_t plan;

	cutensor_initialize(&handle_cutensor, &plan, &worksize, F_dev, X_ijk_dev, dF_temp_dev, size_F->nR_compact, true, size_F);

	void* work = nullptr;
	if (worksize > 0) {
		cudaErrorHandle(cudaMalloc(&work, worksize));
	}

	cuDoubleComplex alpha = make_cuDoubleComplex((double)1/size_F->nx,0);
	cuDoubleComplex beta = make_cuDoubleComplex(0,0);

	// set up cublas
	cublasHandle_t handle_cublas;
	cublasCreate(&handle_cublas);

	cuDoubleComplex alpha_cublas = make_cuDoubleComplex(-1,0);

	// set up block and grid sizes
	dim3 blocksize_X(8, 8, 8);
	int gridnum_X = (int)size_F->const_2Bx/8+1;
	dim3 gridsize_X(gridnum_X, gridnum_X, gridnum_X);

	dim3 blocksize_512_nTot(512, 1, 1);
	dim3 gridsize_512_nTot((int)size_F->nTot_compact/512+1, 1, 1);

	// calculate
	// convolution
	for (int i = 0; i < size_F->const_2Bx; i++) {
		for (int j = 0; j < size_F->const_2Bx; j++) {
			for (int k = 0; k < size_F->const_2Bx; k++) {
				flip_shift <<<gridsize_X, blocksize_X>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
				cudaErrorHandle(cudaGetLastError());

				cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan, (void*)&alpha, F_dev, X_ijk_dev,
					(void*)&beta, dF_temp_dev, dF_temp_dev, work, worksize, 0));

				for (int ip = 0; ip < 3; ip++) {
					cuDoubleComplex* dF_dev_ijkp = dF_dev + i*size_F->nR_compact + 
						j*(size_F->nR_compact*size_F->const_2Bx) + k*(size_F->nR_compact*size_F->const_2Bxs) + ip*size_F->nTot_compact;
					cuDoubleComplex* temp_dev_p = dF_temp_dev + ip*size_F->nR_compact;

					cudaErrorHandle(cudaMemcpy(dF_dev_ijkp, temp_dev_p, size_F->nR_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
				}
			}
		}
	}

	// multiply u
	for (int i = 0; i < 3; i++) {
		for (int l = 0; l <= size_F->lmax; l++)
		{
			int ind_dF = l*(2*l-1)*(2*l+1)/3 + i*size_F->nTot_compact;
			long long int stride_Fnew = size_F->nR_compact;

			int ind_u = l*(2*l-1)*(2*l+1)/3 + i*size_F->nR_compact;
			long long int stride_u = 0;

			cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
				&alpha_cublas, dF_dev+ind_dF, 2*l+1, stride_Fnew,
				u_dev+ind_u, 2*l+1, stride_u,
				&beta, dF_dev+ind_dF, 2*l+1, stride_Fnew, size_F->nx));
		}
	}

	// addup F
	addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, size_F->nTot_compact);
	cudaErrorHandle(cudaGetLastError());

	// free memory
	cudaErrorHandle(cudaFree(X_dev));
	cudaErrorHandle(cudaFree(X_ijk_dev));

	if (worksize > 0) {
		cudaErrorHandle(cudaFree(work));
	}

	////////////////////////////
	// gyro random walk noise //
	////////////////////////////

	// dF_temp(indmn,indmn,l,nx,i,j) = dF_temp(indmn,indmn,l,nx)*u(indmn,indmn,l,i)'*u(indmn,indmn,l,j)'
	// dF = dF + sum(dF_temp,'i','j')

	// calculate
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int l = 0; l <= size_F->lmax; l++) {
				int ind_F = l*(2*l-1)*(2*l+1)/3;
				long long int stride_F = size_F->nR_compact;

				int ind_u1 = l*(2*l-1)*(2*l+1)/3 + i*size_F->nR_compact;
				int ind_u2 = l*(2*l-1)*(2*l+1)/3 + j*size_F->nR_compact;
				long long int stride_u = 0;

				alpha_cublas.x = 1;

				cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
					&alpha_cublas, F_dev+ind_F, 2*l+1, stride_F,
					u_dev+ind_u1, 2*l+1, stride_u,
					&beta, dF_temp_dev+ind_F, 2*l+1, stride_F, size_F->nx));

				alpha_cublas.x = G1[i+j*3];

				cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
					&alpha_cublas, dF_temp_dev+ind_F, 2*l+1, stride_F,
					u_dev+ind_u2, 2*l+1, stride_u,
					&beta, dF_temp_dev+ind_F, 2*l+1, stride_F, size_F->nx));
			}

			add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF_temp_dev, size_F->nTot_compact);
			cudaErrorHandle(cudaGetLastError());
		}
	}

	// free memory
	cudaErrorHandle(cudaFree(u_dev));

	////////////////////////////
	// bias random walk noise //
	////////////////////////////

	// set up arrays
	double* c_dev;
	cudaErrorHandle(cudaMalloc(&c_dev, size_F->const_2Bxs*sizeof(double)));

	double* G2_dev;
	cudaErrorHandle(cudaMalloc(&G2_dev, 9*sizeof(double)));
	cudaErrorHandle(cudaMemcpy(G2_dev, G2, 9*sizeof(double), cudaMemcpyHostToDevice));

	// set up block and grid sizes
	dim3 blocksize_512_nR_nx(512, 1, 1);
	dim3 gridsize_512_nR_nx((int)size_F->nR_compact/512+1, size_F->nx, 1);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j) {
				dim3 blocksize_c(size_F->const_2Bx, 1, 1);
				get_c <<<1, blocksize_c>>> (c_dev, i, j, G2_dev, size_F_dev);
				cudaErrorHandle(cudaGetLastError());
			}
			else {
				dim3 blocksize_c(size_F->const_2Bx, size_F->const_2Bx, 1);
				get_c <<<1, blocksize_c>>> (c_dev, i, j, G2_dev, size_F_dev);
				cudaErrorHandle(cudaGetLastError());
			}

			get_biasRW_small <<<gridsize_512_nR_nx, blocksize_512_nR_nx>>> (dF_temp_dev, F_dev, c_dev, i, j, size_F_dev);
			cudaErrorHandle(cudaGetLastError());

			add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF_temp_dev, size_F->nTot_compact);
			cudaErrorHandle(cudaGetLastError());
		}
	}

	// return
	cudaErrorHandle(cudaMemcpy(dF, dF_dev, size_F->nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

	// free memory
	cudaErrorHandle(cudaFree(G2_dev));
	cudaErrorHandle(cudaFree(c_dev));
	cudaErrorHandle(cudaFree(dF_temp_dev));
	cudaErrorHandle(cudaFree(F_dev));
	cudaErrorHandle(cudaFree(dF_dev));
}

__host__ void get_dF_large(cuDoubleComplex* dF, cuDoubleComplex* F, const cuDoubleComplex* X, const cuDoubleComplex* u,
	const double* G1, const double* G2, const Size_F* size_F, const Size_F* size_F_dev)
{
    //////////////////////////
    // circular convolution //
    //////////////////////////

    // X_ijk = flip(flip(flip(X,1),2),3)
    // X_ijk = circshift(X_ijk,1,i)
    // X_ijk = circshift(X_ijk,2,j)
    // X_ijk = circshift(X_ijk,3,k)
    // dF{r,i,j,k,p} = Fold{r,m,n,l}.*X_ijk{m,n,l,p}
    // dF(indmn,indmn,l,i,j,k,p) = -dF(indmn,indmn,l,i,j,k,p)*u(indmn,indmn,l,p)'
    // dF = sum(dF,'p')

    // set up GPU arrays
    cuDoubleComplex* F_dev;
	if (size_F->nx*size_F->nR_split > size_F->nTot_splitx) {
		cudaErrorHandle(cudaMalloc(&F_dev, size_F->nx*size_F->nR_split*sizeof(cuDoubleComplex)));
	} else {
		cudaErrorHandle(cudaMalloc(&F_dev, size_F->nTot_splitx*sizeof(cuDoubleComplex)));
	}

    cuDoubleComplex* X_dev;
    cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F->nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F->nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* u_dev;
    cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F->nR_compact*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(u_dev, u, 3*size_F->nR_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* dF3_dev;
    cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F->nTot_splitx*sizeof(cuDoubleComplex)));

    cuDoubleComplex* X_ijk_dev;
    cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F->nx*sizeof(cuDoubleComplex)));

    // set up CPU arrays
    permute_F(F, false, size_F);

    cuDoubleComplex* dF3 = new cuDoubleComplex[3*size_F->nTot_compact];

    // set up cutensor
    cutensorHandle_t handle_cutensor;
    cutensorInit(&handle_cutensor);

    size_t worksize[2] = {0,0};
    cutensorContractionPlan_t plan[2];

    cutensor_initialize(&handle_cutensor, plan, worksize, F_dev, X_dev, dF3_dev, size_F->nR_split, false, size_F);
    cutensor_initialize(&handle_cutensor, plan+1, worksize+1, F_dev, X_dev, dF3_dev, size_F->nR_remainder, false, size_F);

    size_t worksize_max = worksize[0]>worksize[1] ? worksize[0] : worksize[1];

    void* cutensor_workspace = nullptr;
    if (worksize_max > 0) {
        cudaErrorHandle(cudaMalloc(&cutensor_workspace, worksize_max));
    }

    cuDoubleComplex alpha = make_cuDoubleComplex((double)1/size_F->nx,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);

    // set up cublas
    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    cuDoubleComplex alpha_cublas = make_cuDoubleComplex(-1,0);

    // set up block and grid sizes
    dim3 blocksize_X(8, 8, 8);
    int gridnum_X = (int)size_F->const_2Bx/8+1;
    dim3 gridsize_X(gridnum_X, gridnum_X, gridnum_X);

    dim3 blocksize_512_nTot(512, 1, 1);
    dim3 gridsize_512_nTot((int)size_F->nTot_splitx/512+1, 1, 1);

    // calculate
    // circular convolution
    for (int is = 0; is < size_F->ns; is++) {
        int nR_split;
        if (is == size_F->ns-1)
            nR_split = size_F->nR_remainder;
        else
            nR_split = size_F->nR_split;

        cudaErrorHandle(cudaMemcpy(F_dev, F + is*size_F->nx*size_F->nR_split,
            size_F->nx*nR_split*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < size_F->const_2Bx; i++) {
            for (int j = 0; j < size_F->const_2Bx; j++) {
                for (int k = 0; k < size_F->const_2Bx; k++) {
                    flip_shift <<<gridsize_X, blocksize_X>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());

                    if (is == size_F->ns-1) {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, plan+1, &alpha, F_dev, X_ijk_dev,
                            &beta, dF3_dev, dF3_dev, cutensor_workspace, worksize[1], 0));
                    } else {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, plan, &alpha, F_dev, X_ijk_dev,
                            &beta, dF3_dev, dF3_dev, cutensor_workspace, worksize[0], 0));
                    }

                    for (int ip = 0; ip < 3; ip++) {
                        int ind_dF = is*size_F->nR_split + (i + j*size_F->const_2Bx + k*size_F->const_2Bxs)*size_F->nR_compact + ip*size_F->nTot_compact;
						cudaErrorHandle(cudaMemcpy(dF3+ind_dF, dF3_dev+ip*nR_split, nR_split*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
                    }
                }
            }
        }
    }

    // set up CPU array
    permute_F(F, true, size_F);

    // multiply u
    for (int k = 0; k < size_F->const_2Bx; k++) {
        for (int ip = 0; ip < 3; ip++) {
            int ind_dF3 = k*size_F->nTot_splitx + ip*size_F->nTot_compact;
            int ind_dF3_dev = ip*size_F->nTot_splitx;
            cudaErrorHandle(cudaMemcpy(F_dev, dF3+ind_dF3, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            for (int l = 0; l <= size_F->lmax; l++)
            {
                int ind_dF = l*(2*l-1)*(2*l+1)/3;
                long long int stride_dF = size_F->nR_compact;

                int ind_u = l*(2*l-1)*(2*l+1)/3 + ip*size_F->nR_compact;
                long long int stride_u = 0;

                cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                    &alpha_cublas, F_dev+ind_dF, 2*l+1, stride_dF,
                    u_dev+ind_u, 2*l+1, stride_u,
                    &beta, dF3_dev+ind_dF3_dev+ind_dF, 2*l+1, stride_dF, size_F->const_2Bxs));
            }
        }

        // addup F
        addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F->nTot_splitx);
        cudaErrorHandle(cudaGetLastError());

        int ind_dF = k*size_F->nTot_splitx;
        cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF3_dev, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // free memory
    cudaErrorHandle(cudaFree(X_dev));
    cudaErrorHandle(cudaFree(X_ijk_dev));

    if (worksize_max > 0) {
        cudaErrorHandle(cudaFree(cutensor_workspace));
    }

    delete[] dF3;

    ////////////////////////////
    // gyro random walk noise //
    ////////////////////////////

    // dF_temp(indmn,indmn,l,nx,i,j) = dF_temp(indmn,indmn,l,nx)*u(indmn,indmn,l,i)'*u(indmn,indmn,l,j)'
    // dF = dF + sum(dF_temp,'i','j')

    for (int k = 0; k < size_F->const_2Bx; k++) {
        cudaErrorHandle(cudaMemcpy(F_dev, F+k*size_F->nTot_splitx, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(dF3_dev, dF+k*size_F->nTot_splitx, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int l = 0; l <= size_F->lmax; l++) {
                    int ind_F = l*(2*l-1)*(2*l+1)/3;
                    long long int stride_F = size_F->nR_compact;

                    int ind_u1 = l*(2*l-1)*(2*l+1)/3 + i*size_F->nR_compact;
                    int ind_u2 = l*(2*l-1)*(2*l+1)/3 + j*size_F->nR_compact;
                    long long int stride_u = 0;

                    alpha_cublas.x = 1;

                    cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                        &alpha_cublas, F_dev+ind_F, 2*l+1, stride_F,
                        u_dev+ind_u1, 2*l+1, stride_u,
                        &beta, dF3_dev+size_F->nTot_splitx+ind_F, 2*l+1, stride_F, size_F->const_2Bxs));

                    alpha_cublas.x = G1[i+j*3];

                    cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                        &alpha_cublas, dF3_dev+size_F->nTot_splitx+ind_F, 2*l+1, stride_F,
                        u_dev+ind_u2, 2*l+1, stride_u,
                        &beta, dF3_dev+2*size_F->nTot_splitx+ind_F, 2*l+1, stride_F, size_F->const_2Bxs));
                }

                add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, dF3_dev+2*size_F->nTot_splitx, size_F->nTot_splitx);
                cudaErrorHandle(cudaGetLastError());
            }
        }

		cudaErrorHandle(cudaMemcpy(dF + k*size_F->nTot_splitx, dF3_dev, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // free memory
    cudaErrorHandle(cudaFree(u_dev));

    ////////////////////////////
    // bias random walk noise //
    ////////////////////////////

    // set up GPU arrays
    double* c_dev;
    cudaErrorHandle(cudaMalloc(&c_dev, 9*size_F->const_2Bxs*sizeof(double)));

    double* G2_dev;
    cudaErrorHandle(cudaMalloc(&G2_dev, 9*sizeof(double)));
    cudaErrorHandle(cudaMemcpy(G2_dev, G2, 9*sizeof(double), cudaMemcpyHostToDevice));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int ind_c = (i+3*j)*size_F->const_2Bxs;
            if (i == j) {
                dim3 blocksize_c(size_F->const_2Bx, 1, 1);
                get_c <<<1, blocksize_c>>> (c_dev+ind_c, i, j, G2_dev, size_F_dev);
                cudaErrorHandle(cudaGetLastError());
            }
            else {
                dim3 blocksize_c(size_F->const_2Bx, size_F->const_2Bx, 1);
                get_c <<<1, blocksize_c>>> (c_dev+ind_c, i, j, G2_dev, size_F_dev);
                cudaErrorHandle(cudaGetLastError());
            }
        }
    }

    // set up block and grid sizes
    dim3 blocksize_512_nR_nx(512,1,1);
    dim3 gridsize_512_nR_nx((int)size_F->nR_compact/512+1, size_F->const_2Bx, size_F->const_2Bx);

    // calculate
    for (int k = 0; k < size_F->const_2Bx; k++) {
        cudaErrorHandle(cudaMemcpy(F_dev, F + k*size_F->nTot_splitx, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(dF3_dev, dF + k*size_F->nTot_splitx, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int ind_c = (i+3*j)*size_F->const_2Bxs;
                get_biasRW_large <<<gridsize_512_nR_nx, blocksize_512_nR_nx>>> (dF3_dev+size_F->nTot_splitx, F_dev, c_dev+ind_c, i, j, k, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, dF3_dev+size_F->nTot_splitx, size_F->nTot_splitx);
                cudaErrorHandle(cudaGetLastError());
            }
        }

		cudaErrorHandle(cudaMemcpy(dF + k*size_F->nTot_splitx, dF3_dev, size_F->nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // free memory
    cudaErrorHandle(cudaFree(c_dev));
    cudaErrorHandle(cudaFree(G2_dev));
    cudaErrorHandle(cudaFree(F_dev));
    cudaErrorHandle(cudaFree(dF3_dev));
}

