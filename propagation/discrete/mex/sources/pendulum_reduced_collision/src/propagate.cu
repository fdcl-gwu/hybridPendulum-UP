#include "propagate.cuh"

#include <math.h>
#include <stdio.h>
#include "omp.h"

void get_df_noise(myReal* df, const myReal* f, const myReal* lambda, myReal* const* fcL, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, int* const* fcL_indx1, const int* fcL_numx1, int* const* fcL_indx2, int* const* fcL_numx2, const Size_f* size_f)
{
    // number of threads
    int nthread = size_f->nx;
    cudaStream_t cudaStreams[nthread];
    for (int i = 0; i < nthread; i++) {
        cudaErrorHandle(cudaStreamCreate(&cudaStreams[i]));
    }

    // maximum memory needed
    int max_numx2 = 0;
    for (int iR = 0; iR < numR; iR++) {
        max_numx2 = (max_numx2 < fcL_numx2[iR][fcL_numx1[iR]]) ? fcL_numx2[iR][fcL_numx1[iR]] : max_numx2;
    }

    // density in
    myReal* fcL_dev;
    cudaErrorHandle(cudaMalloc(&fcL_dev, max_numx2*sizeof(myReal)));

    int* fcL_indx2_dev;
    cudaErrorHandle(cudaMalloc(&fcL_indx2_dev, max_numx2*sizeof(int)));

    myReal* f_dev;
    cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(f_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    myReal* f_temp_dev;
    cudaErrorHandle(cudaMalloc(&f_temp_dev, max_numx2*sizeof(myReal)));

    myReal* df_dev;
    cudaErrorHandle(cudaMalloc(&df_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(df_dev, 0, size_f->nTot*sizeof(myReal)));

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    for (int iR = 0; iR < numR; iR++) {
        int numx2 = fcL_numx2[iR][fcL_numx1[iR]];
        cudaErrorHandle(cudaMemcpy(fcL_dev, fcL[iR], numx2*sizeof(myReal), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(fcL_indx2_dev, fcL_indx2[iR], numx2*sizeof(int), cudaMemcpyHostToDevice));

        get_fold_noise <<<(int)numx2/128+1, 128>>> (f_temp_dev, f_dev+indR[iR], fcL_indx2_dev, numx2, size_f_dev);

        for (int ix1 = 0; ix1 < fcL_numx1[iR]; ix1++) {
            int n = fcL_numx2[iR][ix1+1] - fcL_numx2[iR][ix1];

            cublasErrorHandle(cublasSetStream(handle_cublas, cudaStreams[ix1]));
            cublasErrorHandle(mycublasdot(handle_cublas, n, fcL_dev+fcL_numx2[iR][ix1], 1, f_temp_dev+fcL_numx2[iR][ix1], 1, df_dev+indR[iR]+size_f->nR*fcL_indx1[iR][ix1]));
        }

        printf("No. %d finished, total: %d\n", iR, numR);
    }

    // density out
    int* lambda_indx_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indx_dev, size_f->nx*sizeof(int)));

    for (int iR = 0; iR < numR; iR++) {
        cudaErrorHandle(cudaMemcpy(lambda_indx_dev, lambda_indx[iR], lambda_numx[iR]*sizeof(int), cudaMemcpyHostToDevice));
        get_fout <<<(int)lambda_numx[iR]/128+1, 128>>> (df_dev+indR[iR], f_dev+indR[iR], lambda[iR], lambda_indx_dev, lambda_numx[iR], size_f_dev);
    }

    cudaErrorHandle(cudaMemcpy(df, df_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(fcL_dev));
    cudaErrorHandle(cudaFree(fcL_indx2_dev));
    cudaErrorHandle(cudaFree(f_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(f_temp_dev));
    cudaErrorHandle(cudaFree(df_dev));
    cudaErrorHandle(cudaFree(lambda_indx_dev));

    cublasErrorHandle(cublasDestroy(handle_cublas));
}

void get_df_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f)
{
    // compute fin
    dim3 blocksize_n0Rx(size_f->const_2Bx, 1, 1);
    dim3 gridsize_n0Rx(size_f->const_2Bx, numR, 1);

    myReal* f_dev;
    cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(f_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, numR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(lambda_dev, lambda, numR*sizeof(myReal), cudaMemcpyHostToDevice));

    int* indR_dev;
    cudaErrorHandle(cudaMalloc(&indR_dev, numR*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(indR_dev, indR, numR*sizeof(int), cudaMemcpyHostToDevice));

    int* ind_interp_dev;
    cudaErrorHandle(cudaMalloc(&ind_interp_dev, 4*numR*size_f->nx*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(ind_interp_dev, ind_interp, 4*numR*size_f->nx*sizeof(int), cudaMemcpyHostToDevice));
    
    myReal* coeff_interp_dev;
    cudaErrorHandle(cudaMalloc(&coeff_interp_dev, 4*numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(coeff_interp_dev, coeff_interp, 4*numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
    
    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    myReal* df_dev;
    cudaErrorHandle(cudaMalloc(&df_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(df_dev, 0, size_f->nTot*sizeof(myReal)));

    get_fin_nonoise <<<gridsize_n0Rx, blocksize_n0Rx>>> (df_dev, f_dev, lambda_dev, indR_dev, ind_interp_dev, coeff_interp_dev, size_f_dev);    

    // compute fout
    int* lambda_indx_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indx_dev, size_f->nx*sizeof(int)));

    for (int iR = 0; iR < numR; iR++) {
        cudaErrorHandle(cudaMemcpy(lambda_indx_dev, lambda_indx[iR], lambda_numx[iR]*sizeof(int), cudaMemcpyHostToDevice));
        get_fout <<<(int)lambda_numx[iR]/128+1, 128>>> (df_dev+indR[iR], f_dev+indR[iR], lambda[iR], lambda_indx_dev, lambda_numx[iR], size_f_dev);
    }

    cudaErrorHandle(cudaMemcpy(df, df_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(f_dev));
    cudaErrorHandle(cudaFree(lambda_dev));
    cudaErrorHandle(cudaFree(indR_dev));
    cudaErrorHandle(cudaFree(ind_interp_dev));
    cudaErrorHandle(cudaFree(coeff_interp_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(df_dev));
    cudaErrorHandle(cudaFree(lambda_indx_dev));
}

__global__ void get_fold_noise(myReal* f_temp, const myReal* f, const int* fcL_indx2, const int fcL_numx2, const Size_f* size_f)
{
    int ind_temp = threadIdx.x + blockIdx.x*blockDim.x;

    if (ind_temp < fcL_numx2) {
        int indf = fcL_indx2[ind_temp]*size_f->nR;
        f_temp[ind_temp] = f[indf];
    }
}

__global__ void get_fin_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int* indR, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    int indfR = indR[blockIdx.y];
    int indf = indfR + indx*size_f->nR;
    int indInterp = 4*(blockIdx.y + indx*gridDim.y);

    if (isnan(coeff_interp[indInterp])) {
        df[indf] = 0;
    } else {
        int indf_interp[4];
        for (int i = 0; i < 4; i++) {
            indf_interp[i] = indfR + ind_interp[indInterp+i]*size_f->nR;
        }

        myReal f_interp = 0.0;
        for (int i = 0; i < 4; i++) {
            f_interp += f[indf_interp[i]]*coeff_interp[indInterp+i];
        }

        df[indf] = f_interp*lambda[blockIdx.y];
    }
}

__global__ void get_fout(myReal* df, const myReal* f, const myReal lambda, const int* lambda_indx, const int lambda_numx, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    if (indx < lambda_numx) {
        int indf = lambda_indx[indx]*size_f->nR;
        df[indf] = df[indf] - f[indf]*lambda;
    }
}


