#include "propagate.cuh"

#include <math.h>
#include <stdio.h>

void get_df(myReal* df, const myReal* f, const myReal* lambda, myReal** const* fcL, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, int* const* fcL_indx1, const int* fcL_numx1, int** const* fcL_indx2, int* const* fcL_numx2, const Size_f* size_f)
{
    // density in
    myReal* fcL_dev;
    cudaErrorHandle(cudaMalloc(&fcL_dev, size_f->nx*sizeof(myReal)));

    int* fcL_indx2_dev;
    cudaErrorHandle(cudaMalloc(&fcL_indx2_dev, size_f->nx*sizeof(int)));

    myReal* f_dev;
    cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(f_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    myReal* f_temp_dev;
    cudaErrorHandle(cudaMalloc(&f_temp_dev, size_f->nx*sizeof(myReal)));

    myReal* df_dev;
    cudaErrorHandle(cudaMalloc(&df_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(df_dev, 0, size_f->nTot*sizeof(myReal)));

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    for (int iR = 0; iR < numR; iR++) {
        for (int ix1 = 0; ix1 < fcL_numx1[iR]; ix1++) {
            cudaErrorHandle(cudaMemcpy(fcL_dev, fcL[iR][ix1], fcL_numx2[iR][ix1]*sizeof(myReal), cudaMemcpyHostToDevice));
            cudaErrorHandle(cudaMemcpy(fcL_indx2_dev, fcL_indx2[iR][ix1], fcL_numx2[iR][ix1]*sizeof(int), cudaMemcpyHostToDevice));

            get_fin <<<1, fcL_numx2[iR][ix1]>>> (f_temp_dev, f_dev+indR[iR], fcL_indx2_dev, size_f_dev);
            cublasErrorHandle(mycublasdot(handle_cublas, fcL_numx2[iR][ix1], fcL_dev, 1, f_temp_dev, 1, df_dev+indR[iR]+size_f->nR*fcL_indx1[iR][ix1]));
        }

        printf("fin: No. %d finished\n", iR+1);
    }

    // density out
    int* lambda_indx_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indx_dev, size_f->nx*sizeof(int)));

    for (int iR = 0; iR < numR; iR++) {
        cudaErrorHandle(cudaMemcpy(lambda_indx_dev, lambda_indx[iR], lambda_numx[iR]*sizeof(int), cudaMemcpyHostToDevice));
        get_fout <<<(int)lambda_numx[iR]/512+1, 512>>> (df_dev+indR[iR], f_dev+indR[iR], lambda[iR], lambda_indx_dev, lambda_numx[iR], size_f_dev);
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

__global__ void get_fin(myReal* f_temp, const myReal* f, const int* fcL_indx2, const Size_f* size_f)
{
    int indf = fcL_indx2[threadIdx.x]*size_f->nR;
    f_temp[threadIdx.x] = f[indf];
}

__global__ void get_fout(myReal* df, const myReal* f, const myReal lambda, const int* lambda_indx, const int lambda_numx, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    if (indx < lambda_numx) {
        int indf = lambda_indx[indx]*size_f->nR;
        df[indf] = df[indf] - f[indf]*lambda;
    }
}


