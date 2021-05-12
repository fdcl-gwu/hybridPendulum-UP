#include "propagate.cuh"

#include <math.h>
#include <stdio.h>
#include "cublas_v2.h"

void get_df(myReal* df, const myReal* f, const myReal* x, const myReal* lambda, const myReal* Omega, const char* lambda_cat, const myReal* Gd, Size_f* size_f)
{
    // pre-calculations
    myReal detGd = Gd[0]*Gd[3] - Gd[2]*Gd[1];
    myReal c_normal = 1/(2*PI*mysqrt(detGd));

    myReal invGd[4];
    invGd[0] = Gd[3]/detGd;
    invGd[1] = -Gd[2]/detGd;
    invGd[2] = -Gd[1]/detGd;
    invGd[3] = Gd[0]/detGd;

    myReal dx2 = (x[2]-x[0]) * (x[2]-x[0]);

    // find nonzero lambda
    int* ind_n0 = (int*) malloc(size_f->nR*sizeof(int));
    int nn0 = 0;
    for (int iR = 0; iR < size_f->nR; iR++) {
        if (lambda_cat[iR] != 0) {
            ind_n0[nn0] = iR;
            nn0++;
        }
    }

    // calculate flambda
    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(lambda_dev, lambda, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* f_dev;
    cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(f_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));
    
    myReal* flambda_dev;
    cudaErrorHandle(cudaMalloc(&flambda_dev, size_f->nTot*sizeof(myReal)));

    dim3 blocksize_Rx(size_f->const_2BR, 1, 1);
    dim3 gridsize_Rx(size_f->const_2BRs, size_f->nx, 1);

    get_flambda <<<gridsize_Rx, blocksize_Rx>>> (flambda_dev, f_dev, lambda_dev, dx2, size_f_dev);

    // calculate df
    memset(df, 0, size_f->nTot*sizeof(myReal));

    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* Omega_dev;
    cudaErrorHandle(cudaMalloc(&Omega_dev, 2*nn0*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(Omega_dev, Omega, 2*nn0*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* invGd_dev;
    cudaErrorHandle(cudaMalloc(&invGd_dev, 4*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(invGd_dev, invGd, 4*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* fc_dev;
    cudaErrorHandle(cudaMalloc(&fc_dev, size_f->nx*sizeof(myReal)));

    myReal* df_dev;
    cudaErrorHandle(cudaMalloc(&df_dev, size_f->nTot*sizeof(myReal)));

    dim3 blocksize_x(size_f->const_2Bx, 1, 1);
    dim3 gridsize_x(size_f->const_2Bx, 1, 1);

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    for (int iR = 0; iR < nn0; iR++) {
        for (int ix = 0; ix < size_f->nx; ix++) {
            get_fc <<<gridsize_x, blocksize_x>>> (fc_dev, x_dev+2*ix, Omega_dev+2*iR, invGd_dev, c_normal, nn0, size_f_dev);

            cublasErrorHandle(mycublasdot(handle_cublas, size_f->nx, fc_dev, 1, flambda_dev+ind_n0[iR], size_f->nR, df_dev+ind_n0[iR]+ix*size_f->nR));
        }
    }

    sub_lambda_f <<<gridsize_Rx, blocksize_Rx>>> (df_dev, lambda_dev, f_dev, size_f_dev);

    cudaErrorHandle(cudaMemcpy(df, df_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(lambda_dev));
    cudaErrorHandle(cudaFree(f_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(flambda_dev));
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(Omega_dev));
    cudaErrorHandle(cudaFree(invGd_dev));
    cudaErrorHandle(cudaFree(fc_dev));
    cudaErrorHandle(cudaFree(df_dev));

    cublasErrorHandle(cublasDestroy(handle_cublas));

    free(ind_n0);
}

__global__ void get_fc(myReal* fc, const myReal* x, const myReal* Omega, const myReal* invGd, const myReal c, const int nn0, const Size_f* size_f)
{
    int ind_fc = threadIdx.x + blockIdx.x*size_f->const_2Bx;
    int ind_Omega = 2*nn0*ind_fc;

    myReal dOmega[2];
    dOmega[0] = x[0] - Omega[ind_Omega];
    dOmega[1] = x[1] - Omega[ind_Omega+1];

    myReal fc_local = dOmega[0]*invGd[0]*dOmega[0] + dOmega[1]*invGd[1]*dOmega[0] + dOmega[0]*invGd[2]*dOmega[1] + dOmega[1]*invGd[3]*dOmega[1];
    fc_local = c*myexp(-0.5*fc_local);

    fc[ind_fc] = fc_local;
}

__global__ void get_flambda(myReal* flambda, const myReal* f, const myReal* lambda, const myReal dx2, const Size_f* size_f)
{
    int ind = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->nR;

    flambda[ind] = f[ind] * lambda[ind] * dx2;
}

__global__ void sub_lambda_f(myReal* df, const myReal* lambda, const myReal* f, const Size_f* size_f)
{
    int ind = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->nR;

    df[ind] = df[ind] - lambda[ind]*f[ind];
}
