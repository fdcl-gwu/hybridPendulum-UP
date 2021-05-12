#include "getOmega.cuh"

#include <stdio.h>

void getOmega(myReal* Omega, const myReal* R, const myReal* x, const int* ind_n0, const int nn0, const myReal* epsilon, const Size_f* size_f)
{
    // get t
    myReal* R_dev;
    cudaErrorHandle(cudaMalloc(&R_dev, 9*size_f->nR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(R_dev, R, 9*size_f->nR*sizeof(myReal), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    myReal* t_dev;
    cudaErrorHandle(cudaMalloc(&t_dev, 2*size_f->nR*sizeof(myReal)));

    dim3 blocksize_R(size_f->const_2BR, 1, 1);
    dim3 gridsize_R(size_f->const_2BRs, 1, 1);

    getT <<<gridsize_R, blocksize_R>>> (t_dev, R_dev, size_f_dev);

    // get Omega
    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    int* ind_n0_dev;
    cudaErrorHandle(cudaMalloc(&ind_n0_dev, nn0*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(ind_n0_dev, ind_n0, nn0*sizeof(int), cudaMemcpyHostToDevice));

    myReal* Omega_dev;
    cudaErrorHandle(cudaMalloc(&Omega_dev, 2*nn0*size_f->nx*sizeof(myReal)));

    dim3 blocksize_n0Rx(size_f->const_2Bx, 1, 1);
    dim3 gridsize_n0Rx(size_f->const_2Bx, nn0, 1);

    compute_Omega <<<gridsize_n0Rx, blocksize_n0Rx>>> (Omega_dev, R_dev, x_dev, t_dev, ind_n0_dev, *epsilon, size_f_dev);

    cudaErrorHandle(cudaMemcpy(Omega, Omega_dev, 2*nn0*size_f->nx*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(R_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(t_dev));
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(ind_n0_dev));
    cudaErrorHandle(cudaFree(Omega_dev));
}

__global__ void getT(myReal* t, const myReal* R, const Size_f* size_f)
{
    int indR = threadIdx.x + blockIdx.x*size_f->const_2BR;

    int indR2 = indR*2;
    int indR9 = indR*9;

    myReal normT = mysqrt(R[indR9+7]*R[indR9+7] + R[indR9+8]*R[indR9+8]);
    t[indR2] = R[indR9+8]/normT;
    t[indR2+1] = -R[indR9+7]/normT;
}

__global__ void compute_Omega(myReal* Omega, const myReal* R, const myReal* x, const myReal* t, const int* ind_n0, const myReal epsilon, const Size_f* size_f)
{
    int indR = ind_n0[blockIdx.y];
    int indx = threadIdx.x + blockIdx.x*size_f->const_2Bx;
    int indTot = (blockIdx.y + indx*gridDim.y)*2;

    int indR9 = indR*9;
    int indR2 = indR*2;
    indx = indx*2;

    myReal omega[3];
    omega[0] = R[indR9]*x[indx] + R[indR9+3]*x[indx+1];
    omega[1] = R[indR9+1]*x[indx] + R[indR9+4]*x[indx+1];
    omega[2] = R[indR9+2]*x[indx] + R[indR9+5]*x[indx+1];

    myReal ot = (1.0+epsilon) * (omega[1]*t[indR2] + omega[2]*t[indR2+1]);

    omega[1] = omega[1] - ot*t[indR2];
    omega[2] = omega[2] - ot*t[indR2+1];

    Omega[indTot] = R[indR9]*omega[0] + R[indR9+1]*omega[1] + R[indR9+2]*omega[2];
    Omega[indTot+1] = R[indR9+3]*omega[0] + R[indR9+4]*omega[1] + R[indR9+5]*omega[2];
}

