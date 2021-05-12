#include <math.h>
#include <stdio.h>

#include "getLambda.cuh"

void getLambda(myReal* lambda, char* lambda_cat, const myReal* R, const myReal* x, const myReal* d, const myReal* h, const myReal* r, const myReal* thetat, const myReal* lambda_max, const Size_f* size_f)
{
    // theta0
    myReal theta0;
    theta0 = myasin(*d/mysqrt(*h**h + *r**r)) - myasin(*r/mysqrt(*h**h+*r**r));

    // compute theta
    dim3 blocksize_R(size_f->const_2BR, 1, 1);
    dim3 gridsize_R(size_f->const_2BR, size_f->const_2BR, 1);
    
    myReal* R_dev;
    cudaErrorHandle(cudaMalloc(&R_dev, 9*size_f->nR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(R_dev, R, 9*size_f->nR*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* theta_dev;
    cudaErrorHandle(cudaMalloc(&theta_dev, size_f->nR*sizeof(myReal)));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    getTheta <<<gridsize_R, blocksize_R>>>(theta_dev, R_dev, size_f_dev);

    // compute PC
    myReal* PC_dev;
    cudaErrorHandle(cudaMalloc(&PC_dev, 3*size_f->nR*sizeof(myReal)));

    getPC <<<gridsize_R, blocksize_R>>> (PC_dev, R_dev, theta_dev, *h, *r, size_f_dev);

    // get labmda
    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, size_f->nTot*sizeof(myReal)));

    char* lambda_cat_dev;
    cudaErrorHandle(cudaMalloc(&lambda_cat_dev, size_f->nR*sizeof(char)));

    compute_lambda <<<gridsize_R, blocksize_R>>> (lambda_dev, lambda_cat_dev, theta_dev, theta0, *thetat, *lambda_max, size_f_dev);
    cudaErrorHandle(cudaMemcpy(lambda_cat, lambda_cat_dev, size_f->nR*sizeof(char), cudaMemcpyDeviceToHost));

    // expand lambda
    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    int* ind_n0 = (int*) malloc(size_f->nR*sizeof(int));
    int nn0 = 0;
    for (int iR = 0; iR < size_f->nR; iR++) {
        if (lambda_cat[iR] != 0) {
            ind_n0[nn0] = iR;
            nn0++;
        }
    }

    int* ind_n0_dev;
    cudaErrorHandle(cudaMalloc(&ind_n0_dev, nn0*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(ind_n0_dev, ind_n0, nn0*sizeof(int), cudaMemcpyHostToDevice));

    for (int ix = 1; ix < size_f->nx; ix++) {
        cudaErrorHandle(cudaMemcpy(lambda_dev+ix*size_f->nR, lambda_dev, size_f->nR*sizeof(myReal), cudaMemcpyDeviceToDevice));
    }

    dim3 blocksize_n0Rx(size_f->const_2Bx, 1, 1);
    dim3 gridsize_n0Rx(size_f->const_2Bx, nn0, 1);

    expand_lambda <<<gridsize_n0Rx, blocksize_n0Rx>>> (lambda_dev, R_dev, x_dev, PC_dev, ind_n0_dev, size_f_dev);
    cudaErrorHandle(cudaMemcpy(lambda, lambda_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(R_dev));
    cudaErrorHandle(cudaFree(theta_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(PC_dev));
    cudaErrorHandle(cudaFree(lambda_dev));
    cudaErrorHandle(cudaFree(lambda_cat_dev));
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(ind_n0_dev));

    free(ind_n0);
}

__global__ void getTheta(myReal* theta, const myReal* R, const Size_f* size_f)
{
    int indR = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->const_2BRs;

    myReal dot = R[9*indR+6];
    theta[indR] = myasin(dot);
}

__global__ void getPC(myReal* PC, const myReal* R, const myReal* theta, const myReal h, const myReal r, const Size_f* size_f)
{
    int indR = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->const_2BRs;

    myReal lr3 = h - r*mytan(theta[indR]);
    PC[3*indR] = lr3*R[9*indR+6] + r/mycos(theta[indR]);
    PC[3*indR+1] = lr3*R[9*indR+7];
    PC[3*indR+2] = lr3*R[9*indR+8];
}

__global__ void compute_lambda(myReal* lambda, char* lambda_cat, const myReal* theta, const myReal theta0, const myReal thetat, const myReal lambda_max, const Size_f* size_f)
{
    int indR = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->const_2BRs;

    if (theta[indR] < theta0 - thetat){
        lambda_cat[indR] = 0;
        lambda[indR] = 0;
    } else if (theta[indR] > theta0 + thetat) {
        lambda_cat[indR] = 2;
        lambda[indR] = lambda_max;
    } else {
        lambda_cat[indR] = 1;
        lambda[indR] = lambda_max/2*mysin(PI/(2*thetat)*(theta[indR]-theta0)) + lambda_max/2;
    }
}

__global__ void expand_lambda(myReal* lambda, const myReal* R, const myReal* x, const myReal* PC, const int* ind_n0, const Size_f* size_f)
{
    int indR = ind_n0[blockIdx.y];
    int indx = threadIdx.x + blockIdx.x*size_f->const_2Bx;
    int indtot = indR + indx*size_f->nR;

    int indR9 = indR*9;
    int indR3 = indR*3;
    indx = indx*2;

    myReal omega[2];
    omega[0] = R[indR9+1]*x[indx] + R[indR9+4]*x[indx+1];
    omega[1] = R[indR9+2]*x[indx] + R[indR9+5]*x[indx+1];

    myReal vC = omega[0]*PC[indR3+2] - omega[1]*PC[indR3+1];

    if (vC < 0.0) {
        lambda[indtot] = 0.0;
    }
}
