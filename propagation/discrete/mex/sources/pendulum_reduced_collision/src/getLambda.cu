#include <math.h>
#include <stdio.h>

#include "getLambda.cuh"

void getLambda(myReal* lambda, int* lambda_indR, int* lambda_numR, int** lambda_indx, int* lambda_numx, const myReal* R, const myReal* x, const myReal* d, const myReal* h, const myReal* r, const myReal* thetat, const myReal* lambda_max, const Size_f* size_f)
{
    // theta0
    myReal theta0;
    theta0 = myasin(*d/mysqrt(*h**h + *r**r)) - myasin(*r/mysqrt(*h**h+*r**r));

    // compute theta
    dim3 blocksize_R(size_f->const_2BR, 1, 1);
    dim3 gridsize_R(size_f->const_2BRs, 1, 1);
    
    myReal* R_dev;
    cudaErrorHandle(cudaMalloc(&R_dev, 9*size_f->nR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(R_dev, R, 9*size_f->nR*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* theta_dev;
    cudaErrorHandle(cudaMalloc(&theta_dev, size_f->nR*sizeof(myReal)));

    char* lambda_cat_dev;
    cudaErrorHandle(cudaMalloc(&lambda_cat_dev, size_f->nR*sizeof(char)));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    getTheta <<<gridsize_R, blocksize_R>>>(theta_dev, lambda_cat_dev, R_dev, theta0, *thetat, size_f_dev);

    // get lambda_indR
    char* lambda_cat = (char*) malloc(size_f->nR*sizeof(char));
    cudaErrorHandle(cudaMemcpy(lambda_cat, lambda_cat_dev, size_f->nR*sizeof(char), cudaMemcpyDeviceToHost));

    *lambda_numR = 0;
    for (int iR = 0; iR < size_f->nR; iR++) {
        if (lambda_cat[iR] != 0) {
            lambda_indR[*lambda_numR] = iR;
            (*lambda_numR)++;
        }
    }

    // compute PC
    myReal* PC_dev;
    cudaErrorHandle(cudaMalloc(&PC_dev, 3*size_f->nR*sizeof(myReal)));

    getPC <<<gridsize_R, blocksize_R>>> (PC_dev, R_dev, theta_dev, *h, *r, size_f_dev);

    // get labmda
    dim3 blocksize_indR(512, 1, 1);
    dim3 gridsize_indR((int) *lambda_numR/512+1, 1, 1);
    
    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, (*lambda_numR)*sizeof(myReal)));

    int* lambda_indR_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indR_dev, (*lambda_numR)*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(lambda_indR_dev, lambda_indR, (*lambda_numR)*sizeof(int), cudaMemcpyHostToDevice));

    compute_lambda <<<gridsize_indR, blocksize_indR>>> (lambda_dev, lambda_cat_dev, lambda_indR_dev, theta_dev, theta0, *thetat, *lambda_max, *lambda_numR);

    cudaErrorHandle(cudaMemcpy(lambda, lambda_dev, (*lambda_numR)*sizeof(myReal), cudaMemcpyDeviceToHost));

    // get lambda_indx
    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    int* lambda_indx_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indx_dev, size_f->nx*sizeof(int)));

    dim3 blocksize_x(size_f->const_2Bx, 1, 1);
    dim3 gridsize_x(size_f->const_2Bx, 1, 1);

    for (int iR = 0; iR < *lambda_numR; iR++) {
        expand_lambda <<<gridsize_x, blocksize_x>>> (lambda_indx_dev, R_dev+lambda_indR[iR]*9, x_dev, PC_dev+lambda_indR[iR]*3, size_f_dev);
        
        cudaErrorHandle(cudaMemcpy(lambda_indx[iR], lambda_indx_dev, size_f->nx*sizeof(int), cudaMemcpyDeviceToHost));

        for (int ix = 0; ix < size_f->nx; ix++) {
            if (lambda_indx[iR][ix] == 1) {
                lambda_indx[iR][lambda_numx[iR]] = ix;
                lambda_numx[iR]++;
            }
        }
    }

    // free memory
    cudaErrorHandle(cudaFree(R_dev));
    cudaErrorHandle(cudaFree(theta_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(PC_dev));
    cudaErrorHandle(cudaFree(lambda_dev));
    cudaErrorHandle(cudaFree(lambda_cat_dev));
    cudaErrorHandle(cudaFree(lambda_indR_dev));
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(lambda_indx_dev));

    free(lambda_cat);
}

__global__ void getTheta(myReal* theta, char* lambda_cat, const myReal* R, const myReal theta0, const myReal thetat, const Size_f* size_f)
{
    int indR = threadIdx.x + blockIdx.x*size_f->const_2BR;

    myReal theta_local = myasin(R[9*indR+6]);
    theta[indR] = theta_local;
    
    if (theta[indR] < theta0 - thetat){
        lambda_cat[indR] = 0;
    } else if (theta[indR] > theta0 + thetat) {
        lambda_cat[indR] = 2;
    } else {
        lambda_cat[indR] = 1;
    }
}

__global__ void getPC(myReal* PC, const myReal* R, const myReal* theta, const myReal h, const myReal r, const Size_f* size_f)
{
    int indR = threadIdx.x + blockIdx.x*size_f->const_2BR;
    
    myReal lr3 = h - r*mytan(theta[indR]);
    PC[3*indR] = lr3*R[9*indR+6] + r/mycos(theta[indR]);
    PC[3*indR+1] = lr3*R[9*indR+7];
    PC[3*indR+2] = lr3*R[9*indR+8];
}

__global__ void compute_lambda(myReal* lambda, const char* lambda_cat, const int* lambda_indR, const myReal* theta, const myReal theta0, const myReal thetat, const myReal lambda_max, const int numR)
{
    int ind_lambda = threadIdx.x + blockIdx.x*blockDim.x;
    int indR = lambda_indR[ind_lambda];

    if (ind_lambda < numR) {
        if (lambda_cat[indR] == 0){
            lambda[ind_lambda] = 0;
        } else if (lambda_cat[indR] == 2) {
            lambda[ind_lambda] = lambda_max;
        } else {
            lambda[ind_lambda] = lambda_max/2*mysin(PI/(2*thetat)*(theta[indR]-theta0)) + lambda_max/2;
        }
    }
}

__global__ void expand_lambda(int* lambda_indx, const myReal* R, const myReal* x, const myReal* PC, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*size_f->const_2Bx;
    int indx2 = indx*2;

    myReal omega[2];
    omega[0] = R[1]*x[indx2] + R[4]*x[indx2+1];
    omega[1] = R[2]*x[indx2] + R[5]*x[indx2+1];

    myReal vC = omega[0]*PC[2] - omega[1]*PC[1];

    lambda_indx[indx] = (vC < 0.0) ? 0 : 1;
}
