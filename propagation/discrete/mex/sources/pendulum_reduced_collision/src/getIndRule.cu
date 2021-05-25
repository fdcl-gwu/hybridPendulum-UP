#include "getIndRule.cuh"

#include <stdio.h>
#include <math.h>

void getIndRule(int* ind_interp, myReal* coeff_interp, const myReal* x, const myReal* Omega, const int numR, const Size_f* size_f)
{
    // interpoloation rule:
    // fx = coeff[0]*f00 + coeff[1]*f10 + coeff[2]*f01 + coeff[3]*f11

    // pre-calculation
    myReal dx = x[2]-x[0];
    myReal dx2 = dx*dx;
    myReal L = x[2*(size_f->const_2Bx-1)]-x[0] + dx;

    // compute indRule
    dim3 blocksize_n0Rx(size_f->const_2Bx, 1, 1);
    dim3 gridsize_n0Rx(size_f->const_2Bx, numR, 1);

    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* Omega_dev;
    cudaErrorHandle(cudaMalloc(&Omega_dev, 2*numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(Omega_dev, Omega, 2*numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    int* ind_interp_dev;
    cudaErrorHandle(cudaMalloc(&ind_interp_dev, 4*numR*size_f->nx*sizeof(int)));

    myReal* coeff_interp_dev;
    cudaErrorHandle(cudaMalloc(&coeff_interp_dev, 4*numR*size_f->nx*sizeof(myReal)));

    compute_indRule <<<gridsize_n0Rx, blocksize_n0Rx>>> (ind_interp_dev, coeff_interp_dev, x_dev, Omega_dev, L, dx2, size_f_dev);
    cudaErrorHandle(cudaMemcpy(ind_interp, ind_interp_dev, 4*numR*size_f->nx*sizeof(int), cudaMemcpyDeviceToHost));
    cudaErrorHandle(cudaMemcpy(coeff_interp, coeff_interp_dev, 4*numR*size_f->nx*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(Omega_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(ind_interp_dev));
    cudaErrorHandle(cudaFree(coeff_interp_dev));
}

__global__ void compute_indRule(int* ind_interp, myReal* coeff_interp, const myReal* x, const myReal* Omega, const myReal L, const myReal dx2, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    int indRx = blockIdx.y + indx*gridDim.y;
    int ind_Omega = 2*indRx;
    int ind_out = 4*indRx;

    myReal Omega_local[2] = {Omega[ind_Omega], Omega[ind_Omega+1]};

    if (isnan(Omega_local[0])) {
        for (int i = 0; i < 4; i++) {
            ind_interp[ind_out+i] = 0;
            coeff_interp[ind_out+i] = NAN;
        }
    } else {
        int x1_ind = (int) ((Omega_local[0]/L+0.5) * size_f->const_2Bx);
        int x2_ind = (int) ((Omega_local[1]/L+0.5) * size_f->const_2Bx);
        ind_interp[ind_out] = x1_ind+x2_ind*size_f->const_2Bx;
        ind_interp[ind_out+1] = ind_interp[ind_out]+1;
        ind_interp[ind_out+2] = ind_interp[ind_out]+size_f->const_2Bx;
        ind_interp[ind_out+3] = ind_interp[ind_out+2]+1;

        myReal x1_grid[2];
        myReal x2_grid[2];
        x1_grid[0] = x[2*x1_ind];
        x1_grid[1] = x[2*(x1_ind+1)];
        x2_grid[0] = x[2*x2_ind*size_f->const_2Bx+1];
        x2_grid[1] = x[2*(x2_ind+1)*size_f->const_2Bx+1];

        myReal dx_grid[4];
        dx_grid[0] = Omega_local[0]-x1_grid[0];
        dx_grid[1] = x1_grid[1]-Omega_local[0];
        dx_grid[2] = Omega_local[1]-x2_grid[0];
        dx_grid[3] = x2_grid[1]-Omega_local[1];

        coeff_interp[ind_out] = dx_grid[1]*dx_grid[3]/dx2;
        coeff_interp[ind_out+1] = dx_grid[0]*dx_grid[3]/dx2;
        coeff_interp[ind_out+2] = dx_grid[1]*dx_grid[2]/dx2;
        coeff_interp[ind_out+3] = dx_grid[0]*dx_grid[2]/dx2;
    }
}

