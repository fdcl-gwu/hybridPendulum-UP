#include "getFcL.cuh"

#include <stdio.h>
#include <math.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

void getFcL(myReal* fcL, short* fcL_indx, const myReal* x, const myReal* Omega, const myReal* lambda, const int numR, const myReal* Gd, const int nD, const Size_f* size_f)
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

    // calculate fc
    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
    
    myReal* Omega_dev;
    cudaErrorHandle(cudaMalloc(&Omega_dev, 2*numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(Omega_dev, Omega, 2*numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
    
    myReal* invGd_dev;
    cudaErrorHandle(cudaMalloc(&invGd_dev, 4*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(invGd_dev, invGd, 4*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* fc_x2_dev;
    cudaErrorHandle(cudaMalloc(&fc_x2_dev, size_f->nx*sizeof(myReal)));
    thrust::device_ptr<myReal> fc_x2_thr(fc_x2_dev);

    short* indx2 = (short*) malloc(size_f->nx*sizeof(short));
    for (int i = 0; i < size_f->nx; i++) {
        indx2[i] = i;
    }

    short* indx2_dev;
    cudaErrorHandle(cudaMalloc(&indx2_dev, size_f->nx*sizeof(short)));
    cudaErrorHandle(cudaMemcpy(indx2_dev, indx2, size_f->nx*sizeof(short), cudaMemcpyHostToDevice));

    short* indx2_sort_dev;
    cudaErrorHandle(cudaMalloc(&indx2_sort_dev, size_f->nx*sizeof(short)));
    thrust::device_ptr<short> indx2_sort_thr(indx2_sort_dev);

    myReal* fcL_dev;
    cudaErrorHandle(cudaMalloc(&fcL_dev, nD*numR*size_f->nx*sizeof(myReal)));

    short* fcL_indx_dev;
    cudaErrorHandle(cudaMalloc(&fcL_indx_dev, nD*numR*size_f->nx*sizeof(short)));

    myReal* fc_x2 = (myReal*) malloc(nD*sizeof(myReal));
    short* fc_indx2 = (short*) malloc(nD*sizeof(short));
    myReal* fc_x1x2 = (myReal*) malloc(size_f->nx*size_f->nx*sizeof(myReal));
    short* fc_numx1 = (short*) malloc(size_f->nx*sizeof(short));
    myReal* fc_normal = (myReal*) malloc(size_f->nx*sizeof(myReal));

    myReal* fc_normal_dev;
    cudaErrorHandle(cudaMalloc(&fc_normal_dev, size_f->nx*sizeof(myReal)));

    for (int iR = 0; iR < numR; iR++) {
        memset(fc_numx1, 0, size_f->nx*sizeof(short));

        for (int ix1 = 0; ix1 < size_f->nx; ix1++) {
            // compute all densities for a given R and Omega^+
            get_fc <<<size_f->const_2Bx, size_f->const_2Bx>>> (fc_x2_dev, x_dev+2*ix1, Omega_dev+2*iR, invGd_dev, numR, c_normal);
            
            // find the largest nD densities
            cudaErrorHandle(cudaMemcpy(indx2_sort_dev, indx2_dev, size_f->nx*sizeof(short), cudaMemcpyDeviceToDevice));
            thrust::stable_sort_by_key(fc_x2_thr, fc_x2_thr+size_f->nx, indx2_sort_thr, thrust::greater<myReal>());

            cudaErrorHandle(cudaMemcpy(fcL_dev+iR*nD+ix1*nD*numR, fc_x2_dev, nD*sizeof(myReal), cudaMemcpyDeviceToDevice));
            cudaErrorHandle(cudaMemcpy(fcL_indx_dev+iR*nD+ix1*nD*numR, indx2_sort_dev, nD*sizeof(short), cudaMemcpyDeviceToDevice));

            cudaErrorHandle(cudaMemcpy(fc_x2, fc_x2_dev, nD*sizeof(myReal), cudaMemcpyDeviceToHost));
            cudaErrorHandle(cudaMemcpy(fc_indx2, indx2_sort_dev, nD*sizeof(short), cudaMemcpyDeviceToHost));

            // prepare for normalization
            for (int i = 0; i < nD; i++) {
                fc_x1x2[fc_numx1[fc_indx2[i]] + fc_indx2[i]*size_f->nx] = fc_x2[i];
                fc_numx1[fc_indx2[i]]++;
            }
        }

        // normalization
        for (int ix2 = 0; ix2 < size_f->nx; ix2++) {
            fc_normal[ix2] = thrust::reduce(fc_x1x2+ix2*size_f->nx, fc_x1x2+ix2*size_f->nx+fc_numx1[ix2]);
            fc_normal[ix2] = fc_normal[ix2]*dx2;
        }
        
        cudaErrorHandle(cudaMemcpy(fc_normal_dev, fc_normal, size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
        get_fc_normal <<<size_f->nx, nD>>> (fcL_dev+iR*nD, fc_normal_dev, fcL_indx_dev+iR*nD, numR);

        printf("No. %d finished, total: %d\n", iR+1, numR);
    }

    // calculate fc*lambda
    dim3 blocksize_fcL(nD, size_f->const_2Bx, 1);
    dim3 gridsize_fcL(size_f->const_2Bx, numR, 1);

    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(lambda_dev, lambda, numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    get_fcL <<<gridsize_fcL, blocksize_fcL>>> (fcL_dev, lambda_dev, fcL_indx_dev, dx2);
    cudaErrorHandle(cudaMemcpy(fcL, fcL_dev, nD*numR*size_f->nx*sizeof(myReal), cudaMemcpyDeviceToHost));
    cudaErrorHandle(cudaMemcpy(fcL_indx, fcL_indx_dev, nD*numR*size_f->nx*sizeof(short), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(Omega_dev));
    cudaErrorHandle(cudaFree(invGd_dev));
    cudaErrorHandle(cudaFree(fc_x2_dev));
    cudaErrorHandle(cudaFree(indx2_dev));
    cudaErrorHandle(cudaFree(indx2_sort_dev));
    cudaErrorHandle(cudaFree(fcL_dev));
    cudaErrorHandle(cudaFree(fcL_indx_dev));
    cudaErrorHandle(cudaFree(lambda_dev));

    free(indx2);
    free(fc_x2);
    free(fc_indx2);
    free(fc_x1x2);
    free(fc_numx1);
    free(fc_normal);
}

__global__ void get_fc(myReal* fc_x2, const myReal* x, const myReal* Omega, const myReal* invGd, const int numR, const myReal c_normal)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    int indOmega = 2*numR*indx;
    
    myReal dOmega[2];
    dOmega[0] = x[0] - Omega[indOmega];
    dOmega[1] = x[1] - Omega[indOmega+1];

    myReal fc_local = invGd[0]*dOmega[0]*dOmega[0] + (invGd[1]+invGd[2])*dOmega[0]*dOmega[1] + invGd[3]*dOmega[1]*dOmega[1];
    fc_local = myexp(-0.5*fc_local)*c_normal;

    fc_x2[indx] = fc_local;
}

__global__ void get_fc_normal(myReal* fc, const myReal* fc_normal, const short* fc_indx, const int numR)
{
    int ind_fc = threadIdx.x + blockIdx.x*blockDim.x*numR;
    int ind_normal = fc_indx[ind_fc];

    fc[ind_fc] = fc[ind_fc] / fc_normal[ind_normal];
}

__global__ void get_fcL(myReal* fcL, const myReal* lambda, const short* fcL_indx, const myReal dx2)
{
    int ind_fcL = threadIdx.x + blockIdx.y*blockDim.x + (threadIdx.y+blockIdx.x*blockDim.y)*blockDim.x*gridDim.y;
    int ind_lambda = blockIdx.y + fcL_indx[ind_fcL]*gridDim.y;

    fcL[ind_fcL] *= (lambda[ind_lambda]*dx2);
}

