#include "getFcL.cuh"

#include <stdio.h>
#include <math.h>

void getFcL(myReal*** fcL, int** fcL_indx1, int* fcL_numx1, int*** fcL_indx2, int** fcL_numx2, const myReal* x, const myReal* Omega, const myReal* lambda, const int nn0, int* const* lambda_indx, const int* lambda_numx, const myReal* Gd, const Size_f* size_f)
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

    // calculate fc*lambda
    myReal* x_dev;
    cudaErrorHandle(cudaMalloc(&x_dev, 2*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(x_dev, x, 2*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
    
    myReal* Omega_dev;
    cudaErrorHandle(cudaMalloc(&Omega_dev, 2*nn0*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(Omega_dev, Omega, 2*nn0*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
    
    myReal* invGd_dev;
    cudaErrorHandle(cudaMalloc(&invGd_dev, 4*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(invGd_dev, invGd, 4*sizeof(myReal), cudaMemcpyHostToDevice));

    int* lambda_indx_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indx_dev, size_f->nx*sizeof(int)));

    myReal* fcL_x_dev;
    cudaErrorHandle(cudaMalloc(&fcL_x_dev, size_f->nx*sizeof(myReal)));

    myReal* fcL_x_temp = (myReal*) malloc(size_f->nx*sizeof(myReal));

    for (int iR = 0; iR < nn0; iR++) {
        dim3 blocksize_x(256, 1, 1);
        dim3 gridsize_x((int)lambda_numx[iR]/256+1, 1, 1);

        cudaErrorHandle(cudaMemcpy(lambda_indx_dev, lambda_indx[iR], lambda_numx[iR]*sizeof(int), cudaMemcpyHostToDevice));

        fcL[iR] = (myReal**) malloc(size_f->nx*sizeof(myReal*));

        fcL_indx1[iR] = (int*) malloc(size_f->nx*sizeof(int));
        fcL_indx2[iR] = (int**) malloc(size_f->nx*sizeof(int*));

        fcL_numx1[iR] = 0;
        fcL_numx2[iR] = (int*) malloc(size_f->nx*sizeof(int));

        for (int ix = 0; ix < size_f->nx; ix++) {
            get_fcL_x <<<blocksize_x, gridsize_x>>> (fcL_x_dev, x_dev+2*ix, Omega_dev+2*iR, lambda[iR], lambda_indx_dev, invGd_dev, nn0, lambda_numx[iR], dx2, c_normal);
            
            fcL[iR][fcL_numx1[iR]] = (myReal*) malloc(lambda_numx[iR]*sizeof(myReal));
            fcL_indx2[iR][fcL_numx1[iR]] = (int*) malloc(lambda_numx[iR]*sizeof(int));

            cudaErrorHandle(cudaMemcpy(fcL_x_temp, fcL_x_dev, lambda_numx[iR]*sizeof(myReal), cudaMemcpyDeviceToHost));

            fcL_numx2[iR][fcL_numx1[iR]] = 0;
            for (int ix2 = 0; ix2 < lambda_numx[iR]; ix2++) {
                if (fcL_x_temp[ix2] > 1e-6) {
                    fcL[iR][fcL_numx1[iR]][fcL_numx2[iR][fcL_numx1[iR]]] = fcL_x_temp[ix2];
                    fcL_indx2[iR][fcL_numx1[iR]][fcL_numx2[iR][fcL_numx1[iR]]] = lambda_indx[iR][ix2];
                    fcL_numx2[iR][fcL_numx1[iR]]++;
                }
            }
            
            if (fcL_numx2[iR][fcL_numx1[iR]] == 0) {
                free(fcL[iR][fcL_numx1[iR]]);
                free(fcL_indx2[iR][fcL_numx1[iR]]);
            } else {
                fcL[iR][fcL_numx1[iR]] = (myReal*) realloc(fcL[iR][fcL_numx1[iR]], fcL_numx2[iR][fcL_numx1[iR]]*sizeof(myReal));
                fcL_indx2[iR][fcL_numx1[iR]] = (int*) realloc(fcL_indx2[iR][fcL_numx1[iR]], fcL_numx2[iR][fcL_numx1[iR]]*sizeof(int));
                
                fcL_indx1[iR][fcL_numx1[iR]] = ix;
                fcL_numx1[iR]++;
            }
        }

        fcL[iR] = (myReal**) realloc(fcL[iR], fcL_numx1[iR]*sizeof(myReal*));
        fcL_indx1[iR] = (int*) realloc(fcL_indx1[iR], fcL_numx1[iR]*sizeof(int));
        fcL_indx2[iR] = (int**) realloc(fcL_indx2[iR], fcL_numx1[iR]*sizeof(int*));
        fcL_numx2[iR] = (int*) realloc(fcL_numx2[iR], fcL_numx1[iR]*sizeof(int));

        printf("No. %d finished, totol: %d.\n", iR+1, nn0);
    }

    // free memory
    cudaErrorHandle(cudaFree(x_dev));
    cudaErrorHandle(cudaFree(Omega_dev));
    cudaErrorHandle(cudaFree(invGd_dev));
    cudaErrorHandle(cudaFree(lambda_indx_dev));
    cudaErrorHandle(cudaFree(fcL_x_dev));

    free(fcL_x_temp);
}

__global__ void get_fcL_x(myReal* fcL_x, const myReal* x, const myReal* Omega, const myReal lambda, const int* lambda_indx, const myReal* invGd, const int nn0, const int lambda_numx, const myReal dx2, const myReal c_normal)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (indx < lambda_numx) {
        int indOmega = 2*lambda_indx[indx]*nn0;

        myReal dOmega[2];
        dOmega[0] = x[0] - Omega[indOmega];
        dOmega[1] = x[1] - Omega[indOmega+1];

        myReal fc_local = invGd[0]*dOmega[0]*dOmega[0] + (invGd[1]+invGd[2])*dOmega[0]*dOmega[1] + invGd[3]*dOmega[1]*dOmega[1];
        fc_local = myexp(-0.5*fc_local)*c_normal;
        fc_local = fc_local*lambda*dx2;

        fcL_x[indx] = fc_local;
    }
}

