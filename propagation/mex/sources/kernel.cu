
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include <math.h>
#include <chrono>

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    ///////////////////
    // set up arrays //
    ///////////////////

    // get Fold from matlab
    cuDoubleComplex* Fold = (cuDoubleComplex*) mxGetComplexDoubles(prhs[0]);
    const mwSize* size_Fold = mxGetDimensions(prhs[0]);

    Size_F size_F;
    init_Size_F(&size_F, (int)size_Fold[2], (int)size_Fold[3]/2);

    cuDoubleComplex* Fold_compact = new cuDoubleComplex[size_F.nTot_compact];
    modify_F(Fold, Fold_compact, true, &size_F);

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(size_F), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fnew_compact = new cuDoubleComplex[size_F.nTot_compact];
    
    // get X from matlab
    cuDoubleComplex* X = (cuDoubleComplex*) mxGetComplexDoubles(prhs[1]);

    cuDoubleComplex* X_dev;
    cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F.nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // get dt from matlab
    double* dt = mxGetDoubles(prhs[2]);

    // get u from matlab
    cuDoubleComplex* u = (cuDoubleComplex*) mxGetComplexDoubles(prhs[3]);

    cuDoubleComplex* u_compact = new cuDoubleComplex[3*size_F.nR_compact];
    modify_u(u, u_compact, &size_F);

    cuDoubleComplex* u_dev;
    cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F.nR_compact*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(u_dev, u_compact, 3*size_F.nR_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // get G1 from matlab
    double* G1 = mxGetDoubles(prhs[4]);

    // get G2 from matlab
    double* G2 = mxGetDoubles(prhs[5]);

    double* G2_dev;
    cudaErrorHandle(cudaMalloc(&G2_dev, 9*sizeof(double)));
    cudaErrorHandle(cudaMemcpy(G2_dev, G2, 9*sizeof(double), cudaMemcpyHostToDevice));

    // set up temporary variables for calculation
    cuDoubleComplex* dF = new cuDoubleComplex[3*size_F.nTot_compact];

    cuDoubleComplex* X_ijk_dev;
    cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));

    /////////////////////////////////
    // set up block and grid sizes //
    /////////////////////////////////

    // flip the circshift X
    dim3 blocksize_X(8, 8, 8);
    int gridnum_X = ceil((double) size_F.const_2Bx/8);
    dim3 gridsize_X(gridnum_X, gridnum_X, gridnum_X);

    // add
    dim3 blocksize_512(512, 1, 1);
    dim3 gridsize_512(ceil((double) size_F.nTot_splitx/512), 1, 1);

    ////////////////////
    // set up tensors //
    ////////////////////

    cuDoubleComplex* Fold_dev;
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nx * size_F.nR_split * sizeof(cuDoubleComplex)));

    cuDoubleComplex* dF_dev;
    cudaErrorHandle(cudaMalloc(&dF_dev, 3 * size_F.nR_split * sizeof(cuDoubleComplex)));

    cutensorHandle_t handle_cutensor;
    cutensorInit(&handle_cutensor);

    size_t worksize[2] = {0,0};
    cutensorContractionPlan_t plan[2];

    cutensor_initialize(&handle_cutensor, plan, worksize, Fold_dev, X_dev, dF_dev, size_F.nR_split, &size_F);
    cutensor_initialize(&handle_cutensor, plan+1, worksize+1, Fold_dev, X_dev, dF_dev, size_F.nR_remainder, &size_F);

    void* cutensor_workspace;
    cudaErrorHandle(cudaMalloc(&cutensor_workspace, worksize[0]));

    cuDoubleComplex alpha = make_cuDoubleComplex((double)1/size_F.nx,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);

    ///////////////////
    // set up cublas //
    ///////////////////

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    ///////////////
    // calculate //
    ///////////////

    // circular convolution
    // X_ijk = flip(flip(flip(X,1),2),3)
    // X_ijk = circshift(X_ijk,1,i)
    // X_ijk = circshift(X_ijk,2,j)
    // X_ijk = circshift(X_ijk,3,k)
    // dF{r,i,j,k,p} = Fold{r,m,n,l}.*X_ijk{m,n,l,p}

    permute_F(Fold_compact, false, &size_F);

    for (int is = 0; is < size_F.ns; is++) {
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        int nR_split;
        if (is == size_F.ns-1)
            nR_split = size_F.nR_remainder;
        else
            nR_split = size_F.nR_split;

        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact + is*size_F.nx*size_F.nR_split,
            size_F.nx*nR_split*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < size_F.const_2Bx; i++) {
            for (int j = 0; j < size_F.const_2Bx; j++) {
                for (int k = 0; k < size_F.const_2Bx; k++) {
                    flip_shift <<<gridsize_X, blocksize_X>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());

                    if (is == size_F.ns-1) {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, plan+1, &alpha, Fold_dev, X_ijk_dev,
                            &beta, dF_dev, dF_dev, cutensor_workspace, worksize[1], 0));
                    } else {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, plan, &alpha, Fold_dev, X_ijk_dev,
                            &beta, dF_dev, dF_dev, cutensor_workspace, worksize[0], 0));
                    }

                    for (int ip = 0; ip < 3; ip++) {
                        int ind_dF = is*size_F.nR_split + (i + j*size_F.const_2Bx + k*size_F.const_2Bxs)*size_F.nR_compact + ip*size_F.nTot_compact;
                        cudaMemcpy(dF+ind_dF, dF_dev+ip*nR_split, nR_split*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                    }
                }
            }
        }

        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1e6 << "[second]" << std::endl;
    }

    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(dF_dev));
    cudaErrorHandle(cudaFree(cutensor_workspace));

    permute_F(Fold_compact, true, &size_F);

    // multiply u
    // dF(indmn,indmn,l,i,j,k,p) = -dF(indmn,indmn,l,i,j,k,p)*u(indmn,indmn,l,p)'

    cuDoubleComplex alpha_cublas = make_cuDoubleComplex(-1,0);

    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cuDoubleComplex* dF_dev_result;
    cudaErrorHandle(cudaMalloc(&dF_dev_result, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    for (int ip = 0; ip < 3; ip++) {
        for (int k = 0; k < size_F.const_2Bx; k++) {
            int ind_dF = k*size_F.nTot_splitx + ip*size_F.nTot_compact;
            cudaErrorHandle(cudaMemcpy(dF_dev, dF+ind_dF, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            for (int l = 0; l <= size_F.lmax; l++)
            {
                int ind_dF_dev = l*(2*l-1)*(2*l+1)/3;
                long long int stride_dF = size_F.nR_compact;

                int ind_u = l*(2*l-1)*(2*l+1)/3 + ip*size_F.nR_compact;
                long long int stride_u = 0;

                cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                    &alpha_cublas, dF_dev+ind_dF_dev, 2*l+1, stride_dF,
                    u_dev+ind_u, 2*l+1, stride_u,
                    &beta, dF_dev_result+ind_dF_dev, 2*l+1, stride_dF, size_F.const_2Bxs));

                cudaDeviceSynchronize();
            }

            cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF_dev_result, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        }
    }

    cudaFree(dF_dev);
    cudaFree(dF_dev_result);

    // addup F
    // dF = sum(dF,'p')

    cudaErrorHandle(cudaMalloc(&dF_dev, 3*size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    for (int k = 0; k < size_F.const_2Bx; k++) {
        for (int ip = 0; ip < 3; ip++) {
            int ind_dF = k*size_F.nTot_splitx + ip*size_F.nTot_compact;
            int ind_dF_dev = ip*size_F.nTot_splitx;

            cudaErrorHandle(cudaMemcpy(dF_dev+ind_dF_dev, dF+ind_dF, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }

        addup_F <<<gridsize_512, blocksize_512>>> (dF_dev, size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        cudaDeviceSynchronize();

        int ind_dF = k*size_F.nTot_splitx;
        cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    cudaFree(dF_dev);

    // gyro random walk noise
    // dF_temp(indmn,indmn,l,nx,i,j) = dF_temp(indmn,indmn,l,nx)*u(indmn,indmn,l,i)'*u(indmn,indmn,l,j)'
    // dF = dF + sum(dF_temp,'i','j')

    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cuDoubleComplex* dF_dev_temp1;
    cudaErrorHandle(cudaMalloc(&dF_dev_temp1, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cuDoubleComplex* dF_dev_temp2;
    cudaErrorHandle(cudaMalloc(&dF_dev_temp2, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    for (int k = 0; k < size_F.const_2Bx; k++) {
        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact + k*size_F.nTot_splitx, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(dF_dev, dF + k*size_F.nTot_splitx, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int l = 0; l <= size_F.lmax; l++) {
                    int ind_F = l*(2*l-1)*(2*l+1)/3;
                    long long int stride_F = size_F.nR_compact;

                    int ind_u1 = l*(2*l-1)*(2*l+1)/3 + i*size_F.nR_compact;
                    int ind_u2 = l*(2*l-1)*(2*l+1)/3 + j*size_F.nR_compact;
                    long long int stride_u = 0;

                    alpha_cublas.x = 1;

                    cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                        &alpha_cublas, Fold_dev+ind_F, 2*l+1, stride_F,
                        u_dev+ind_u1, 2*l+1, stride_u,
                        &beta, dF_dev_temp1+ind_F, 2*l+1, stride_F, size_F.const_2Bxs));

                    cudaDeviceSynchronize();

                    alpha_cublas.x = G1[i+j*3];

                    cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                        &alpha_cublas, dF_dev_temp1+ind_F, 2*l+1, stride_F,
                        u_dev+ind_u2, 2*l+1, stride_u,
                        &beta, dF_dev_temp2+ind_F, 2*l+1, stride_F, size_F.const_2Bxs));

                    cudaDeviceSynchronize();
                }

                add_F <<<gridsize_512, blocksize_512>>> (dF_dev, dF_dev_temp2, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                cudaDeviceSynchronize();
            }
        }

        cudaMemcpy(dF + k*size_F.nTot_splitx, dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }

    cudaFree(Fold_dev);
    cudaFree(dF_dev);
    cudaFree(dF_dev_temp1);
    cudaFree(dF_dev_temp2);

    // bias random walk noise
    // take derivative of the linear part
    double* c_dev;
    cudaErrorHandle(cudaMalloc(&c_dev, 9*size_F.const_2Bxs*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&dF_dev_temp1, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int ind_c = (i+3*j)*size_F.const_2Bxs;
            if (i == j) {
                dim3 blocksize_c(size_F.const_2Bx, 1, 1);
                get_c <<<1, blocksize_c>>> (c_dev+ind_c, i, j, G2_dev, size_F_dev);
                cudaErrorHandle(cudaGetLastError());
            }
            else {
                dim3 blocksize_c(size_F.const_2Bx, size_F.const_2Bx, 1);
                get_c <<<1, blocksize_c>>> (c_dev+ind_c, i, j, G2_dev, size_F_dev);
                cudaErrorHandle(cudaGetLastError());
            }
        }
    }

    dim3 blocksize_512_nRnx(512,1,1);
    dim3 gridsize_512_nRnx(ceil((float) size_F.nR_compact/512), size_F.const_2Bx, size_F.const_2Bx);

    for (int k = 0; k < size_F.const_2Bx; k++) {
        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact + k*size_F.nTot_splitx, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(dF_dev, dF + k*size_F.nTot_splitx, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int ind_c = (i+3*j)*size_F.const_2Bxs;
                get_biasRW <<<gridsize_512_nRnx, blocksize_512_nRnx>>> (dF_dev_temp1, Fold_dev, c_dev+ind_c, i, j, k, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                add_F <<<gridsize_512, blocksize_512>>> (dF_dev, dF_dev_temp1, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                cudaDeviceSynchronize();
            }
        }

        cudaMemcpy(dF + k*size_F.nTot_splitx, dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    }

    cudaFree(c_dev);
    cudaFree(Fold_dev);
    cudaFree(dF_dev);
    cudaFree(dF_dev_temp1);

    // Fnew = Fold + dt*dF

    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cuDoubleComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    for (int k = 0; k < size_F.const_2Bx; k++) {
        int ind_F = k*size_F.nTot_splitx;
        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(dF_dev, dF+ind_F, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512, blocksize_512>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        cudaDeviceSynchronize();

        cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    cudaFree(Fold_dev);
    cudaFree(dF_dev);
    cudaFree(Fnew_dev);

    // gather Fnew
    modify_F(Fnew_compact, Fnew, false, &size_F);

    // free memeory
    cudaErrorHandle(cudaFree(X_dev));
    cudaErrorHandle(cudaFree(X_ijk_dev));
    cudaErrorHandle(cudaFree(u_dev));
    cudaErrorHandle(cudaFree(size_F_dev));
    cudaErrorHandle(cudaFree(G2_dev));

    delete[] Fold_compact;
    delete[] Fnew_compact;
    delete[] u_compact;
    delete[] dF;
}

