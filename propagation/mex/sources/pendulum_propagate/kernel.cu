
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include <math.h>

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    ////////////////////////////
    // get arrays from Matlab //
    ////////////////////////////

    // get Fold from matlab
    cuDoubleComplex* Fold = (cuDoubleComplex*) mxGetComplexDoubles(prhs[0]);
    const mwSize* size_Fold = mxGetDimensions(prhs[0]);

    Size_F size_F;
    init_Size_F(&size_F, (int)size_Fold[2], (int)size_Fold[3]/2);

    cuDoubleComplex* Fold_compact = new cuDoubleComplex[size_F.nTot_compact];
    modify_F(Fold, Fold_compact, true, &size_F);

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(Size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(Size_F), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fnew_compact = new cuDoubleComplex[size_F.nTot_compact];
    
    // get X from matlab
    cuDoubleComplex* X = (cuDoubleComplex*) mxGetComplexDoubles(prhs[1]);

    // get OJO from matlab
    cuDoubleComplex* OJO = (cuDoubleComplex*) mxGetComplexDoubles(prhs[2]);

    // get MR from matlab
    cuDoubleComplex* MR = (cuDoubleComplex*) mxGetComplexDoubles(prhs[3]);

    cuDoubleComplex* MR_compact = new cuDoubleComplex[3*size_F.nR_compact];
    modify_u(MR, MR_compact, &size_F);

    // get dt from matlab
    double* dt = mxGetDoubles(prhs[4]);

    // get L from matlab
    double* L = mxGetDoubles(prhs[5]);

    // get u from matlab
    cuDoubleComplex* u = (cuDoubleComplex*) mxGetComplexDoubles(prhs[6]);

    cuDoubleComplex* u_compact = new cuDoubleComplex[3*size_F.nR_compact];
    modify_u(u, u_compact, &size_F);

    // get CG from matlab
    double** CG = new double* [size_F.BR*size_F.BR];
    for (int l1 = 0; l1 < size_F.BR; l1++) {
        for (int l2 = 0; l2 < size_F.BR; l2++) {
            int ind_CG = l1+l2*size_F.BR;
            CG[ind_CG] = mxGetDoubles(mxGetCell(prhs[7], ind_CG));
        }
    }

    ////////////////////////////
    // circular_convolution X //
    ////////////////////////////

    // X_ijk = flip(flip(flip(X,1),2),3)
    // X_ijk = circshift(X_ijk,1,i)
    // X_ijk = circshift(X_ijk,2,j)
    // X_ijk = circshift(X_ijk,3,k)
    // dF{r,i,j,k,p} = Fold{r,m,n,l}.*X_ijk{m,n,l,p}

    std::cout << "Circular convolution with X begin\n";

    // set up GPU arrays
    cuDoubleComplex* Fold_dev;
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nx*size_F.nR_split * sizeof(cuDoubleComplex)));

    cuDoubleComplex* X_dev;
    cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F.nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* X_ijk_dev;
    cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));

    cuDoubleComplex* dF3_dev;
    cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F.nR_split*sizeof(cuDoubleComplex)));

    // set up CPU arrays
    permute_F(Fold_compact, false, &size_F);

    cuDoubleComplex* dF3 = new cuDoubleComplex[3*size_F.nTot_compact];

    // set up cutensor
    cutensorHandle_t handle_cutensor;
    cutensorInit(&handle_cutensor);

    cutensorContractionPlan_t plan[2];
    size_t worksize[2] = {0,0};

    cutensor_initialize(&handle_cutensor, &plan[0], &worksize[0], Fold_dev, X_ijk_dev, dF3_dev, size_F.nR_split, &size_F);
    cutensor_initialize(&handle_cutensor, &plan[1], &worksize[1], Fold_dev, X_ijk_dev, dF3_dev, size_F.nR_remainder, &size_F);

    void* cutensor_workspace;
    cudaErrorHandle(cudaMalloc(&cutensor_workspace, worksize[0]));

    cuDoubleComplex alpha_cutensor = make_cuDoubleComplex(-(double)1/size_F.nx,0);
    cuDoubleComplex beta_cutensor = make_cuDoubleComplex(0,0);

    // set up blocksize and gridsize
    dim3 blocksize_8(8, 8, 8);
    int gridnum_8 = ceil((double) size_F.const_2Bx/8);
    dim3 gridsize_8(gridnum_8, gridnum_8, gridnum_8);

    // calculate
    for (int is = 0; is < size_F.ns; is++) {
        int nR_split;
        if (is == size_F.ns-1)
            nR_split = size_F.nR_remainder;
        else
            nR_split = size_F.nR_split;

        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact + is*size_F.nx*size_F.nR_split, size_F.nx*nR_split*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < size_F.const_2Bx; i++) {
            for (int j = 0; j < size_F.const_2Bx; j++) {
                for (int k = 0; k < size_F.const_2Bx; k++) {
                    flip_shift <<<gridsize_8, blocksize_8>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());

                    if (is == size_F.ns-1) {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan[1], &alpha_cutensor, Fold_dev, X_ijk_dev,
                            &beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize[1], 0));
                    } else {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan[0], &alpha_cutensor, Fold_dev, X_ijk_dev,
                            &beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize[0], 0));
                    }

                    for (int ip = 0; ip < 3; ip++) {
                        int ind_dF3 = is*size_F.nR_split + (i + j*size_F.const_2Bx + k*size_F.const_2Bxs)*size_F.nR_compact + ip*size_F.nTot_compact;
                        cudaMemcpy(dF3+ind_dF3, dF3_dev+ip*nR_split, nR_split*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                    }
                }
            }
        }
    }

    // free memory
    cudaErrorHandle(cudaFree(X_dev));
    cudaErrorHandle(cudaFree(X_ijk_dev));
    cudaErrorHandle(cudaFree(dF3_dev));

    ////////////////
    // multiply u //
    ////////////////

    // dF(indmn,indmn,l,i,j,k,p) = -dF(indmn,indmn,l,i,j,k,p)*u(indmn,indmn,l,p)'

    // set up GPU arrays
    cuDoubleComplex* u_dev;
    cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F.nR_compact*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(u_dev, u_compact, 3*size_F.nR_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* dF_dev;
    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    cuDoubleComplex* dF_dev_result;
    cudaErrorHandle(cudaMalloc(&dF_dev_result, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    // set up CPU arrays
    permute_F(Fold_compact, true, &size_F);

    // set up cublas
    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    cuDoubleComplex alpha_cublas = make_cuDoubleComplex(1,0);
    cuDoubleComplex beta_cublas = make_cuDoubleComplex(0,0);

    // calculate
    for (int ip = 0; ip < 3; ip++) {
        for (int k = 0; k < size_F.const_2Bx; k++) {
            int ind_dF3 = k*size_F.nTot_splitx + ip*size_F.nTot_compact;
            cudaErrorHandle(cudaMemcpy(dF_dev, dF3+ind_dF3, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            for (int l = 0; l <= size_F.lmax; l++)
            {
                int ind_dF = l*(2*l-1)*(2*l+1)/3;
                long long int stride_dF = size_F.nR_compact;

                int ind_u = l*(2*l-1)*(2*l+1)/3 + ip*size_F.nR_compact;
                long long int stride_u = 0;

                cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                    &alpha_cublas, dF_dev+ind_dF, 2*l+1, stride_dF,
                    u_dev+ind_u, 2*l+1, stride_u,
                    &beta_cublas, dF_dev_result+ind_dF, 2*l+1, stride_dF, size_F.const_2Bxs));
            }

            cudaErrorHandle(cudaMemcpy(dF3+ind_dF3, dF_dev_result, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        }
    }

    // free memory
    cudaErrorHandle(cudaFree(u_dev));
    cudaErrorHandle(cudaFree(dF_dev_result));

    /////////////
    // addup F //
    /////////////

    // dF = sum(dF,'p')

    // set up GPU arrays
    cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    // set up CPU arrays
    cuDoubleComplex* dF = new cuDoubleComplex[size_F.nTot_compact];

    // set up blocksize and gridsize
    dim3 blocksize_512_nTot(512, 1, 1);
    dim3 gridsize_512_nTot(ceil((double) size_F.nTot_splitx/512), 1, 1);

    // calculate
    for (int k = 0; k < size_F.const_2Bx; k++) {
        for (int ip = 0; ip < 3; ip++) {
            int ind_dF3 = k*size_F.nTot_splitx + ip*size_F.nTot_compact;
            int ind_dF3_dev = ip*size_F.nTot_splitx;

            cudaErrorHandle(cudaMemcpy(dF3_dev+ind_dF3_dev, dF3+ind_dF3, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }

        addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        int ind_dF = k*size_F.nTot_splitx;
        cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF3_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // free memory
    cudaErrorHandle(cudaFree(dF3_dev));

    //////////////////////////////
    // circular convolution OJO //
    //////////////////////////////

    // OJO_ijk = flip(flip(flip(OJO,1),2),3)
    // OJO_ijk = circshift(OJO_ijk,1,i)
    // OJO_ijk = circshift(OJO_ijk,2,j)
    // OJO_ijk = circshift(OJO_ijk,3,k)
    // dF{r,i,j,k,p} = Fold{r,m,n,l}.*OJO_ijk{m,n,l,p}
    // dF{r,i,j,k,p} = dF{r,i,j,k,p}*c(p)

    std::cout << "circular convolution with OJO begin\n";

    // set up GPU arrays
    cuDoubleComplex* OJO_dev;
    cudaErrorHandle(cudaMalloc(&OJO_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(OJO_dev, OJO, 3*size_F.nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* OJO_ijk_dev;
    cudaErrorHandle(cudaMalloc(&OJO_ijk_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));

    cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F.nR_split*sizeof(cuDoubleComplex)));

    // set up CPU arrays
    permute_F(Fold_compact, false, &size_F);

    // set up blocksize and gridsize
    dim3 blocksize_512_nR(512, 1, 1);
    dim3 gridsize_512_nR(ceil((double) size_F.nR_split/512), 1, 1);

    // calculate
    for (int is = 0; is < size_F.ns; is++) {
        int nR_split;
        if (is == size_F.ns-1) {
            nR_split = size_F.nR_remainder;
        } else {
            nR_split = size_F.nR_split;
        }

        gridsize_512_nR.x = ceil((double) nR_split/512);

        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact + is*size_F.nx*size_F.nR_split, size_F.nx*nR_split*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        for (int i = 0; i < size_F.const_2Bx; i++) {
            for (int j = 0; j < size_F.const_2Bx; j++) {
                for (int k = 0; k < size_F.const_2Bx; k++) {
                    flip_shift <<<gridsize_8, blocksize_8>>> (OJO_dev, OJO_ijk_dev, i, j, k, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());

                    if (is == size_F.ns-1) {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan[1], &alpha_cutensor, Fold_dev, OJO_ijk_dev,
                            &beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize[1], 0));
                    } else {
                        cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan[0], &alpha_cutensor, Fold_dev, OJO_ijk_dev,
                            &beta_cutensor, dF3_dev, dF3_dev, cutensor_workspace, worksize[0], 0));
                    }

                    double c[3];
                    deriv_x(c, i, size_F.Bx, *L);
                    deriv_x(c+1, j, size_F.Bx, *L);
                    deriv_x(c+2, k, size_F.Bx, *L);

                    mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF3_dev, c[0], nR_split, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());
                    mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF3_dev+nR_split, c[1], nR_split, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());
                    mulImg_FR <<<gridsize_512_nR, blocksize_512_nR>>> (dF3_dev+2*nR_split, c[2], nR_split, size_F_dev);
                    cudaErrorHandle(cudaGetLastError());

                    for (int ip = 0; ip < 3; ip++) {
                        int ind_dF3 = is*size_F.nR_split + (i + j*size_F.const_2Bx + k*size_F.const_2Bxs)*size_F.nR_compact + ip*size_F.nTot_compact;
                        cudaMemcpy(dF3+ind_dF3, dF3_dev+ip*nR_split, nR_split*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                    }
                }
            }
        }
    }

    // free memory
    cudaErrorHandle(cudaFree(OJO_dev));
    cudaErrorHandle(cudaFree(OJO_ijk_dev));
    cudaErrorHandle(cudaFree(dF3_dev));
    cudaErrorHandle(cudaFree(Fold_dev));

    /////////////
    // addup F //
    /////////////

    // dF = sum(dF,'p')

    // set up GPU arrays
    cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    // calculate
    for (int k = 0; k < size_F.const_2Bx; k++) {
        for (int ip = 0; ip < 3; ip++) {
            int ind_dF3 = k*size_F.nTot_splitx + ip*size_F.nTot_compact;
            int ind_dF3_dev = ip*size_F.nTot_splitx;

            cudaErrorHandle(cudaMemcpy(dF3_dev+ind_dF3_dev, dF3+ind_dF3, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }

        addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        int ind_dF = k*size_F.nTot_splitx;
        cudaErrorHandle(cudaMemcpy(dF_dev, dF+ind_dF, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF3_dev, size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // free memory
    cudaErrorHandle(cudaFree(dF3_dev));

    ///////////////////////
    // kronecker product //
    ///////////////////////

    std::cout << "kronecker product begin\n";

    // set up GPU arrays
    double** CG_dev = new double* [size_F.BR*size_F.BR];
    for (int l1 = 0; l1 <= size_F.lmax; l1++) {
        for (int l2 = 0; l2 <= size_F.lmax; l2++) {
            int m = (2*l1+1)*(2*l2+1);
            int ind_CG = l1+l2*size_F.BR;
            cudaErrorHandle(cudaMalloc(&CG_dev[ind_CG], m*m*sizeof(double)));
            cudaErrorHandle(cudaMemcpy(CG_dev[ind_CG], CG[ind_CG], m*m*sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    cuDoubleComplex* MR_dev;
    cudaErrorHandle(cudaMalloc(&MR_dev, 3*size_F.nR_compact*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(MR_dev, MR_compact, 3*size_F.nR_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    double* FMR_real_dev;
    double* FMR_imag_dev;
    int m = (2*size_F.lmax+1) * (2*size_F.lmax+1);
    cudaErrorHandle(cudaMalloc(&FMR_real_dev, m*m*size_F.const_2Bx*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&FMR_imag_dev, m*m*size_F.const_2Bx*sizeof(double)));

    double* FMR_real_temp_dev;
    double* FMR_imag_temp_dev;
    cudaErrorHandle(cudaMalloc(&FMR_real_temp_dev, (2*size_F.lmax+1)*m*size_F.const_2Bx*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&FMR_imag_temp_dev, (2*size_F.lmax+1)*m*size_F.const_2Bx*sizeof(double)));

    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx2*sizeof(cuDoubleComplex)));

    cudaErrorHandle(cudaMalloc(&dF3_dev, 3*size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    // set up CPU arrays
    permute_F(Fold_compact, true, &size_F);

    // get c
    double* c = new double[size_F.const_2Bx];
    for (int i = 0; i < size_F.const_2Bx; i++) {
        deriv_x(&c[i], i, size_F.Bx, *L);
    }

    double* c_dev;
    cudaErrorHandle(cudaMalloc(&c_dev, size_F.const_2Bx*sizeof(double)));
    cudaErrorHandle(cudaMemcpy(c_dev, c, size_F.const_2Bx*sizeof(double), cudaMemcpyHostToDevice));

    // set up block size
    dim3 blocksize_kron(1,1,1);
    dim3 gridsize_kron(1,1,1);
    bool iswrapped = false;

    // set up cublas
    double alpha_cublas_real = 1;
    double beta_cublas_real = 0;

    // calculate
    for (int j = 0; j < size_F.const_2Bx; j++) {
        for (int k = 0; k < size_F.const_2Bx; k++) {
            int ind_Fold = (j*size_F.const_2Bx + k*size_F.const_2Bxs)*size_F.nR_compact;

            cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_Fold, size_F.nTot_splitx2*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cudaErrorHandle(cudaMemset(dF3_dev, 0, 3*size_F.nTot_splitx2*sizeof(cuDoubleComplex)));

            for (int ip = 0; ip < 3; ip++) {
                int ind_MR = ip*size_F.nR_compact;
                int ind_dF3_dev = ip*size_F.nTot_splitx2;

                for (int l = 0; l <= size_F.lmax; l++) {
                    dim3 blocksize_add_FMR(512, 1, 1);
                    dim3 gridsize_add_FMR((int)512/((2*l+1)*(2*l+1))+1, size_F.const_2Bx, 1);

                    int ind_cumR = l*(2*l-1)*(2*l+1)/3;

                    for (int l1 = 0; l1 <= size_F.lmax; l1++) {
                        int ind_F_cumR = l1*(2*l1-1)*(2*l1+1)/3;

                        for (int l2 = 0; l2 <= size_F.lmax; l2++) {
                            int ind_MR_cumR = l2*(2*l2-1)*(2*l2+1)/3;

                            int ind_CG = l1+l2*size_F.BR;
                            int m = (2*l1+1)*(2*l2+1);
                            int ind_CG_l1l2 = (l*l-(l1-l2)*(l1-l2))*m;

                            if (abs(l1-l2)<=l && l1+l2>=l) {
                                get_blocksize(&gridsize_kron, &blocksize_kron, &iswrapped, l1, l2, &size_F);

                                kron_FMR <<<gridsize_kron, blocksize_kron>>> (FMR_real_dev, FMR_imag_dev, Fold_dev, MR_dev+ind_MR, ind_F_cumR, ind_MR_cumR, iswrapped, 2*l1+1, 2*l2+1, size_F_dev);
                                cudaErrorHandle(cudaGetLastError());

                                // FMR is now a (2l1+1)(2l2+1)-by-(2l1+1)(2l2+1)-by-(2Bx) tensor

                                // real part
                                alpha_cublas_real = -1;
                                cublasErrorHandle(cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    2*l+1, m, m, &alpha_cublas_real, CG_dev[ind_CG]+ind_CG_l1l2, m, 0,
                                    FMR_real_dev, m, m*m, &beta_cublas_real, FMR_real_temp_dev, 2*l+1, (2*l+1)*m, size_F.const_2Bx));

                                alpha_cublas_real = (double) m/(2*l+1);
                                cublasErrorHandle(cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    2*l+1, 2*l+1, m, &alpha_cublas_real, FMR_real_temp_dev, 2*l+1, (2*l+1)*m,
                                    CG_dev[ind_CG]+ind_CG_l1l2, m, 0, &beta_cublas_real, FMR_real_dev, 2*l+1, (2*l+1)*(2*l+1), size_F.const_2Bx));

                                // imaginary part
                                alpha_cublas_real = -1;
                                cublasErrorHandle(cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                    2*l+1, m, m, &alpha_cublas_real, CG_dev[ind_CG]+ind_CG_l1l2, m, 0,
                                    FMR_imag_dev, m, m*m, &beta_cublas_real, FMR_imag_temp_dev, 2*l+1, (2*l+1)*m, size_F.const_2Bx));

                                alpha_cublas_real = (double) m/(2*l+1);
                                cublasErrorHandle(cublasDgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    2*l+1, 2*l+1, m, &alpha_cublas_real, FMR_imag_temp_dev, 2*l+1, (2*l+1)*m,
                                    CG_dev[ind_CG]+ind_CG_l1l2, m, 0, &beta_cublas_real, FMR_imag_dev, 2*l+1, (2*l+1)*(2*l+1), size_F.const_2Bx));

                                // FMR is now a (2l+1)-by-(2l+1)-by-(2Bx) tensor

                                // add
                                add_FMR <<<gridsize_add_FMR, blocksize_add_FMR>>> (dF3_dev+ind_dF3_dev, FMR_real_dev, FMR_imag_dev, ind_cumR, (2*l+1)*(2*l+1), size_F_dev);
                            }
                        }
                    }
                }

                int ind_dF3 = ind_Fold + ip*size_F.nTot_compact;
                cudaErrorHandle(cudaMemcpy(dF3+ind_dF3, dF3_dev+ind_dF3_dev, size_F.nTot_splitx2*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
            }
        }
    }

    // take derivative about x
    dim3 blocksize_deriv(512,1,1);
    dim3 gridsize_deriv(ceil((double) size_F.nR_compact/512), size_F.const_2Bx, size_F.const_2Bx);

    for (int k = 0; k < size_F.const_2Bx; k++) {
        for (int ip = 0; ip < 3; ip++) {
            int ind_dF3 = k*size_F.nTot_splitx + ip*size_F.nTot_compact;

            cudaErrorHandle(cudaMemcpy(dF3_dev, dF3+ind_dF3, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            mulImg_FTot <<<gridsize_deriv, blocksize_deriv>>> (dF3_dev, c_dev, ip, k, size_F_dev);
            cudaErrorHandle(cudaGetLastError());

            cudaErrorHandle(cudaMemcpy(dF3+ind_dF3, dF3_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        }
    }

    // add up F
    for (int k = 0; k < size_F.const_2Bx; k++) {
        for (int ip = 0; ip < 3; ip++) {
            int ind_dF3 = k*size_F.nTot_splitx + ip*size_F.nTot_compact;
            int ind_dF3_dev = ip*size_F.nTot_splitx;

            cudaErrorHandle(cudaMemcpy(dF3_dev+ind_dF3_dev, dF3+ind_dF3, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        }

        addup_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF3_dev, size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        int ind_dF = k*size_F.nTot_splitx;
        cudaErrorHandle(cudaMemcpy(dF_dev, dF+ind_dF, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        add_F <<<gridsize_512_nTot, blocksize_512_nTot>>> (dF_dev, dF3_dev, size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        cudaErrorHandle(cudaMemcpy(dF+ind_dF, dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // free memory
    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(dF3_dev));
    cudaErrorHandle(cudaFree(MR_dev));
    cudaErrorHandle(cudaFree(FMR_real_dev));
    cudaErrorHandle(cudaFree(FMR_imag_dev));
    cudaErrorHandle(cudaFree(c_dev));

    for (int l1 = 0; l1 <= size_F.lmax; l1++) {
        for (int l2 = 0; l2 <= size_F.lmax; l2++) {
            int ind_CG = l1+l2*size_F.BR;
            cudaErrorHandle(cudaFree(CG_dev[ind_CG]));
        }
    }

    delete[] c;

    ///////////////
    // integrate //
    ///////////////

    // Fnew = Fold + dt*dF

    // set up GPU arrays
    cuDoubleComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

    for (int k = 0; k < size_F.const_2Bx; k++) {
        int ind_F = k*size_F.nTot_splitx;
        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        cudaErrorHandle(cudaMemcpy(dF_dev, dF+ind_F, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F_dev);
        cudaErrorHandle(cudaGetLastError());

        cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    }

    // gather Fnew
    modify_F(Fnew_compact, Fnew, false, &size_F);

    // free memory
    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(Fnew_dev));
    cudaErrorHandle(cudaFree(dF_dev));
    cudaErrorHandle(cudaFree(size_F_dev));

    delete[] dF3;
    delete[] dF;

    delete[] Fold_compact;
    delete[] Fnew_compact;
    delete[] u_compact;
    delete[] MR_compact;
    delete[] CG;
}

