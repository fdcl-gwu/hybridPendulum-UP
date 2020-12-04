
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include <math.h>

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

    cuDoubleComplex* Fold_dev;
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(size_F), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fnew_compact = new cuDoubleComplex[size_F.nTot_compact];

    cuDoubleComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_dev, 3*size_F.nTot_compact*sizeof(cuDoubleComplex)));

    cuDoubleComplex* Fnew_buffer_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_buffer_dev, 3*size_F.nTot_compact*sizeof(cuDoubleComplex)));
    
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

    // set up temporary variables for calculation
    cuDoubleComplex* temp_dev;
    cudaErrorHandle(cudaMalloc(&temp_dev, 3*size_F.nR_compact*sizeof(cuDoubleComplex)));

    cuDoubleComplex* X_ijk_dev;
    cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));

    /////////////////////////////////
    // set up block and grid sizes //
    /////////////////////////////////

    // flip the circshift X
    dim3 blocksize_X(8, 8, 8);
    int gridnum_X = ceil((double) size_F.const_2Bx/8);
    dim3 gridsize_X(gridnum_X, gridnum_X, gridnum_X);

    // dddup_F
    dim3 blocksize_add(512, 1, 1);
    dim3 gridsize_add(ceil((double) size_F.nR_compact/512), size_F.nx, 1);

    ////////////////////
    // set up tensors //
    ////////////////////
    int mode_Fold[4] = {'r','i','j','k'};
    int mode_X[4] = {'i','j','k','p'};
    int mode_temp[2] = {'r','p'};

    int64_t extent_Fold[4] = {size_F.nR_compact, size_F.const_2Bx, size_F.const_2Bx, size_F.const_2Bx};
    int64_t extent_X[4] = {size_F.const_2Bx, size_F.const_2Bx, size_F.const_2Bx, 3};
    int64_t extent_temp[2] = {size_F.nR_compact, 3};

    cutensorHandle_t handle_cutensor;
    cutensorInit(&handle_cutensor);

    cutensorTensorDescriptor_t desc_Fold;
    cutensorTensorDescriptor_t desc_X;
    cutensorTensorDescriptor_t desc_temp;
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle_cutensor, &desc_Fold,
        4, extent_Fold, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle_cutensor, &desc_X,
        4, extent_X, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle_cutensor, &desc_temp,
        2, extent_temp, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));

    uint32_t alignmentRequirement_Fold;
    uint32_t alignmentRequirement_X;
    uint32_t alignmentRequirement_temp;
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle_cutensor,
        Fold_dev, &desc_Fold, &alignmentRequirement_Fold));
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle_cutensor,
        X_dev, &desc_X, &alignmentRequirement_X));
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle_cutensor,
        temp_dev, &desc_temp, &alignmentRequirement_temp));

    cutensorContractionDescriptor_t desc;
    cutensorErrorHandle(cutensorInitContractionDescriptor(&handle_cutensor, &desc,
        &desc_Fold, mode_Fold, alignmentRequirement_Fold,
        &desc_X, mode_X, alignmentRequirement_X,
        &desc_temp, mode_temp, alignmentRequirement_temp,
        &desc_temp, mode_temp, alignmentRequirement_temp,
        CUTENSOR_COMPUTE_32F));

    cutensorContractionFind_t find;
    cutensorErrorHandle(cutensorInitContractionFind(&handle_cutensor, &find, CUTENSOR_ALGO_DEFAULT));

    size_t worksize = 0;
    cutensorErrorHandle(cutensorContractionGetWorkspace(&handle_cutensor, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void* work = nullptr;
    if (worksize > 0) {
        cudaErrorHandle(cudaMalloc(&work, worksize));
    }

    cutensorContractionPlan_t plan;
    cutensorErrorHandle(cutensorInitContractionPlan(&handle_cutensor, &plan, &desc, &find, worksize));

    cuDoubleComplex alpha = make_cuDoubleComplex((double)1/size_F.nx,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);

    ///////////////////
    // set up cublas //
    ///////////////////

    cublasHandle_t handle_cublas;
    cublasCreate(&handle_cublas);

    cuDoubleComplex alpha_cublas = make_cuDoubleComplex(-1,0);

    ///////////////
    // calculate //
    ///////////////

    // circular convolution
    for (int i = 0; i < size_F.const_2Bx; i++) {
        for (int j = 0; j < size_F.const_2Bx; j++) {
            for (int k = 0; k < size_F.const_2Bx; k++) {

                flip_shift <<<gridsize_X, blocksize_X>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                cudaDeviceSynchronize();

                cutensorErrorHandle(cutensorContraction(&handle_cutensor, &plan, (void*)&alpha, Fold_dev, X_ijk_dev,
                    (void*)&beta, temp_dev, temp_dev, work, worksize, 0));

                cudaDeviceSynchronize();

                for (int n = 0; n < 3; n++) {
                    cuDoubleComplex* Fnew_dev_ijkn = Fnew_dev + i*size_F.nR_compact + 
                        j*(size_F.nR_compact*size_F.const_2Bx) + k*(size_F.nR_compact*size_F.const_2Bxs) + n*size_F.nTot_compact;
                    cuDoubleComplex* temp_dev_n = temp_dev + n*size_F.nR_compact;

                    cudaErrorHandle(cudaMemcpy(Fnew_dev_ijkn, temp_dev_n, size_F.nR_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice));
                }

                cudaDeviceSynchronize();
            }
        }
    }

    // multiply u
    for (int i = 0; i < 3; i++) {
        for (int l = 0; l <= size_F.lmax; l++)
        {
            int ind_Fnew = l*(2*l-1)*(2*l+1)/3 + i*size_F.nTot_compact;
            long long int stride_Fnew = size_F.nR_compact;

            int ind_u = l*(2*l-1)*(2*l+1)/3 + i*size_F.nR_compact;
            long long int stride_u = 0;

            cublasErrorHandle(cublasZgemmStridedBatched(handle_cublas, CUBLAS_OP_N, CUBLAS_OP_T, 2*l+1, 2*l+1, 2*l+1,
                &alpha_cublas, Fnew_dev+ind_Fnew, 2*l+1, stride_Fnew,
                u_dev+ind_u, 2*l+1, stride_u,
                &beta, Fnew_dev+ind_Fnew, 2*l+1, stride_Fnew, size_F.nx));

            cudaDeviceSynchronize();
        }
    }

    // addup F
    addup_F <<<gridsize_add, blocksize_add>>> (Fnew_dev, Fold_dev, dt[0], size_F_dev);
    cudaErrorHandle(cudaGetLastError());

    cudaDeviceSynchronize();

    // gather Fnew
    cudaErrorHandle(cudaMemcpy(Fnew_compact, Fnew_dev, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    modify_F(Fnew_compact, Fnew, false, &size_F);

    cudaFree(Fold_dev);
    cudaFree(Fnew_dev);
    cudaFree(Fnew_buffer_dev);
    cudaFree(X_dev);
    cudaFree(X_ijk_dev);
    cudaFree(u_dev);

    delete[] Fold_compact;
    delete[] Fnew_compact;
}

