
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include <math.h>

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    /* set up arrays */
    cuDoubleComplex* Fold = (cuDoubleComplex*) mxGetComplexDoubles(prhs[0]);
    size_t dims_F[6] = {2*lmax+1, 2*lmax+1, lmax+1, 2*B, 2*B, 2*B};
    plhs[0] = mxCreateUninitNumericArray(6, dims_F, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fold_dev;
    cuDoubleComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fold_dev, nTot*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&Fnew_dev, nTot*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(Fold_dev, Fold, nTot*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* X = (cuDoubleComplex*) mxGetComplexDoubles(prhs[1]);
    cuDoubleComplex* X_dev;
    cuDoubleComplex* X_ijk_dev;
    cudaErrorHandle(cudaMalloc(&X_dev, 3*nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(X_dev, X, 3*nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    double* dt = mxGetDoubles(prhs[2]);

    cuDoubleComplex* u = (cuDoubleComplex*) mxGetComplexDoubles(prhs[3]);
    cuDoubleComplex* u_dev;
    cudaErrorHandle(cudaMalloc(&u_dev, 3*nR*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(u_dev, u, 3*nR*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    cuDoubleComplex* temp_dev;
    cudaErrorHandle(cudaMalloc(&temp_dev, 3*nR*sizeof(cuDoubleComplex)));

    /* set up block and grid sizes */
    dim3 blocksize_X(8, 8, 8);
    int gridnum_X = ceil((double) const_2B/8);
    dim3 gridsize_X(gridnum_X, gridnum_X, gridnum_X);

    dim3 blocksize_u(2*lmax+1, 2*lmax+1, 1);
    dim3 gridsize_u(lmax+1, 1, 1);

    dim3 blocksize_add(512,1,1);
    dim3 gridsize_add(ceil((double) nTot/512),1,1);

    /* set up tensors */
    int mode_Fold[6] = {'m','n','l','i','j','k'};
    int mode_X[4] = {'i','j','k','p'};
    int mode_temp[4] = {'m','n','l','p'};

    int64_t extent_Fold[6] = {2*lmax+1, 2*lmax+1, lmax+1, 2*B, 2*B, 2*B};
    int64_t extent_X[4] = {2*B, 2*B, 2*B, 3};
    int64_t extent_temp[4] = {2*lmax+1, 2*lmax+1, lmax+1, 3};

    cutensorHandle_t handle;
    cutensorInit(&handle);

    cutensorTensorDescriptor_t desc_Fold;
    cutensorTensorDescriptor_t desc_X;
    cutensorTensorDescriptor_t desc_temp;
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle, &desc_Fold,
        6, extent_Fold, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle, &desc_X,
        4, extent_X, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle, &desc_temp,
        4, extent_temp, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));

    uint32_t alignmentRequirement_Fold;
    uint32_t alignmentRequirement_X;
    uint32_t alignmentRequirement_temp;
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle,
        Fold_dev, &desc_Fold, &alignmentRequirement_Fold));
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle,
        X_dev, &desc_X, &alignmentRequirement_X));
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle,
        temp_dev, &desc_temp, &alignmentRequirement_temp));

    cutensorContractionDescriptor_t desc;
    cutensorErrorHandle(cutensorInitContractionDescriptor(&handle, &desc,
        &desc_Fold, mode_Fold, alignmentRequirement_Fold,
        &desc_X, mode_X, alignmentRequirement_X,
        &desc_temp, mode_temp, alignmentRequirement_temp,
        &desc_temp, mode_temp, alignmentRequirement_temp,
        CUTENSOR_COMPUTE_32F));

    cutensorContractionFind_t find;
    cutensorErrorHandle(cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT));

    size_t worksize = 0;
    cutensorErrorHandle(cutensorContractionGetWorkspace(&handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

    void* work = nullptr;
    if (worksize > 0) {
        cudaErrorHandle(cudaMalloc(&work, worksize));
    }

    cutensorContractionPlan_t plan;
    cutensorErrorHandle(cutensorInitContractionPlan(&handle, &plan, &desc, &find, worksize));

    cuDoubleComplex alpha = make_cuDoubleComplex((double)1/nx,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);

    /* calculate */
    for (int i = 0; i < const_2B; i++) {
        for (int j = 0; j < const_2B; j++) {
            for (int k = 0; k < const_2B; k++) {
                flip_shift <<<gridsize_X, blocksize_X>>> (X_dev, X_ijk_dev, i, j, k);
                cudaErrorHandle(cudaGetLastError());

                cutensorErrorHandle(cutensorContraction(&handle, &plan, (void*)&alpha, Fold_dev, X_ijk_dev,
                    (void*)&beta, temp_dev, temp_dev, work, worksize, 0));

                cudaDeviceSynchronize();

                derivate <<<gridsize_u, blocksize_u>>> (temp_dev, u_dev, Fnew_dev, i, j, k);

                cudaDeviceSynchronize();
            }
        }
    }

    add_dF <<<gridsize_add, blocksize_add>>> (Fnew_dev, Fold_dev, dt[0]);

    cudaMemcpy(Fnew, Fnew_dev, nTot*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cudaFree(Fold_dev);
    cudaFree(Fnew_dev);
    cudaFree(X_dev);
    cudaFree(X_ijk_dev);
    cudaFree(u_dev);
}

