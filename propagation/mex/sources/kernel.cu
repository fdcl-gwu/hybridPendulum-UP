
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

    cuDoubleComplex* Fold_dev;
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(Fold_dev, Fold, size_F.nTot*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(size_F), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot*sizeof(cuDoubleComplex)));
    
    // get X from matlab
    cuDoubleComplex* X = (cuDoubleComplex*) mxGetComplexDoubles(prhs[1]);

    cuDoubleComplex* X_dev;
    cudaErrorHandle(cudaMalloc(&X_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(X_dev, X, 3*size_F.nx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // get dt from matlab
    double* dt = mxGetDoubles(prhs[2]);

    // get u from matlab
    cuDoubleComplex* u = (cuDoubleComplex*) mxGetComplexDoubles(prhs[3]);

    cuDoubleComplex* u_dev;
    cudaErrorHandle(cudaMalloc(&u_dev, 3*size_F.nR*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(u_dev, u, 3*size_F.nR*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // set up temporary variables for calculation
    cuDoubleComplex* temp_dev;
    cudaErrorHandle(cudaMalloc(&temp_dev, 3*size_F.nR*sizeof(cuDoubleComplex)));

    cuDoubleComplex* X_ijk_dev;
    cudaErrorHandle(cudaMalloc(&X_ijk_dev, 3*size_F.nx*sizeof(cuDoubleComplex)));

    /////////////////////////////////
    // set up block and grid sizes //
    /////////////////////////////////

    // flip the circshift X
    dim3 blocksize_X(8, 8, 8);
    int gridnum_X = ceil((double) size_F.const_2Bx/8);
    dim3 gridsize_X(gridnum_X, gridnum_X, gridnum_X);

    // multiply u
    dim3 blocksize_u(size_F.const_2lp1, size_F.const_2lp1, 1);
    dim3 gridsize_u(size_F.const_lp1, 1, 1);

    // add dF
    dim3 blocksize_add(512,1,1);
    dim3 gridsize_add(ceil((double) size_F.nTot/512),1,1);

    ////////////////////
    // set up tensors //
    ////////////////////
    int mode_Fold[6] = {'m','n','l','i','j','k'};
    int mode_X[4] = {'i','j','k','p'};
    int mode_temp[4] = {'m','n','l','p'};

    int64_t extent_Fold[6] = {size_F.const_2lp1, size_F.const_2lp1, size_F.const_lp1, size_F.const_2Bx, size_F.const_2Bx, size_F.const_2Bx};
    int64_t extent_X[4] = {size_F.const_2Bx, size_F.const_2Bx, size_F.const_2Bx, 3};
    int64_t extent_temp[4] = {size_F.const_2lp1, size_F.const_2lp1, size_F.const_lp1, 3};

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

    cuDoubleComplex alpha = make_cuDoubleComplex((double)1/size_F.nx,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);

    ///////////////
    // calculate //
    ///////////////

    for (int i = 0; i < size_F.const_2Bx; i++) {
        for (int j = 0; j < size_F.const_2Bx; j++) {
            for (int k = 0; k < size_F.const_2Bx; k++) {

                flip_shift <<<gridsize_X, blocksize_X>>> (X_dev, X_ijk_dev, i, j, k, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                cutensorErrorHandle(cutensorContraction(&handle, &plan, (void*)&alpha, Fold_dev, X_ijk_dev,
                    (void*)&beta, temp_dev, temp_dev, work, worksize, 0));

                cudaDeviceSynchronize();

                derivate <<<gridsize_u, blocksize_u>>> (temp_dev, u_dev, Fnew_dev, i, j, k, size_F_dev);
                cudaErrorHandle(cudaGetLastError());

                cudaDeviceSynchronize();
            }
        }
    }

    add_dF <<<gridsize_add, blocksize_add>>> (Fnew_dev, Fold_dev, dt[0], size_F_dev);
    cudaErrorHandle(cudaGetLastError());

    cudaErrorHandle(cudaMemcpy(Fnew, Fnew_dev, size_F.nTot*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    cudaFree(Fold_dev);
    cudaFree(Fnew_dev);
    cudaFree(X_dev);
    cudaFree(X_ijk_dev);
    cudaFree(u_dev);
}

