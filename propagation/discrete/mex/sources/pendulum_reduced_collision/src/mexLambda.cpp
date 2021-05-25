#include "setup.hpp"
#include "getLambda.cuh"
#include "string.h"

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab
    myReal* R = mymxGetReal(prhs[0]);
    const mwSize* size_R = mxGetDimensions(prhs[0]);

    myReal* x = mymxGetReal(prhs[1]);
    const mwSize* size_x = mxGetDimensions(prhs[1]);

    myReal* d = mymxGetReal(prhs[2]);
    myReal* h = mymxGetReal(prhs[3]);
    myReal* r = mymxGetReal(prhs[4]);

    myReal* thetat = mymxGetReal(prhs[5]);
    myReal* lambda_max = mymxGetReal(prhs[6]);

    Size_f size_f;
    init_Size_f(&size_f, (int)size_R[2]/2, (int)size_x[1]/2);

    // set up output
    mwSize size_PC[4] = {3, (mwSize)size_f.const_2BR, (mwSize)size_f.const_2BR, (mwSize)size_f.const_2BR};
    plhs[3] = mxCreateNumericArray(4, size_PC, mymxRealClass, mxREAL);
    myReal* PC = mymxGetReal(plhs[3]);

    // compute
    myReal* lambda = (myReal*) malloc(size_f.nR*sizeof(myReal));
    int* lambda_indR = (int*) malloc(size_f.nR*sizeof(int));
    int lambda_numR;
    int** lambda_indx = (int**) malloc(size_f.nR*sizeof(int*));
    for (int iR = 0; iR < size_f.nR; iR++) {
        lambda_indx[iR] = (int*) malloc(size_f.nx*sizeof(int));
    }
    int* lambda_numx = (int*) calloc(size_f.nR, sizeof(int));

    getLambda(lambda, lambda_indR, &lambda_numR, lambda_indx, lambda_numx, PC, R, x, d, h, r, thetat, lambda_max, &size_f);
    
    // set up output
    mwSize size_lambda[1] = {(mwSize) lambda_numR};
    plhs[0] = mxCreateNumericArray(1, size_lambda, mymxRealClass, mxREAL);
    myReal* lambda_return = mymxGetReal(plhs[0]);
    memcpy(lambda_return, lambda, lambda_numR*sizeof(myReal));

    plhs[1] = mxCreateNumericArray(1, size_lambda, mxINT32_CLASS, mxREAL);
    int* lambda_indR_return = mxGetInt32s(plhs[1]);
    memcpy(lambda_indR_return, lambda_indR, lambda_numR*sizeof(int));

    plhs[2] = mxCreateCellArray(1, size_lambda);
    for (int iR = 0; iR < lambda_numR; iR++) {
        mwSize size_indx[1] = {(mwSize)lambda_numx[iR]};
        mxArray* lambda_indx_mx = mxCreateNumericArray(1, size_indx, mxINT32_CLASS, mxREAL);
        int* lambda_indx_iR = mxGetInt32s(lambda_indx_mx);
        memcpy(lambda_indx_iR, lambda_indx[iR], lambda_numx[iR]*sizeof(int));
        mxSetCell(plhs[2], iR, lambda_indx_mx);
    }

    // free memory
    free(lambda);
    free(lambda_indR);
    for (int iR = 0; iR < size_f.nR; iR++) {
        free(lambda_indx[iR]);
    }
    free(lambda_indx);
    free(lambda_numx);
}
