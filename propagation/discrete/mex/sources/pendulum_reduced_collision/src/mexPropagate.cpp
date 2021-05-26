#include "propagate.cuh"

#include <stdio.h>
#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab

    // if prhs[0] == true
    // prhs[1]: f
    // prhs[2]: lambda
    // prhs[3]: indR
    // prhs[4]: fcL
    // prhs[5]: fcL_indx

    // if prhs[0] == false
    // prhs[1]: f
    // prhs[2]: lambda
    // prhs[3]: indR
    // prhs[4]: lambda_indx
    // prhs[5]: ind_interp
    // prhs[6]: coeff_interp
    
    bool* noise = mxGetLogicals(prhs[0]);    

    myReal* f = mymxGetReal(prhs[1]);
    const mwSize* size_fin = mxGetDimensions(prhs[1]);

    Size_f size_f;
    init_Size_f(&size_f, (int)size_fin[0]/2, (int)size_fin[3]/2);

    myReal* lambda = mymxGetReal(prhs[2]);
    const mwSize* size_R = mxGetDimensions(prhs[2]);
    int numR = size_R[0];

    int* indR = mxGetInt32s(prhs[3]);

    // setup output
    size_t size_fout[5] = {(size_t)size_f.const_2BR, (size_t)size_f.const_2BR, (size_t)size_f.const_2BR, (size_t)size_f.const_2Bx, (size_t)size_f.const_2Bx};
    plhs[0] = mxCreateUninitNumericArray(5, size_fout, mymxRealClass, mxREAL);
    myReal* df = mymxGetReal(plhs[0]);

    // more arrays from matlab and compute
    if (*noise) {
        // get arrays from matlab
        myReal* fcL = mymxGetReal(prhs[4]);
        short* fcL_indx = mxGetInt16s(prhs[5]);

        const mwSize* size_fcL = mxGetDimensions(prhs[4]);
        int nD = size_fcL[0];

        // compute
        get_df_noise(df, f, lambda, fcL, numR, indR, fcL_indx, nD, &size_f);
    } else {
        // get arrays from matlab
        int** lambda_indx = (int**) malloc(numR*sizeof(int*));
        int* lambda_numx = (int*) malloc(numR*sizeof(int*));

        for (int iR = 0; iR < numR; iR++) {
            const mwSize* size_lambda_x1 = mxGetDimensions(mxGetCell(prhs[4], iR));
            lambda_numx[iR] = size_lambda_x1[0];
            lambda_indx[iR] = mxGetInt32s(mxGetCell(prhs[4], iR));
        }

        int* ind_interp = mxGetInt32s(prhs[5]);
        myReal* coeff_interp = mymxGetReal(prhs[6]);

        // compute
        get_df_nonoise(df, f, lambda, numR, indR, lambda_indx, lambda_numx, ind_interp, coeff_interp, &size_f);
        
        // free memory
        free(lambda_indx);
        free(lambda_numx);
    }
}

