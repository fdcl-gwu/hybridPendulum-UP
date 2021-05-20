#include "propagate.cuh"

#include <stdio.h>
#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab
    // prhs[0]: f
    // prhs[1]: lambda
    // prhs[2]: fcL
    // prhs[3]: indR
    // prhs[4]: lambda_indx
    // prhs[5]: fcL_indx1
    // prhs[6]: fcL_indx2
    // prhs[7]: fcL_numx2

    myReal* f = mymxGetReal(prhs[0]);
    const mwSize* size_fin = mxGetDimensions(prhs[0]);

    Size_f size_f;
    init_Size_f(&size_f, (int)size_fin[0]/2, (int)size_fin[3]/2);

    myReal* lambda = mymxGetReal(prhs[1]);
    const mwSize* size_R = mxGetDimensions(prhs[1]);
    int numR = size_R[0];

    int* indR = mxGetInt32s(prhs[3]);

    int** lambda_indx = (int**) malloc(numR*sizeof(int*));
    int* lambda_numx = (int*) malloc(numR*sizeof(int*));

    myReal** fcL = (myReal**) malloc(numR*sizeof(myReal*));
    int** fcL_indx1 = (int**) malloc(numR*sizeof(int*));
    int* fcL_numx1 = (int*) malloc(numR*sizeof(int));
    int** fcL_indx2 = (int**) malloc(numR*sizeof(int*));
    int** fcL_numx2 = (int**) malloc(numR*sizeof(int*));

    for (int iR = 0; iR < numR; iR++) {
        // lambda
        const mwSize* size_lambda_x1 = mxGetDimensions(mxGetCell(prhs[4], iR));
        lambda_numx[iR] = size_lambda_x1[0];
        lambda_indx[iR] = mxGetInt32s(mxGetCell(prhs[4], iR));

        // fcL
        const mwSize* size_fcL_x1 = mxGetDimensions(mxGetCell(prhs[5], iR));
        fcL_numx1[iR] = size_fcL_x1[0];
        
        fcL[iR] = mymxGetReal(mxGetCell(prhs[2], iR));
        fcL_indx1[iR] = mxGetInt32s(mxGetCell(prhs[5], iR));
        fcL_indx2[iR] = mxGetInt32s(mxGetCell(prhs[6], iR));
        fcL_numx2[iR] = mxGetInt32s(mxGetCell(prhs[7], iR));
    }
    
    // setup output
    size_t size_fout[5] = {(size_t)size_f.const_2BR, (size_t)size_f.const_2BR, (size_t)size_f.const_2BR, (size_t)size_f.const_2Bx, (size_t)size_f.const_2Bx};
    plhs[0] = mxCreateUninitNumericArray(5, size_fout, mymxRealClass, mxREAL);
    myReal* df = mymxGetReal(plhs[0]);

    // compute
    get_df(df, f, lambda, fcL, numR, indR, lambda_indx, lambda_numx, fcL_indx1, fcL_numx1, fcL_indx2, fcL_numx2, &size_f);

    // free memory
    free(lambda_indx);
    free(lambda_numx);
    free(fcL);
    free(fcL_indx1);
    free(fcL_numx1);
    free(fcL_indx2);
    free(fcL_numx2);
}


