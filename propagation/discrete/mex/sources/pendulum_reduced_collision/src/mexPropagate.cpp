#include "propagate.cuh"

#include <stdio.h>
#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab
    myReal* f = mymxGetReal(prhs[0]);
    const mwSize* size_fin = mxGetDimensions(prhs[0]);

    Size_f size_f;
    init_Size_f(&size_f, (int)size_fin[0]/2, (int)size_fin[3]/2);

    myReal* x = mymxGetReal(prhs[1]);
    myReal* lambda = mymxGetReal(prhs[2]);
    myReal* Omega = mymxGetReal(prhs[3]);
    char* lambda_cat = (char*) mxGetInt8s(prhs[4]);
    myReal* Gd = mymxGetReal(prhs[5]);

    // setup output
    size_t size_fout[5] = {(size_t)size_f.const_2BR, (size_t)size_f.const_2BR, (size_t)size_f.const_2BR, (size_t)size_f.const_2Bx, (size_t)size_f.const_2Bx};
    plhs[0] = mxCreateUninitNumericArray(5, size_fout, mymxRealClass, mxREAL);
    myReal* fnew = mymxGetReal(plhs[0]);

    // compute
    get_df(fnew, f, x, lambda, Omega, lambda_cat, Gd, &size_f);
}


