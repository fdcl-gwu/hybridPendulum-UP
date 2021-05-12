#include "setup.hpp"
#include "getLambda.cuh"

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

    size_t size_lambda[5] = {(size_t) size_f.const_2BR, (size_t) size_f.const_2BR, (size_t) size_f.const_2BR, (size_t) size_f.const_2Bx, (size_t) size_f.const_2Bx};
    plhs[0] = mxCreateUninitNumericArray(5, size_lambda, mymxRealClass, mxREAL);
    myReal* lambda = mymxGetReal(plhs[0]);

    plhs[1] = mxCreateUninitNumericArray(3, size_lambda, mxINT8_CLASS, mxREAL);
    char* lambda_cat = (char*) mxGetInt8s(plhs[1]);

    // compute
    getLambda(lambda, lambda_cat, R, x, d, h, r, thetat, lambda_max, &size_f);
}
