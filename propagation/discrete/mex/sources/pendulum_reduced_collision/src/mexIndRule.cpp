#include "getIndRule.cuh"

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from matlab
    myReal* x = mymxGetReal(prhs[0]);
    myReal* Omega = mymxGetReal(prhs[1]);

    const mwSize* size_Omega = mxGetDimensions(prhs[1]);
    int numR = size_Omega[1];
    Size_f size_f;
    init_Size_f(&size_f, 1, size_Omega[2]/2);

    // setup output
    mwSize size_interp[4] = {4, (mwSize)numR, (mwSize)size_f.const_2Bx, (mwSize)size_f.const_2Bx};
    plhs[0] = mxCreateNumericArray(4, size_interp, mxINT32_CLASS, mxREAL);
    int* ind_interp = mxGetInt32s(plhs[0]);

    plhs[1] = mxCreateNumericArray(4, size_interp, mymxRealClass, mxREAL);
    myReal* coeff_interp = mymxGetReal(plhs[1]);

    // compute
    getIndRule(ind_interp, coeff_interp, x, Omega, numR, &size_f);
}
