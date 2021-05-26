#include "getFcL.cuh"

#include "string.h"
#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab
    myReal* x = mymxGetReal(prhs[0]);
    const mwSize* size_x = mxGetDimensions(prhs[0]);
    
    Size_f size_f;
    init_Size_f(&size_f, 0, size_x[1]/2);

    myReal* Omega = mymxGetReal(prhs[1]);
    const mwSize* size_Omega = mxGetDimensions(prhs[1]);
    int numR = size_Omega[1];

    myReal* lambda = mymxGetReal(prhs[2]);
    myReal* Gd = mymxGetReal(prhs[3]); 
    int* nD = mxGetInt32s(prhs[4]);

    // setup output
    mwSize size_fcL[4] = {(mwSize)(*nD), (mwSize)numR, (mwSize)size_f.const_2Bx, (mwSize)size_f.const_2Bx};
    plhs[0] = mxCreateNumericArray(4, size_fcL, mymxRealClass, mxREAL);
    myReal* fcL = mymxGetReal(plhs[0]);
    
    plhs[1] = mxCreateNumericArray(4, size_fcL, mxINT16_CLASS, mxREAL);
    short* fcL_indx = mxGetInt16s(plhs[1]);

    // compute
    getFcL(fcL, fcL_indx, x, Omega, lambda, numR, Gd, *nD, &size_f);
}


