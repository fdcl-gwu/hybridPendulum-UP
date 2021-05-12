#include "getOmega.cuh"

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get arrays from Matlab
    myReal* R = mymxGetReal(prhs[0]);
    const mwSize* size_R = mxGetDimensions(prhs[0]);

    myReal* x = mymxGetReal(prhs[1]);
    const mwSize* size_x = mxGetDimensions(prhs[1]);
    
    Size_f size_f;
    init_Size_f(&size_f, size_R[2]/2, size_x[1]/2);

    char* lambda_cat = (char*) mxGetInt8s(prhs[2]);

    myReal* epsilon = mymxGetReal(prhs[3]);

    // find nonzero lambda
    int* ind_n0 = (int*) malloc(size_f.nR*sizeof(int));
    int nn0 = 0;
    for (int iR = 0; iR < size_f.nR; iR++) {
        if (lambda_cat[iR] != 0) {
            ind_n0[nn0] = iR;
            nn0++;
        }
    }

    // set up output
    size_t size_Omega[4] = {2, (size_t)nn0, (size_t)size_f.const_2Bx, (size_t)size_f.const_2Bx};
    plhs[0] = mxCreateUninitNumericArray(4, size_Omega, mymxRealClass, mxREAL);
    myReal* Omega = mymxGetReal(plhs[0]);

    // calculate
    getOmega(Omega, R, x, ind_n0, nn0, epsilon, &size_f);

    // free memory
    free(ind_n0);
}

