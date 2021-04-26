#include "getc.hpp"
#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // get f from matlab
    myReal* f = mymxGetReal(prhs[0]);
    const mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
    const mwSize* size_f_mat = mxGetDimensions(prhs[0]);

    Size_F size_F;
    Size_f size_f;
    init_Size_F(&size_F, (int)size_f_mat[0]/2, (int)size_f_mat[3]/2, (int)ndims);
    init_Size_f(&size_f, (int)size_f_mat[0]/2, (int)size_f_mat[3]/2, (int)ndims);

    // get e from matlab
    myReal* e = mymxGetReal(prhs[1]);
    const mwSize* size_e_mat = mxGetDimensions(prhs[1]);

    Size_e size_e;
    init_Size_e(&size_e, (int)size_e_mat[1], (int)size_e_mat[2]);

    // get L from matlab
    myReal* L = mymxGetReal(prhs[2]);

    // calculate
    size_t size_c[2] = {(size_t)size_e.ns, 3};
    plhs[0] = mxCreateUninitNumericArray(2, size_c, mymxRealClass, mxREAL);
    myReal* c = mymxGetReal(plhs[0]);

    get_color(c, f, e, L, &size_F, &size_f, &size_e);
}

