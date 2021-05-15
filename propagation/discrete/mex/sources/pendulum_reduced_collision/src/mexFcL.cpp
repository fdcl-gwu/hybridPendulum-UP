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
    int nn0 = size_Omega[1];

    myReal* lambda = mymxGetReal(prhs[2]);

    int** lambda_indx = (int**) malloc(nn0*sizeof(int*));
    int* lambda_numx = (int*) malloc(nn0*sizeof(int));
    for (int iR = 0; iR < nn0; iR++) {
        lambda_indx[iR] = mxGetInt32s(mxGetCell(prhs[3], iR));
        const mwSize* size_ind = mxGetDimensions(mxGetCell(prhs[3], iR));
        lambda_numx[iR] = size_ind[0];
    }

    myReal* Gd = mymxGetReal(prhs[4]); 

    // compute
    myReal*** fcL = (myReal***) malloc(nn0*sizeof(myReal**));
    int** fcL_indx1 = (int**) malloc(nn0*sizeof(int*));
    int* fcL_numx1 = (int*) malloc(nn0*sizeof(int));
    int*** fcL_indx2 = (int***) malloc(nn0*sizeof(int**));
    int** fcL_numx2 = (int**) malloc(nn0*sizeof(int*)); 

    getFcL(fcL, fcL_indx1, fcL_numx1, fcL_indx2, fcL_numx2, x, Omega, lambda, nn0, lambda_indx, lambda_numx, Gd, &size_f);

    // setup output
    mwSize size_R[1] = {(mwSize)nn0};
    plhs[0] = mxCreateCellArray(1, size_R);
    plhs[1] = mxCreateCellArray(1, size_R);
    plhs[2] = mxCreateCellArray(1, size_R);

    for (int iR = 0; iR < nn0; iR++) {
        // fcL_indx1
        mwSize size_x1[1] = {(mwSize)fcL_numx1[iR]};
        mxArray* fcL_indx1_mx = mxCreateNumericArray(1, size_x1, mxINT32_CLASS, mxREAL);
        mxSetCell(plhs[1], iR, fcL_indx1_mx);
        
        int* fcL_indx1_temp = mxGetInt32s(fcL_indx1_mx);
        memcpy(fcL_indx1_temp, fcL_indx1[iR], fcL_numx1[iR]*sizeof(int));
        free(fcL_indx1[iR]);

        // fcL, fcL_indx2
        mxArray* fcL_temp = mxCreateCellArray(1, size_x1);
        mxArray* fcL_indx2_temp = mxCreateCellArray(1, size_x1);
        
        for (int ix = 0; ix < fcL_numx1[iR]; ix++) {
            mwSize size_x2[1] = {(mwSize) fcL_numx2[iR][ix]};
            mxArray* fcL_mx = mxCreateNumericArray(1, size_x2, mymxRealClass, mxREAL);
            mxArray* fcL_indx2_mx = mxCreateNumericArray(1, size_x2, mxINT32_CLASS, mxREAL);
            mxSetCell(fcL_temp, ix, fcL_mx);
            mxSetCell(fcL_indx2_temp, ix, fcL_indx2_mx);
            
            myReal* fcL_temp_temp = mymxGetReal(fcL_mx);
            int* fcL_indx2_temp_temp = mxGetInt32s(fcL_indx2_mx);
            memcpy(fcL_temp_temp, fcL[iR][ix], fcL_numx2[iR][ix]*sizeof(myReal));
            memcpy(fcL_indx2_temp_temp, fcL_indx2[iR][ix], fcL_numx2[iR][ix]*sizeof(int));
            free(fcL[iR][ix]);
            free(fcL_indx2[iR][ix]);    
        }

        mxSetCell(plhs[0], iR, fcL_temp);
        mxSetCell(plhs[2], iR, fcL_indx2_temp);

        free(fcL[iR]);
        free(fcL_indx2[iR]);
        free(fcL_numx2[iR]);
    }

    // free memory
    free(lambda_indx);
    free(lambda_numx);
    free(fcL);
    free(fcL_indx1);
    free(fcL_numx1);
    free(fcL_indx2);
    free(fcL_numx2);
}


