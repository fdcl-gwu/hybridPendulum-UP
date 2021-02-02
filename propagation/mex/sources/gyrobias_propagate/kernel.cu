
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    ///////////////////////////////
    // set up inputs and outputs //
    ///////////////////////////////

    // get Fold from matlab
    cuDoubleComplex* Fold = (cuDoubleComplex*) mxGetComplexDoubles(prhs[0]);
    const mwSize* size_Fold = mxGetDimensions(prhs[0]);

    Size_F size_F;
    init_Size_F(&size_F, (int)size_Fold[2], (int)size_Fold[3]/2);

    cuDoubleComplex* Fold_compact = new cuDoubleComplex[size_F.nTot_compact];
    modify_F(Fold, Fold_compact, true, &size_F);

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(size_F), cudaMemcpyHostToDevice));
    
    // get X from matlab
    cuDoubleComplex* X = (cuDoubleComplex*) mxGetComplexDoubles(prhs[1]);

    // get dt from matlab
    double* dt = mxGetDoubles(prhs[2]);

    // get u from matlab
    cuDoubleComplex* u = (cuDoubleComplex*) mxGetComplexDoubles(prhs[3]);

    cuDoubleComplex* u_compact = new cuDoubleComplex[3*size_F.nR_compact];
    modify_u(u, u_compact, &size_F);

    // get G1 from matlab
    double* G1 = mxGetDoubles(prhs[4]);

    // get G2 from matlab
    double* G2 = mxGetDoubles(prhs[5]);

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fnew_compact = new cuDoubleComplex[size_F.nTot_compact];

    //////////////////
    // calculate dF //
    //////////////////

    // if the problem is too large, split arrays
    bool issmall = (size_F.BR<=10 && size_F.Bx<=15);

    // set up CPU arrays
    cuDoubleComplex* dF = new cuDoubleComplex[size_F.nTot_compact];

    // calculate
    if (issmall) {
        get_dF_small(dF, Fold_compact, X, u_compact, G1, G2, &size_F, size_F_dev);
    } else {
        get_dF_large(dF, Fold_compact, X, u_compact, G1, G2, &size_F, size_F_dev);
    }

    ///////////////
    // integrate //
    ///////////////

    // Fnew = Fold + dt*dF

    // set up GPU arrays
    cuDoubleComplex* Fold_dev;
    cuDoubleComplex* dF_dev;
    cuDoubleComplex* Fnew_dev;

    // calculate
    if (issmall) {
        cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMemcpy(dF_dev, dF, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));

        // set up block and grid sizes
        dim3 blocksize_512_nTot(512, 1, 1);
        dim3 gridsize_512_nTot((int)size_F.nTot_compact/512+1, 1, 1);

        // Fnew = Fold + dt*dF
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F.nTot_compact);
        cudaErrorHandle(cudaMemcpy(Fnew_compact, Fnew_dev, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    } else {
        // set up GPU arrays
        cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex)));

        // set up block and grid sizes
        dim3 blocksize_512_nTot(512, 1, 1);
        dim3 gridsize_512_nTot((int)size_F.nTot_splitx/512+1, 1, 1);

        // calculate
        for (int k = 0; k < size_F.const_2Bx; k++) {
            int ind_F = k*size_F.nTot_splitx;
            cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            cudaErrorHandle(cudaMemcpy(dF_dev, dF+ind_F, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F.nTot_splitx);
            cudaErrorHandle(cudaGetLastError());

            cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        }
    }

    // gather Fnew
    modify_F(Fnew_compact, Fnew, false, &size_F);

    // free memeory
    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(dF_dev));
    cudaErrorHandle(cudaFree(Fnew_dev));
    cudaErrorHandle(cudaFree(size_F_dev));

    delete[] Fold_compact;
    delete[] Fnew_compact;
    delete[] u_compact;
    delete[] dF;
}

