
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include <string.h>

#include "mex.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    ////////////////////////////
    // get arrays from Matlab //
    ////////////////////////////

    // get Fold from matlab
    cuDoubleComplex* Fold = (cuDoubleComplex*) mxGetComplexDoubles(prhs[0]);
    const mwSize* size_Fold = mxGetDimensions(prhs[0]);

    Size_F size_F;
    init_Size_F(&size_F, (int)size_Fold[2], (int)size_Fold[3]/2);

    cuDoubleComplex* Fold_compact = new cuDoubleComplex[size_F.nTot_compact];
    modify_F(Fold, Fold_compact, true, &size_F);

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(Size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(Size_F), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mxDOUBLE_CLASS, mxCOMPLEX);
    cuDoubleComplex* Fnew = (cuDoubleComplex*) mxGetComplexDoubles(plhs[0]);

    cuDoubleComplex* Fnew_compact = new cuDoubleComplex[size_F.nTot_compact];
    
    // get X from matlab
    cuDoubleComplex* X = (cuDoubleComplex*) mxGetComplexDoubles(prhs[1]);

    // get OJO from matlab
    cuDoubleComplex* OJO = (cuDoubleComplex*) mxGetComplexDoubles(prhs[2]);

    // get MR from matlab
    cuDoubleComplex* MR = (cuDoubleComplex*) mxGetComplexDoubles(prhs[3]);

    cuDoubleComplex* MR_compact = new cuDoubleComplex[3*size_F.nR_compact];
    modify_u(MR, MR_compact, &size_F);

    // get dt from matlab
    double* dt = mxGetDoubles(prhs[4]);

    // get L from matlab
    double* L = mxGetDoubles(prhs[5]);

    // get u from matlab
    cuDoubleComplex* u = (cuDoubleComplex*) mxGetComplexDoubles(prhs[6]);

    cuDoubleComplex* u_compact = new cuDoubleComplex[3*size_F.nR_compact];
    modify_u(u, u_compact, &size_F);

    // get CG from matlab
    double** CG = new double* [size_F.BR*size_F.BR];
    for (int l1 = 0; l1 < size_F.BR; l1++) {
        for (int l2 = 0; l2 < size_F.BR; l2++) {
            int ind_CG = l1+l2*size_F.BR;
            CG[ind_CG] = mxGetDoubles(mxGetCell(prhs[7], ind_CG));
        }
    }
    
    // get method from matlab
    char* method;
    method = mxArrayToString(prhs[8]);

    //////////////////
    // calculate dF //
    //////////////////

    // set up arrays
    cuDoubleComplex* dF1;
    cuDoubleComplex* dF2;
    cuDoubleComplex* dF3;
    cuDoubleComplex* dF4;

    cuDoubleComplex* Fold_dev;
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
    cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // set up blocksize and gridsize
    dim3 blocksize_512_nTot(512, 1, 1);
    dim3 gridsize_512_nTot((int)size_F.nTot_compact/512+1, 1, 1);

    // calculate
    // dF1
    dF1 = new cuDoubleComplex[size_F.nTot_compact];
    get_dF(dF1, Fold_compact, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

    if (stricmp(method,"midpoint") == 0 || stricmp(method,"runge-kutta") == 0) {
        // dF2
        cuDoubleComplex* F2_dev;
        cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));

        cuDoubleComplex* dF1_dev;
        cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMemcpy(dF1_dev, dF1, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0]/2, size_F_dev);

        cuDoubleComplex* F2 = new cuDoubleComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F2, F2_dev, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        dF2 = new cuDoubleComplex[size_F.nTot_compact];
        get_dF(dF2, F2, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

        delete[] F2;
        cudaErrorHandle(cudaFree(F2_dev));
        cudaErrorHandle(cudaFree(dF1_dev));
    }
    
    if (stricmp(method,"runge-kutta") == 0) {
        // dF3
        cuDoubleComplex* F3_dev;
        cudaErrorHandle(cudaMalloc(&F3_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));

        cuDoubleComplex* dF2_dev;
        cudaErrorHandle(cudaMalloc(&dF2_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMemcpy(dF2_dev, dF2, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F3_dev, Fold_dev, dF2_dev, dt[0]/2, size_F_dev);

        cuDoubleComplex* F3 = new cuDoubleComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F3, F3_dev, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        dF3 = new cuDoubleComplex[size_F.nTot_compact];
        get_dF(dF3, F3, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

        delete[] F3;
        cudaErrorHandle(cudaFree(F3_dev));
        cudaErrorHandle(cudaFree(dF2_dev));

        // dF4
        cuDoubleComplex* F4_dev;
        cudaErrorHandle(cudaMalloc(&F4_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));

        cuDoubleComplex* dF3_dev;
        cudaErrorHandle(cudaMalloc(&dF3_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));
        cudaErrorHandle(cudaMemcpy(dF3_dev, dF3, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F4_dev, Fold_dev, dF3_dev, dt[0], size_F_dev);

        cuDoubleComplex* F4 = new cuDoubleComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F4, F4_dev, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

        dF4 = new cuDoubleComplex[size_F.nTot_compact];
        get_dF(dF4, F4, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

        delete[] F4;
        cudaErrorHandle(cudaFree(F4_dev));
        cudaErrorHandle(cudaFree(dF3_dev));
    }

    ///////////////
    // integrate //
    ///////////////

    // Fnew = Fold + dt*dF

    // set up arrays
    cuDoubleComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));

    cuDoubleComplex* dF_dev;
    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_compact*sizeof(cuDoubleComplex)));

    // calculate
    if (stricmp(method,"euler") == 0) {
        cudaErrorHandle(cudaMemcpy(dF_dev, dF1, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F_dev);

        delete[] dF1;
    } else if (stricmp(method,"midpoint") == 0) {
        cudaErrorHandle(cudaMemcpy(dF_dev, dF2, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F_dev);

        delete[] dF1;
        delete[] dF2;
    } else if (stricmp(method,"runge-kutta") == 0) {
        cudaErrorHandle(cudaMemcpy(dF_dev, dF1, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/6, size_F_dev);

        cudaErrorHandle(cudaMemcpy(dF_dev, dF2, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/3, size_F_dev);

        cudaErrorHandle(cudaMemcpy(dF_dev, dF3, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/3, size_F_dev);

        cudaErrorHandle(cudaMemcpy(dF_dev, dF4, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/6, size_F_dev);

        delete[] dF1;
        delete[] dF2;
        delete[] dF3;
        delete[] dF4;
    } else {
        mexPrintf("'method' must be 'euler', 'midpoint', or 'runge-kutta'. Return Fold.\n");
        Fnew_dev = Fold_dev;
    }

    // gather Fnew
    cudaErrorHandle(cudaMemcpy(Fnew_compact, Fnew_dev, size_F.nTot_compact*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    modify_F(Fnew_compact, Fnew, false, &size_F);

    // free memory
    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(Fnew_dev));
    cudaErrorHandle(cudaFree(dF_dev));
    cudaErrorHandle(cudaFree(size_F_dev));

    delete[] Fold_compact;
    delete[] Fnew_compact;
    delete[] MR_compact;
    delete[] u_compact;
    delete[] CG;
}

