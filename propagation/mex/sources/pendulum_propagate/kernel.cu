
#include "integrate.cuh"

#include <stdio.h>
#include <iostream>

#include <string.h>

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    ////////////////////////////
    // get arrays from Matlab //
    ////////////////////////////

    // get Fold from matlab
    myComplex* Fold = (myComplex*) mymxGetComplex(prhs[0]);
    const mwSize* size_Fold = mxGetDimensions(prhs[0]);

    Size_F size_F;
    init_Size_F(&size_F, (int)size_Fold[2], (int)size_Fold[3]/2);

    myComplex* Fold_compact = new myComplex[size_F.nTot_compact];
    modify_F(Fold, Fold_compact, true, &size_F);

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(Size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(Size_F), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(6, (size_t*) size_Fold, mymxRealClass, mxCOMPLEX);
    myComplex* Fnew = (myComplex*) mymxGetComplex(plhs[0]);

    myComplex* Fnew_compact = new myComplex[size_F.nTot_compact];
    
    // get X from matlab
    myComplex* X = (myComplex*) mymxGetComplex(prhs[1]);

    // get OJO from matlab
    myComplex* OJO = (myComplex*) mymxGetComplex(prhs[2]);

    // get MR from matlab
    myComplex* MR = (myComplex*) mymxGetComplex(prhs[3]);

    myComplex* MR_compact = new myComplex[3*size_F.nR_compact];
    modify_u(MR, MR_compact, &size_F);

    // get dt from matlab
    myReal* dt = mymxGetReal(prhs[4]);

    // get L from matlab
    myReal* L = mymxGetReal(prhs[5]);

    // get u from matlab
    myComplex* u = (myComplex*) mymxGetComplex(prhs[6]);

    myComplex* u_compact = new myComplex[3*size_F.nR_compact];
    modify_u(u, u_compact, &size_F);

    // get CG from matlab
    myReal** CG = new myReal* [size_F.BR*size_F.BR];
    for (int l1 = 0; l1 < size_F.BR; l1++) {
        for (int l2 = 0; l2 < size_F.BR; l2++) {
            int ind_CG = l1+l2*size_F.BR;
            CG[ind_CG] = mymxGetReal(mxGetCell(prhs[7], ind_CG));
        }
    }
    
    // get method from matlab
    char* method;
    method = mxArrayToString(prhs[8]);

    //////////////////
    // calculate dF //
    //////////////////

    // if the problem is too large, split arrays
    bool issmall = (size_F.BR<=10 && size_F.Bx<=10);

    // set up arrays
    myComplex* dF1;
    myComplex* dF2;
    myComplex* dF3;
    myComplex* dF4;

    myComplex* Fold_dev;

    if (issmall) {
        // set up arrays
        cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_compact*sizeof(myComplex)));
        cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

        // set up blocksize and gridsize
        dim3 blocksize_512_nTot(512, 1, 1);
        dim3 gridsize_512_nTot((int)size_F.nTot_compact/512+1, 1, 1);

        // calculate
        // dF1
        dF1 = new myComplex[size_F.nTot_compact];
        get_dF_small(dF1, Fold_compact, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

        if (stricmp(method,"midpoint") == 0 || stricmp(method,"RK4") == 0) {
            // dF2
            myComplex* F2_dev;
            cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_compact*sizeof(myComplex)));

            myComplex* dF1_dev;
            cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_compact*sizeof(myComplex)));
            cudaErrorHandle(cudaMemcpy(dF1_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0]/2, size_F.nTot_compact);

            myComplex* F2 = new myComplex[size_F.nTot_compact];
            cudaErrorHandle(cudaMemcpy(F2, F2_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));

            dF2 = new myComplex[size_F.nTot_compact];
            get_dF_small(dF2, F2, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F2;
            cudaErrorHandle(cudaFree(F2_dev));
            cudaErrorHandle(cudaFree(dF1_dev));
        }

        if (stricmp(method,"RK2") == 0) {
            // dF2
            myComplex* F2_dev;
            cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_compact*sizeof(myComplex)));

            myComplex* dF1_dev;
            cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_compact*sizeof(myComplex)));
            cudaErrorHandle(cudaMemcpy(dF1_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0], size_F.nTot_compact);

            myComplex* F2 = new myComplex[size_F.nTot_compact];
            cudaErrorHandle(cudaMemcpy(F2, F2_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));

            dF2 = new myComplex[size_F.nTot_compact];
            get_dF_small(dF2, F2, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F2;
            cudaErrorHandle(cudaFree(F2_dev));
            cudaErrorHandle(cudaFree(dF1_dev));
        }

        if (stricmp(method,"RK4") == 0) {
            // dF3
            myComplex* F3_dev;
            cudaErrorHandle(cudaMalloc(&F3_dev, size_F.nTot_compact*sizeof(myComplex)));

            myComplex* dF2_dev;
            cudaErrorHandle(cudaMalloc(&dF2_dev, size_F.nTot_compact*sizeof(myComplex)));
            cudaErrorHandle(cudaMemcpy(dF2_dev, dF2, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F3_dev, Fold_dev, dF2_dev, dt[0]/2, size_F.nTot_compact);

            myComplex* F3 = new myComplex[size_F.nTot_compact];
            cudaErrorHandle(cudaMemcpy(F3, F3_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));

            dF3 = new myComplex[size_F.nTot_compact];
            get_dF_small(dF3, F3, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F3;
            cudaErrorHandle(cudaFree(F3_dev));
            cudaErrorHandle(cudaFree(dF2_dev));

            // dF4
            myComplex* F4_dev;
            cudaErrorHandle(cudaMalloc(&F4_dev, size_F.nTot_compact*sizeof(myComplex)));

            myComplex* dF3_dev;
            cudaErrorHandle(cudaMalloc(&dF3_dev, size_F.nTot_compact*sizeof(myComplex)));
            cudaErrorHandle(cudaMemcpy(dF3_dev, dF3, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F4_dev, Fold_dev, dF3_dev, dt[0], size_F.nTot_compact);

            myComplex* F4 = new myComplex[size_F.nTot_compact];
            cudaErrorHandle(cudaMemcpy(F4, F4_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));

            dF4 = new myComplex[size_F.nTot_compact];
            get_dF_small(dF4, F4, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F4;
            cudaErrorHandle(cudaFree(F4_dev));
            cudaErrorHandle(cudaFree(dF3_dev));
        }
    } else {
        // set up arrays
        cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_splitx*sizeof(myComplex)));

        // set up blocksize and gridsize
        dim3 blocksize_512_nTot(512, 1, 1);
        dim3 gridsize_512_nTot((int)size_F.nTot_splitx/512+1, 1, 1);
    
        // calculate
        // dF1
        dF1 = new myComplex[size_F.nTot_compact];
        get_dF_large(dF1, Fold_compact, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

        if (stricmp(method,"midpoint") == 0 || stricmp(method,"RK4") == 0) {
            // dF2
            myComplex* F2_dev;
            cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* dF1_dev;
            cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* F2 = new myComplex[size_F.nTot_compact];

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF1_dev, dF1+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0]/2, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(F2+ind_F, F2_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            dF2 = new myComplex[size_F.nTot_compact];
            get_dF_large(dF2, F2, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F2;
            cudaErrorHandle(cudaFree(F2_dev));
            cudaErrorHandle(cudaFree(dF1_dev));
        }

        if (stricmp(method,"RK2") == 0) {
            // dF2
            myComplex* F2_dev;
            cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* dF1_dev;
            cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* F2 = new myComplex[size_F.nTot_compact];

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF1_dev, dF1+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0], size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(F2+ind_F, F2_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            dF2 = new myComplex[size_F.nTot_compact];
            get_dF_large(dF2, F2, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F2;
            cudaErrorHandle(cudaFree(F2_dev));
            cudaErrorHandle(cudaFree(dF1_dev));
        }

        if (stricmp(method,"RK4") == 0) {
            // dF3
            myComplex* F3_dev;
            cudaErrorHandle(cudaMalloc(&F3_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* dF2_dev;
            cudaErrorHandle(cudaMalloc(&dF2_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* F3 = new myComplex[size_F.nTot_compact];

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF2_dev, dF2+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F3_dev, Fold_dev, dF2_dev, dt[0]/2, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(F3+ind_F, F3_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            dF3 = new myComplex[size_F.nTot_compact];
            get_dF_large(dF3, F3, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F3;
            cudaErrorHandle(cudaFree(F3_dev));
            cudaErrorHandle(cudaFree(dF2_dev));

            // dF4
            myComplex* F4_dev;
            cudaErrorHandle(cudaMalloc(&F4_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* dF3_dev;
            cudaErrorHandle(cudaMalloc(&dF3_dev, size_F.nTot_splitx*sizeof(myComplex)));

            myComplex* F4 = new myComplex[size_F.nTot_compact];

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF3_dev, dF3+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F4_dev, Fold_dev, dF3_dev, dt[0], size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(F4+ind_F, F4_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            dF4 = new myComplex[size_F.nTot_compact];
            get_dF_large(dF4, F4, X, OJO, MR_compact, L, u_compact, CG, &size_F, size_F_dev);

            delete[] F4;
            cudaErrorHandle(cudaFree(F4_dev));
            cudaErrorHandle(cudaFree(dF3_dev));
        }
    }

    // free memory
    delete[] MR_compact;
    delete[] u_compact;
    delete[] CG;

    ///////////////
    // integrate //
    ///////////////

    // Fnew = Fold + dt*dF1 (euler)
    // Fnew = Fold + dt*dF2 (midpoint)
    // Fnew = Fold + dt/3*dF1 + dt/6*dF2 + dt/6*dF3 + dt/3*dF4 (RK4)

    // set up GPU arrays
    myComplex* Fnew_dev;
    myComplex* dF_dev;

    // calculate
    if (issmall) {
        // set up arrays
        cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_compact*sizeof(myComplex)));
        cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_compact*sizeof(myComplex)));

        // set up blocksize and gridsize
        dim3 blocksize_512_nTot(512, 1, 1);
        dim3 gridsize_512_nTot((int)size_F.nTot_compact/512+1, 1, 1);

        // calculate
        if (stricmp(method,"euler") == 0) {
            cudaErrorHandle(cudaMemcpy(dF_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F.nTot_compact);

            delete[] dF1;
        } else if (stricmp(method,"midpoint") == 0) {
            cudaErrorHandle(cudaMemcpy(dF_dev, dF2, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F.nTot_compact);

            delete[] dF1;
            delete[] dF2;
        } else if (stricmp(method,"RK2") == 0) {
            cudaErrorHandle(cudaMemcpy(dF_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/2, size_F.nTot_compact);

            cudaErrorHandle(cudaMemcpy(dF_dev, dF2, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/2, size_F.nTot_compact);

            delete[] dF1;
            delete[] dF2;
        } else if (stricmp(method,"RK4") == 0) {
            cudaErrorHandle(cudaMemcpy(dF_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/6, size_F.nTot_compact);

            cudaErrorHandle(cudaMemcpy(dF_dev, dF2, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/3, size_F.nTot_compact);

            cudaErrorHandle(cudaMemcpy(dF_dev, dF3, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/3, size_F.nTot_compact);

            cudaErrorHandle(cudaMemcpy(dF_dev, dF4, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));
            integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fnew_dev, dF_dev, dt[0]/6, size_F.nTot_compact);

            delete[] dF1;
            delete[] dF2;
            delete[] dF3;
            delete[] dF4;
        } else {
            mexPrintf("'method' must be 'euler', 'midpoint', 'RK2', or 'RK4'. Return Fold.\n");
            Fnew_dev = Fold_dev;
        }

        cudaErrorHandle(cudaMemcpy(Fnew_compact, Fnew_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));
    } else {
        // set up arrays
        cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_splitx*sizeof(myComplex)));
        cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_splitx*sizeof(myComplex)));

        // set up blocksize and gridsize
        dim3 blocksize_512_nTot(512, 1, 1);
        dim3 gridsize_512_nTot((int)size_F.nTot_splitx/512+1, 1, 1);

        // calculate
        if (stricmp(method,"euler") == 0) {
            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF1+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            delete[] dF1;
        } else if (stricmp(method,"midpoint") == 0) {
            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF2+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0], size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            delete[] dF1;
            delete[] dF2;
        } else if (stricmp(method,"RK2") == 0) {
            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF1+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/2, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fnew_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF2+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/2, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            delete[] dF1;
            delete[] dF2;
        } else if (stricmp(method,"RK4") == 0) {
            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF1+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/6, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fnew_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF2+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/3, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fnew_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF3+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/3, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            for (int k = 0; k < size_F.const_2Bx; k++) {
                int ind_F = k*size_F.nTot_splitx;
                cudaErrorHandle(cudaMemcpy(Fold_dev, Fnew_compact+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));
                cudaErrorHandle(cudaMemcpy(dF_dev, dF4+ind_F, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyHostToDevice));

                integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (Fnew_dev, Fold_dev, dF_dev, dt[0]/6, size_F.nTot_splitx);
                cudaErrorHandle(cudaGetLastError());

                cudaErrorHandle(cudaMemcpy(Fnew_compact+ind_F, Fnew_dev, size_F.nTot_splitx*sizeof(myComplex), cudaMemcpyDeviceToHost));
            }

            delete[] dF1;
            delete[] dF2;
            delete[] dF3;
            delete[] dF4;
        } else {
            mexPrintf("'method' must be 'euler', 'midpoint', 'RK2', or 'RK4'. Return Fold.\n");
            Fnew_dev = Fold_dev;
        }
    }

    // gather Fnew
    modify_F(Fnew_compact, Fnew, false, &size_F);

    // free memory
    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(Fnew_dev));
    cudaErrorHandle(cudaFree(dF_dev));
    cudaErrorHandle(cudaFree(size_F_dev));

    delete[] Fold_compact;
    delete[] Fnew_compact;
}

