
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
    Size_f size_f;
    init_Size_F(&size_F, (int)size_Fold[2], (int)size_Fold[3]/2);
    init_Size_f(&size_f, (int)size_Fold[2], (int)size_Fold[3]/2);

    myComplex* Fold_compact = new myComplex[size_F.nTot_compact];
    modify_F(Fold, Fold_compact, true, &size_F);

    Size_F* size_F_dev;
    cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(Size_F)));
    cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(Size_F), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, &size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    // set up output Fnew
    plhs[0] = mxCreateUninitNumericArray(5, (size_t*) size_Fold, mymxRealClass, mxCOMPLEX);
    myComplex* Fnew = (myComplex*) mymxGetComplex(plhs[0]);
    memset(Fnew, 0, size_F.nTot*sizeof(myComplex));

    myComplex* Fnew_compact = new myComplex[size_F.nTot_compact];

    // get f from matlab
    myReal* f = mymxGetReal(prhs[1]);
    
    // get X from matlab
    myComplex* X = (myComplex*) mymxGetComplex(prhs[2]);

    // get MR from matlab
    myReal* mR = mymxGetReal(prhs[3]);

    // get b from matlab
    myReal* b = mymxGetReal(prhs[4]);

    // get G from matlab
    myReal* G = mymxGetReal(prhs[5]);

    // get dt from matlab
    myReal* dt = mymxGetReal(prhs[6]);

    // get L from matlab
    myReal* L = mymxGetReal(prhs[7]);

    // get u from matlab
    myComplex* u = (myComplex*) mymxGetComplex(prhs[8]);

    myComplex* u_compact = new myComplex[2*size_F.nR_compact];
    modify_u(u, u_compact, &size_F);

    // get d from matlab
    myReal* d = mymxGetReal(prhs[9]);

    // get w from matlab
    myReal* w = mymxGetReal(prhs[10]);
    
    // get method from matlab
    char* method;
    method = mxArrayToString(prhs[11]);

    //////////////////////////
    // some pre-calculation //
    //////////////////////////

    myReal* d_dev;
    cudaErrorHandle(cudaMalloc(&d_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(d_dev, d, size_F.nR*size_f.const_2BR*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* w_dev;
    cudaErrorHandle(cudaMalloc(&w_dev, size_f.const_2BR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(w_dev, w, size_f.const_2BR*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* dw_dev;
    cudaErrorHandle(cudaMalloc(&dw_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(dw_dev, 0, size_F.nR*size_f.const_2BR*sizeof(myReal)));

    myReal* dl_dev;
    cudaErrorHandle(cudaMalloc(&dl_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(dl_dev, 0, size_F.nR*size_f.const_2BR*sizeof(myReal)));

    dim3 blocksize_dw(size_F.const_2lp1, 1, 1);
    dim3 gridsize_dw(size_F.const_2lp1, size_F.const_lp1, size_f.const_2BR);
    mul_dw <<<gridsize_dw, blocksize_dw>>> (dw_dev, d_dev, w_dev, size_F_dev);
    mul_dl <<<gridsize_dw, blocksize_dw>>> (dl_dev, d_dev, size_F_dev);

    // free memory
    cudaErrorHandle(cudaFree(d_dev));
    cudaErrorHandle(cudaFree(w_dev));

    //////////////////
    // calculate dF //
    //////////////////

    // set up arrays
    myComplex* dF1;
    myComplex* dF2;
    myComplex* dF3;
    myComplex* dF4;

    myComplex* Fold_dev;

    // set up arrays
    cudaErrorHandle(cudaMalloc(&Fold_dev, size_F.nTot_compact*sizeof(myComplex)));
    cudaErrorHandle(cudaMemcpy(Fold_dev, Fold_compact, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

    // set up blocksize and gridsize
    dim3 blocksize_512_nTot(512, 1, 1);
    dim3 gridsize_512_nTot((int)size_F.nTot_compact/512+1, 1, 1);

    // calculate
    // dF1
    dF1 = new myComplex[size_F.nTot_compact];
    get_dF(dF1, Fold_compact, f, X, mR, b, G, L, u_compact, dw_dev, &size_F, size_F_dev, &size_f, size_f_dev);

    if (stricmp(method,"midpoint") == 0 || stricmp(method,"RK4") == 0) {
        // dF2
        myComplex* F2_dev;
        cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_compact*sizeof(myComplex)));

        myComplex* dF1_dev;
        cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_compact*sizeof(myComplex)));
        cudaErrorHandle(cudaMemcpy(dF1_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0]/2, size_F.nTot_compact);
        cudaErrorHandle(cudaFree(dF1_dev));

        myComplex* F2 = new myComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F2, F2_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));
        cudaErrorHandle(cudaFree(F2_dev));

        myComplex* F2_complete = new myComplex[size_F.nTot];
        modify_F(F2, F2_complete, false, &size_F);
        myReal* f2 = new myReal[size_f.nTot];
        fftSO3R_backward(f2, F2_complete, dl_dev, &size_F, size_F_dev, &size_f, size_f_dev);
        delete[] F2_complete;

        dF2 = new myComplex[size_F.nTot_compact];
        get_dF(dF2, F2, f2, X, mR, b, G, L, u_compact, dw_dev, &size_F, size_F_dev, &size_f, size_f_dev);

        delete[] F2;
        delete[] f2;
    }

    if (stricmp(method,"RK2") == 0) {
        // dF2
        myComplex* F2_dev;
        cudaErrorHandle(cudaMalloc(&F2_dev, size_F.nTot_compact*sizeof(myComplex)));

        myComplex* dF1_dev;
        cudaErrorHandle(cudaMalloc(&dF1_dev, size_F.nTot_compact*sizeof(myComplex)));
        cudaErrorHandle(cudaMemcpy(dF1_dev, dF1, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F2_dev, Fold_dev, dF1_dev, dt[0], size_F.nTot_compact);
        cudaErrorHandle(cudaFree(dF1_dev));

        myComplex* F2 = new myComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F2, F2_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));
        cudaErrorHandle(cudaFree(F2_dev));

        myComplex* F2_complete = new myComplex[size_F.nTot];
        modify_F(F2, F2_complete, false, &size_F);
        myReal* f2 = new myReal[size_f.nTot];
        fftSO3R_backward(f2, F2_complete, dl_dev, &size_F, size_F_dev, &size_f, size_f_dev);
        delete[] F2_complete;

        dF2 = new myComplex[size_F.nTot_compact];
        get_dF(dF2, F2, f2, X, mR, b, G, L, u_compact, dw_dev, &size_F, size_F_dev, &size_f, size_f_dev);

        delete[] F2;
        delete[] f2;
    }

    if (stricmp(method,"RK4") == 0) {
        // dF3
        myComplex* F3_dev;
        cudaErrorHandle(cudaMalloc(&F3_dev, size_F.nTot_compact*sizeof(myComplex)));

        myComplex* dF2_dev;
        cudaErrorHandle(cudaMalloc(&dF2_dev, size_F.nTot_compact*sizeof(myComplex)));
        cudaErrorHandle(cudaMemcpy(dF2_dev, dF2, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F3_dev, Fold_dev, dF2_dev, dt[0]/2, size_F.nTot_compact);
        cudaErrorHandle(cudaFree(dF2_dev));

        myComplex* F3 = new myComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F3, F3_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));
        cudaErrorHandle(cudaFree(F3_dev));

        myComplex* F3_complete = new myComplex[size_F.nTot];
        modify_F(F3, F3_complete, false, &size_F);
        myReal* f3 = new myReal[size_f.nTot];
        fftSO3R_backward(f3, F3_complete, dl_dev, &size_F, size_F_dev, &size_f, size_f_dev);
        delete[] F3_complete;

        dF3 = new myComplex[size_F.nTot_compact];
        get_dF(dF3, F3, f3, X, mR, b, G, L, u_compact, dw_dev, &size_F, size_F_dev, &size_f, size_f_dev);

        delete[] F3;
        delete[] f3;

        // dF4
        myComplex* F4_dev;
        cudaErrorHandle(cudaMalloc(&F4_dev, size_F.nTot_compact*sizeof(myComplex)));

        myComplex* dF3_dev;
        cudaErrorHandle(cudaMalloc(&dF3_dev, size_F.nTot_compact*sizeof(myComplex)));
        cudaErrorHandle(cudaMemcpy(dF3_dev, dF3, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyHostToDevice));

        integrate_Fnew <<<gridsize_512_nTot, blocksize_512_nTot>>> (F4_dev, Fold_dev, dF3_dev, dt[0], size_F.nTot_compact);
        cudaErrorHandle(cudaFree(dF3_dev));

        myComplex* F4 = new myComplex[size_F.nTot_compact];
        cudaErrorHandle(cudaMemcpy(F4, F4_dev, size_F.nTot_compact*sizeof(myComplex), cudaMemcpyDeviceToHost));
        cudaErrorHandle(cudaFree(F4_dev));

        myComplex* F4_complete = new myComplex[size_F.nTot];
        modify_F(F4, F4_complete, false, &size_F);
        myReal* f4 = new myReal[size_f.nTot];
        fftSO3R_backward(f4, F4_complete, dl_dev, &size_F, size_F_dev, &size_f, size_f_dev);
        delete[] F4_complete;

        dF4 = new myComplex[size_F.nTot_compact];
        get_dF(dF4, F4, f4, X, mR, b, G, L, u_compact, dw_dev, &size_F, size_F_dev, &size_f, size_f_dev);

        delete[] F4;
        delete[] f4;
    }

    // free memory
    cudaErrorHandle(cudaFree(dw_dev));
    cudaErrorHandle(cudaFree(dl_dev));

    delete[] u_compact;

    ///////////////
    // integrate //
    ///////////////

    // Fnew = Fold + dt*dF1 (euler)
    // Fnew = Fold + dt*dF2 (midpoint)
    // Fnew = Fold + dt/3*dF1 + dt/6*dF2 + dt/6*dF3 + dt/3*dF4 (RK4)

    // set up GPU arrays
    myComplex* Fnew_dev;
    cudaErrorHandle(cudaMalloc(&Fnew_dev, size_F.nTot_compact*sizeof(myComplex)));
    myComplex* dF_dev;
    cudaErrorHandle(cudaMalloc(&dF_dev, size_F.nTot_compact*sizeof(myComplex)));

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

    // gather Fnew
    modify_F(Fnew_compact, Fnew, false, &size_F);

    // free memory
    cudaErrorHandle(cudaFree(Fold_dev));
    cudaErrorHandle(cudaFree(Fnew_dev));
    cudaErrorHandle(cudaFree(dF_dev));
    cudaErrorHandle(cudaFree(size_F_dev));
    cudaErrorHandle(cudaFree(size_f_dev));

    delete[] Fold_compact;
    delete[] Fnew_compact;
}

