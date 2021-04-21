#include "setup.cuh"
#include "mex.h"
#include "string.h"

void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // determine forward or backward transform
    char* direction = mxArrayToString(prhs[0]);
    myReal* d = mymxGetReal(prhs[2]);

    if (strcasecmp(direction,"forward") == 0) {
        // determine dimensions
        myReal* f = mymxGetReal(prhs[1]);
        myReal* w = mymxGetReal(prhs[3]);

        const mwSize ndims = mxGetNumberOfDimensions(prhs[1]);
        const mwSize* size_in = mxGetDimensions(prhs[1]);
        int BR = (int) size_in[0]/2;
        int Bx = (int) size_in[3]/2;

        Size_F size_F;
        Size_f size_f;
        init_Size_F(&size_F, BR, Bx, (int) ndims);
        init_Size_f(&size_f, BR, Bx, (int) ndims);

        Size_f* size_f_dev;
        cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
        cudaErrorHandle(cudaMemcpy(size_f_dev, &size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

        Size_F* size_F_dev;
        cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(Size_F)));
        cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(Size_F), cudaMemcpyHostToDevice));

        // set up output
        size_t size_out[6] = {size_F.const_2lp1, size_F.const_2lp1, size_F.const_lp1, size_F.const_2Bx, size_F.const_2Bx, size_F.const_2Bx};
        plhs[0] = mxCreateUninitNumericArray(ndims, (size_t*) size_out, mymxRealClass, mxCOMPLEX);
        myComplex* F = (myComplex*) mymxGetComplex(plhs[0]);

        // initialize GPU variables
        myReal* f_dev;
        cudaErrorHandle(cudaMalloc(&f_dev, size_f.nTot*sizeof(myReal)));
        cudaErrorHandle(cudaMemcpy(f_dev, f, size_f.nTot*sizeof(myReal), cudaMemcpyHostToDevice));

        myComplex* F1_dev;
        cudaErrorHandle(cudaMalloc(&F1_dev, size_f.nTot*sizeof(myComplex)));

        // Fourier transform for R1 and R3
        cufftHandle cufft_plan;
        int n[3] = {size_f.const_2BR, size_f.const_2BR};
        int inembed[3] = {size_f.nR, size_f.const_2BRs};
        int onembed[3] = {size_f.nR, size_f.const_2BRs};
        int istride = 1;
        int ostride = 1;
        int idist = size_f.nR;
        int odist = size_f.nR;

        cufftErrorHandle(cufftPlanMany(&cufft_plan, 2, n, inembed, istride, idist, onembed, ostride, odist, myfftForwardType_R, size_f.nx));
        for (int j = 0; j < size_f.const_2BR; j++) {
            int ind_f = j*size_f.const_2BR;
            cufftErrorHandle(myfftForwardExec_R(cufft_plan, (myfftReal*) f_dev+ind_f, (myfftComplex*) F1_dev+ind_f));
        }

        dim3 blocksize_supplement_R(BR-1, 1, 1);
        dim3 gridsize_supplement_R(size_f.const_2BR, size_f.const_2BR, size_f.nx);
        supplement_R <<<gridsize_supplement_R, blocksize_supplement_R>>> (F1_dev, size_f_dev);

        cufftErrorHandle(cufftDestroy(cufft_plan));

        // Fourier transform for x
        n[0] = size_f.const_2Bx; n[1] = size_f.const_2Bx; n[2] = size_f.const_2Bx;
        inembed[0] = size_f.const_2Bx; inembed[1] = size_f.const_2Bx; inembed[2] = size_f.const_2Bx;
        onembed[0] = size_f.const_2Bx; onembed[1] = size_f.const_2Bx; onembed[2] = size_f.const_2Bx;
        istride = size_f.nR;
        ostride = size_f.nR;
        idist = 1;
        odist = 1;

        cufftErrorHandle(cufftPlanMany(&cufft_plan, (int)ndims-3, n, inembed, istride, idist, onembed, ostride, odist, myfftForwardType_x, 1));
        for (int i = 0; i < size_f.const_2BR; i++) {
            for (int j = 0; j < size_f.const_2BR; j++) {
                for (int k = 0; k < size_f.const_2BR; k++) {
                    int ind_f = i + k*size_f.const_2BR + j*size_f.const_2BRs;
                    cufftErrorHandle(myfftForwardExec_x(cufft_plan, (myfftComplex*) F1_dev+ind_f, (myfftComplex*) F1_dev+ind_f, CUFFT_FORWARD));
                }
            }
        }

        cufftErrorHandle(cufftDestroy(cufft_plan));

        // fftshift and flip
        dim3 blocksize_flip(size_f.const_2BR, 1, 1);
        dim3 gridsize_flip(size_f.const_2BRs, size_f.nx, 1);

        shiftflip <<<gridsize_flip, blocksize_flip, size_f.const_2BR*sizeof(myComplex)>>> (F1_dev, 1, size_f_dev);
        shiftflip <<<gridsize_flip, blocksize_flip, size_f.const_2BR*sizeof(myComplex)>>> (F1_dev, 3, size_f_dev);

        // Fourier transform for R2
        myReal* d_dev;
        cudaErrorHandle(cudaMalloc(&d_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
        cudaErrorHandle(cudaMemcpy(d_dev, d, size_F.nR*size_f.const_2BR*sizeof(myReal), cudaMemcpyHostToDevice));

        myReal* w_dev;
        cudaErrorHandle(cudaMalloc(&w_dev, size_f.const_2BR*sizeof(myReal)));
        cudaErrorHandle(cudaMemcpy(w_dev, w, size_f.const_2BR*sizeof(myReal), cudaMemcpyHostToDevice));

        myReal* dw_dev;
        cudaErrorHandle(cudaMalloc(&dw_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
        cudaErrorHandle(cudaMemset(dw_dev, 0, size_F.nR*size_f.const_2BR*sizeof(myReal)));

        dim3 blocksize_dw(size_F.const_2lp1, 1, 1);
        dim3 gridsize_dw(size_F.const_2lp1, size_F.const_lp1, size_f.const_2BR);
        mul_dw <<<gridsize_dw, blocksize_dw>>> (dw_dev, d_dev, w_dev, size_F_dev);

        myComplex* F_dev;
        cudaErrorHandle(cudaMalloc(&F_dev, size_F.nx*sizeof(myComplex)));
        cudaErrorHandle(cudaMemset(F_dev, 0, size_F.nx*sizeof(myComplex)));

        int mode_F1[2] = {'k','x'};
        int mode_dw[1] = {'k'};
        int mode_F[1] = {'x'};

        int64_t extent_F1[2] = {size_f.const_2BR, size_f.nx};
        int64_t extent_dw[1] = {size_f.const_2BR};
        int64_t extent_F[1] = {size_f.nx};

        int64_t stride_F1[2] = {size_f.const_2BR, size_f.nR};
        int64_t stride_dw[1] = {size_F.nR};
        int64_t stride_F[1] = {1};

        cutensorHandle_t cutensor_handle;
        cutensorInit(&cutensor_handle);

        cutensorTensorDescriptor_t desc_F1;
        cutensorTensorDescriptor_t desc_dw;
        cutensorTensorDescriptor_t desc_F;
        cutensorErrorHandle(cutensorInitTensorDescriptor(&cutensor_handle, &desc_F1, 2, extent_F1, stride_F1, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));
        cutensorErrorHandle(cutensorInitTensorDescriptor(&cutensor_handle, &desc_dw, 1, extent_dw, stride_dw, mycutensor_Realtype, CUTENSOR_OP_IDENTITY));
        cutensorErrorHandle(cutensorInitTensorDescriptor(&cutensor_handle, &desc_F, 1, extent_F, stride_F, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));

        uint32_t alignment_F1;
        uint32_t alignment_dw;
        uint32_t alignment_F;
        cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, F1_dev, &desc_F1, &alignment_F1));
        cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, dw_dev, &desc_dw, &alignment_dw));
        cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, F_dev, &desc_F, &alignment_F));

        cutensorContractionDescriptor_t desc;
        cutensorErrorHandle(cutensorInitContractionDescriptor(&cutensor_handle, &desc, &desc_F1, mode_F1, alignment_F1,
            &desc_dw, mode_dw, alignment_dw,
            &desc_F, mode_F, alignment_F,
            &desc_F, mode_F, alignment_F, mycutensor_computetype));

        cutensorContractionFind_t find;
        cutensorErrorHandle(cutensorInitContractionFind(&cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));

        size_t worksize;
        cutensorErrorHandle(cutensorContractionGetWorkspace(&cutensor_handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));
        void* work = nullptr;
        if (worksize > 0) {
            cudaErrorHandle(cudaMalloc(&work, worksize));
        }

        cutensorContractionPlan_t cutensor_plan;
        cutensorErrorHandle(cutensorInitContractionPlan(&cutensor_handle, &cutensor_plan, &desc, &find, worksize));

        myComplex alpha = make_myComplex(1.0, 0.0);
        myComplex beta = make_myComplex(0.0, 0.0);

        for (int l = 0; l <= size_F.lmax; l++) {
            for (int m = -l; m <= l; m++) {
                for (int n = -l; n <= l; n++) {
                    int ind_F1 = m+size_F.lmax + (n+size_F.lmax)*size_f.const_2BRs;
                    int ind_F = m+size_F.lmax + (n+size_F.lmax)*size_F.const_2lp1 + l*size_F.const_2lp1s;

                    cutensorErrorHandle(cutensorContraction(&cutensor_handle, &cutensor_plan, &alpha, F1_dev+ind_F1, dw_dev+ind_F,
                        &beta, F_dev, F_dev, work, worksize, 0));

                    cudaErrorHandle(cudaMemcpy2D(F+ind_F, size_F.nR*sizeof(myComplex), F_dev, sizeof(myComplex), sizeof(myComplex), size_F.nx, cudaMemcpyDeviceToHost));
                }
            }
        }

        // free memory
        cudaErrorHandle(cudaFree(size_f_dev));
        cudaErrorHandle(cudaFree(size_F_dev));
        cudaErrorHandle(cudaFree(f_dev));
        cudaErrorHandle(cudaFree(F1_dev));
        cudaErrorHandle(cudaFree(d_dev));
        cudaErrorHandle(cudaFree(w_dev));
        cudaErrorHandle(cudaFree(dw_dev));
        cudaErrorHandle(cudaFree(F_dev));

        if (worksize > 0) {
            cudaErrorHandle(cudaFree(work));
        }
    } else {
        myComplex* F = (myComplex*) mymxGetComplex(prhs[1]);

        const mwSize ndims = mxGetNumberOfDimensions(prhs[1]);
        const mwSize* size_in = mxGetDimensions(prhs[1]);
        int BR = (int) size_in[2];
        int Bx = (int) size_in[3]/2;

        Size_F size_F;
        Size_f size_f;
        init_Size_F(&size_F, BR, Bx, (int) ndims);
        init_Size_f(&size_f, BR, Bx, (int) ndims);

        Size_f* size_f_dev;
        cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
        cudaErrorHandle(cudaMemcpy(size_f_dev, &size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

        Size_F* size_F_dev;
        cudaErrorHandle(cudaMalloc(&size_F_dev, sizeof(Size_F)));
        cudaErrorHandle(cudaMemcpy(size_F_dev, &size_F, sizeof(Size_F), cudaMemcpyHostToDevice));

        // inverse Fourier transform for R2
        myComplex* F_dev;
        cudaErrorHandle(cudaMalloc(&F_dev, size_F.nTot*sizeof(myComplex)));
        cudaErrorHandle(cudaMemcpy(F_dev, F, size_F.nTot*sizeof(myComplex), cudaMemcpyHostToDevice));

        myReal* d_dev;
        cudaErrorHandle(cudaMalloc(&d_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
        cudaErrorHandle(cudaMemcpy(d_dev, d, size_F.nR*size_f.const_2BR*sizeof(myReal), cudaMemcpyHostToDevice));

        myReal* dl_dev;
        cudaErrorHandle(cudaMalloc(&dl_dev, size_F.nR*size_f.const_2BR*sizeof(myReal)));
        cudaErrorHandle(cudaMemset(dl_dev, 0, size_F.nR*size_f.const_2BR*sizeof(myReal)));

        dim3 blocksize_dl(size_F.const_2lp1, 1, 1);
        dim3 gridsize_dl(size_F.const_2lp1, size_F.const_lp1, size_f.const_2BR);
        mul_dl <<<gridsize_dl, blocksize_dl>>> (dl_dev, d_dev, size_F_dev);

        myComplex* F1_dev;
        cudaErrorHandle(cudaMalloc(&F1_dev, size_f.nTot*sizeof(myComplex)));
        cudaErrorHandle(cudaMemset(F1_dev, 0, size_f.nTot*sizeof(myComplex)));

        myComplex* F1_temp_dev;
        cudaErrorHandle(cudaMalloc(&F1_temp_dev, size_f.nx*sizeof(myComplex)));
        cudaErrorHandle(cudaMemset(F1_temp_dev, 0, size_f.nx*sizeof(myComplex)));

        cutensorHandle_t cutensor_handle;
        cutensorInit(&cutensor_handle);

        int mode_F[2] = {'l','x'};
        int mode_dl[1] = {'l'};
        int mode_F1[1] = {'x'};

        size_t* worksize = new size_t[size_F.BR];
        cutensorContractionPlan_t* cutensor_plan = new cutensorContractionPlan_t[size_F.BR];

        for (int l = 0; l <= size_F.lmax; l++) {
            int64_t extent_F[2] = {size_F.BR-l, size_f.nx};
            int64_t extent_dl[1] = {size_F.BR-l};
            int64_t extent_F1[1] = {size_f.nx};

            int64_t stride_F[2] = {size_F.const_2lp1s, size_F.nR};
            int64_t stride_dl[1] = {size_F.const_2lp1s};
            int64_t stride_F1[1] = {1};

            cutensorTensorDescriptor_t desc_F;
            cutensorTensorDescriptor_t desc_dl;
            cutensorTensorDescriptor_t desc_F1;
            cutensorErrorHandle(cutensorInitTensorDescriptor(&cutensor_handle, &desc_F, 2, extent_F, stride_F, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));
            cutensorErrorHandle(cutensorInitTensorDescriptor(&cutensor_handle, &desc_dl, 1, extent_dl, stride_dl, mycutensor_Realtype, CUTENSOR_OP_IDENTITY));
            cutensorErrorHandle(cutensorInitTensorDescriptor(&cutensor_handle, &desc_F1, 1, extent_F1, stride_F1, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));

            uint32_t alignment_F;
            uint32_t alignment_dl;
            uint32_t alignment_F1;
            cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, F_dev, &desc_F, &alignment_F));
            cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, dl_dev, &desc_dl, &alignment_dl));
            cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, F1_temp_dev, &desc_F1, &alignment_F1));

            cutensorContractionDescriptor_t desc;
            cutensorErrorHandle(cutensorInitContractionDescriptor(&cutensor_handle, &desc, &desc_F, mode_F, alignment_F,
                &desc_dl, mode_dl, alignment_dl,
                &desc_F1, mode_F1, alignment_F1,
                &desc_F1, mode_F1, alignment_F1, mycutensor_computetype));

            cutensorContractionFind_t find;
            cutensorErrorHandle(cutensorInitContractionFind(&cutensor_handle, &find, CUTENSOR_ALGO_DEFAULT));

            cutensorErrorHandle(cutensorContractionGetWorkspace(&cutensor_handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, worksize+l));
            
            cutensorErrorHandle(cutensorInitContractionPlan(&cutensor_handle, cutensor_plan+l, &desc, &find, worksize[l]));
        }

        size_t worksize_max = 0;
        for (int l = 0; l <= size_F.lmax; l++) {
            worksize_max = (worksize_max < worksize[l]) ? worksize[l] : worksize_max;
        }

        void* work = nullptr;
        if (worksize_max > 0) {
            cudaErrorHandle(cudaMalloc(&work, worksize_max));
        }

        myComplex alpha = make_myComplex((myReal) 1.0/size_f.nx, 0.0);
        myComplex beta = make_myComplex(0.0, 0.0);

        for (int m = -size_F.lmax; m <= size_F.lmax; m++) {
            for (int n = -size_F.lmax; n <= size_F.lmax; n++) {
                int mp = (m>=0) ? m : -m;
                int np = (n>=0) ? n : -n;
                int lmin = (mp >= np) ? mp : np;

                for (int k = 0; k < size_f.const_2BR; k++) {
                    int ind_F = m+size_F.lmax + (n+size_F.lmax)*size_F.const_2lp1 + lmin*size_F.const_2lp1s;
                    int ind_dl = ind_F + k*size_F.nR;
                    int ind_F1 = m+size_F.lmax + k*size_f.const_2BR + (n+size_F.lmax)*size_f.const_2BRs;

                    cutensorErrorHandle(cutensorContraction(&cutensor_handle, cutensor_plan+lmin, &alpha, F_dev+ind_F, dl_dev+ind_dl, &beta, F1_temp_dev, F1_temp_dev, work, worksize[lmin], 0));
                    cudaErrorHandle(cudaMemcpy2D(F1_dev+ind_F1, size_f.nR*sizeof(myComplex), F1_temp_dev, sizeof(myComplex), sizeof(myComplex), size_F.nx, cudaMemcpyDeviceToDevice));
                }
            }
        }

        // fftshift and flip
        dim3 blocksize_flip(size_f.const_2BR, 1, 1);
        dim3 gridsize_flip(size_f.const_2BRs, size_f.nx, 1);

        shiftflip <<<gridsize_flip, blocksize_flip, size_f.const_2BR*sizeof(myComplex)>>> (F1_dev, 1, size_f_dev);
        shiftflip <<<gridsize_flip, blocksize_flip, size_f.const_2BR*sizeof(myComplex)>>> (F1_dev, 3, size_f_dev);

        // inverse Fourier transform for x
        cufftHandle cufft_plan;
        int n[3] = {size_f.const_2Bx, size_f.const_2Bx, size_f.const_2Bx};
        int inembed[3] = {size_f.const_2Bx, size_f.const_2Bx, size_f.const_2Bx};
        int onembed[3] = {size_f.const_2Bx, size_f.const_2Bx, size_f.const_2Bx};
        int istride = size_f.nR;
        int ostride = size_f.nR;
        int idist = 1;
        int odist = 1;

        cufftErrorHandle(cufftPlanMany(&cufft_plan, (int)ndims-3, n, inembed, istride, idist, onembed, ostride, odist, myfftBackwardType_x, 1));
        for (int i = 0; i < size_f.const_2BR; i++) {
            for (int j = 0; j < size_f.const_2BR; j++) {
                for (int k = 0; k < size_f.const_2BR; k++) {
                    int ind_f = i + k*size_f.const_2BR + j*size_f.const_2BRs;
                    cufftErrorHandle(myfftBackwardExec_x(cufft_plan, (myfftComplex*) F1_dev+ind_f, (myfftComplex*) F1_dev+ind_f, CUFFT_INVERSE));
                }
            }
        }

        cufftErrorHandle(cufftDestroy(cufft_plan));

        // Fourier transform for R1 and R3
        myReal* f_dev;
        cudaErrorHandle(cudaMalloc(&f_dev, size_f.nTot*sizeof(myReal)));

        n[0] = size_f.const_2BR; n[1] = size_f.const_2BR;
        inembed[0] = size_f.nR; inembed[1] = size_f.const_2BRs;
        onembed[0] = size_f.nR; onembed[1] = size_f.const_2BRs;
        istride = 1;
        ostride = 1;
        idist = size_f.nR;
        odist = size_f.nR;

        cufftErrorHandle(cufftPlanMany(&cufft_plan, 2, n, inembed, istride, idist, onembed, ostride, odist, myfftBackwardType_R, size_f.nx));
        for (int j = 0; j < size_f.const_2BR; j++) {
            int ind_f = j*size_f.const_2BR;
            cufftErrorHandle(myfftBackwardExec_R(cufft_plan, (myfftComplex*) F1_dev+ind_f, (myfftReal*) f_dev+ind_f));
        }

        cufftErrorHandle(cufftDestroy(cufft_plan));

        // set up output
        size_t size_out[6] = {size_f.const_2BR, size_f.const_2BR, size_f.const_2BR, size_F.const_2Bx, size_F.const_2Bx, size_F.const_2Bx};
        plhs[0] = mxCreateUninitNumericArray(ndims, (size_t*) size_out, mymxRealClass, mxREAL);
        myReal* f = mymxGetReal(plhs[0]);

        cudaErrorHandle(cudaMemcpy(f, f_dev, size_f.nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

        // free memory
        cudaErrorHandle(cudaFree(size_f_dev));
        cudaErrorHandle(cudaFree(size_F_dev));
        cudaErrorHandle(cudaFree(F_dev));
        cudaErrorHandle(cudaFree(d_dev));
        cudaErrorHandle(cudaFree(dl_dev));
        cudaErrorHandle(cudaFree(F1_dev));
        cudaErrorHandle(cudaFree(F1_temp_dev));
        cudaErrorHandle(cudaFree(f_dev));

        if (worksize_max > 0) {
            cudaErrorHandle(cudaFree(work));
        }

        delete[] worksize;
        delete[] cutensor_plan;
    }
}
