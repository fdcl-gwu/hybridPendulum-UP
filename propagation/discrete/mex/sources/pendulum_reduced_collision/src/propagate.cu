#include "propagate.cuh"

#include <math.h>
#include <stdio.h>

void get_df_noise(myReal* df, const myReal* f, const myReal* lambda, const myReal* fcL, const int numR, const int* indR, const short* fcL_indx, const int nD, const Size_f* size_f)
{
    // set up fold
    myReal* f_dev;
    cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(f_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    int* indR_dev;
    cudaErrorHandle(cudaMalloc(&indR_dev, numR*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(indR_dev, indR, numR*sizeof(int), cudaMemcpyHostToDevice));

    short* fcL_indx_dev;
    cudaErrorHandle(cudaMalloc(&fcL_indx_dev, nD*numR*size_f->nx*sizeof(short)));
    cudaErrorHandle(cudaMemcpy(fcL_indx_dev, fcL_indx, nD*numR*size_f->nx*sizeof(short), cudaMemcpyHostToDevice));

    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    myReal* fold_dev;
    cudaErrorHandle(cudaMalloc(&fold_dev, nD*numR*size_f->nx*sizeof(myReal)));

    dim3 blocksize_fcL(nD, size_f->const_2Bx, 1);
    dim3 gridsize_fcL(size_f->const_2Bx, numR, 1);

    get_fold_noise <<<gridsize_fcL, blocksize_fcL>>> (fold_dev, f_dev, indR_dev, fcL_indx_dev, size_f_dev);

    // calculate fin
    myReal* fcL_dev;
    cudaErrorHandle(cudaMalloc(&fcL_dev, nD*numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(fcL_dev, fcL, nD*numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* fin_dev;
    cudaErrorHandle(cudaMalloc(&fin_dev, numR*size_f->nx*sizeof(myReal)));

    cutensorHandle_t handle;
    cutensorInit(&handle);

    int32_t mode_fold[3] = {'x','R','y'};
    int32_t mode_fcL[3] = {'x','R','y'};
    int32_t mode_fin[2] = {'R','y'};

    int64_t extent_fold[3] = {nD, numR, size_f->nx};
    int64_t extent_fcL[3] = {nD, numR, size_f->nx};
    int64_t extent_fin[2] = {numR, size_f->nx};

    cutensorTensorDescriptor_t desc_fold;
    cutensorTensorDescriptor_t desc_fcL;
    cutensorTensorDescriptor_t desc_fin;
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle, &desc_fold, 3, extent_fold, NULL, mycudaRealType, CUTENSOR_OP_IDENTITY));
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle, &desc_fcL, 3, extent_fcL, NULL, mycudaRealType, CUTENSOR_OP_IDENTITY));
    cutensorErrorHandle(cutensorInitTensorDescriptor(&handle, &desc_fin, 2, extent_fin, NULL, mycudaRealType, CUTENSOR_OP_IDENTITY));

    uint32_t alignment_fold;
    uint32_t alignment_fcL;
    uint32_t alignment_fin;
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle, fold_dev, &desc_fold, &alignment_fold));
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle, fcL_dev, &desc_fcL, &alignment_fcL));
    cutensorErrorHandle(cutensorGetAlignmentRequirement(&handle, fin_dev, &desc_fin, &alignment_fin));

    cutensorContractionDescriptor_t desc;
    cutensorErrorHandle(cutensorInitContractionDescriptor(&handle, &desc, &desc_fold, mode_fold, alignment_fold,
        &desc_fcL, mode_fcL, alignment_fcL,
        &desc_fin, mode_fin, alignment_fin,
        &desc_fin, mode_fin, alignment_fin, mycutensor_computetype));

    cutensorContractionFind_t find;
    cutensorErrorHandle(cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT));

    size_t worksize;
    cutensorErrorHandle(cutensorContractionGetWorkspace(&handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));
    void* workspace = nullptr;
    if (worksize > 0) {
        cudaErrorHandle(cudaMalloc(&workspace, worksize));
    }

    cutensorContractionPlan_t plan;
    cutensorErrorHandle(cutensorInitContractionPlan(&handle, &plan, &desc, &find, worksize));

    myReal alpha = 1.0;
    myReal beta = 0.0;

    cutensorErrorHandle(cutensorContraction(&handle, &plan, &alpha, fold_dev, fcL_dev, &beta, fin_dev, fin_dev, workspace, worksize, 0));

    // free memory
    cudaErrorHandle(cudaFree(fcL_indx_dev));
    cudaErrorHandle(cudaFree(fold_dev));
    cudaErrorHandle(cudaFree(fcL_dev));

    if (worksize > 0) {
        cudaErrorHandle(cudaFree(workspace));
    }

    // compute fout
    myReal* df_dev;
    cudaErrorHandle(cudaMalloc(&df_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(df_dev, 0, size_f->nTot*sizeof(myReal)));

    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(lambda_dev, lambda, numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));

    dim3 blocksize_n0Rx(size_f->const_2Bx, 1, 1);
    dim3 gridsize_n0Rx(size_f->const_2Bx, numR, 1);
    
    get_fout_noise <<<gridsize_n0Rx, blocksize_n0Rx>>> (df_dev, fin_dev, f_dev, lambda_dev, indR_dev, size_f_dev);
    cudaErrorHandle(cudaMemcpy(df, df_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(f_dev));
    cudaErrorHandle(cudaFree(indR_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(fin_dev));
    cudaErrorHandle(cudaFree(df_dev));
    cudaErrorHandle(cudaFree(lambda_dev));
}

void get_df_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f)
{
    // compute fin
    dim3 blocksize_n0Rx(size_f->const_2Bx, 1, 1);
    dim3 gridsize_n0Rx(size_f->const_2Bx, numR, 1);

    myReal* f_dev;
    cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(f_dev, f, size_f->nTot*sizeof(myReal), cudaMemcpyHostToDevice));

    myReal* lambda_dev;
    cudaErrorHandle(cudaMalloc(&lambda_dev, numR*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(lambda_dev, lambda, numR*sizeof(myReal), cudaMemcpyHostToDevice));

    int* indR_dev;
    cudaErrorHandle(cudaMalloc(&indR_dev, numR*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(indR_dev, indR, numR*sizeof(int), cudaMemcpyHostToDevice));

    int* ind_interp_dev;
    cudaErrorHandle(cudaMalloc(&ind_interp_dev, 4*numR*size_f->nx*sizeof(int)));
    cudaErrorHandle(cudaMemcpy(ind_interp_dev, ind_interp, 4*numR*size_f->nx*sizeof(int), cudaMemcpyHostToDevice));
    
    myReal* coeff_interp_dev;
    cudaErrorHandle(cudaMalloc(&coeff_interp_dev, 4*numR*size_f->nx*sizeof(myReal)));
    cudaErrorHandle(cudaMemcpy(coeff_interp_dev, coeff_interp, 4*numR*size_f->nx*sizeof(myReal), cudaMemcpyHostToDevice));
    
    Size_f* size_f_dev;
    cudaErrorHandle(cudaMalloc(&size_f_dev, sizeof(Size_f)));
    cudaErrorHandle(cudaMemcpy(size_f_dev, size_f, sizeof(Size_f), cudaMemcpyHostToDevice));

    myReal* df_dev;
    cudaErrorHandle(cudaMalloc(&df_dev, size_f->nTot*sizeof(myReal)));
    cudaErrorHandle(cudaMemset(df_dev, 0, size_f->nTot*sizeof(myReal)));

    get_fin_nonoise <<<gridsize_n0Rx, blocksize_n0Rx>>> (df_dev, f_dev, lambda_dev, indR_dev, ind_interp_dev, coeff_interp_dev, size_f_dev);    

    // compute fout
    int* lambda_indx_dev;
    cudaErrorHandle(cudaMalloc(&lambda_indx_dev, size_f->nx*sizeof(int)));

    for (int iR = 0; iR < numR; iR++) {
        cudaErrorHandle(cudaMemcpy(lambda_indx_dev, lambda_indx[iR], lambda_numx[iR]*sizeof(int), cudaMemcpyHostToDevice));
        get_fout_nonoise <<<(int)lambda_numx[iR]/128+1, 128>>> (df_dev+indR[iR], f_dev+indR[iR], lambda[iR], lambda_indx_dev, lambda_numx[iR], size_f_dev);
    }

    cudaErrorHandle(cudaMemcpy(df, df_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

    // free memory
    cudaErrorHandle(cudaFree(f_dev));
    cudaErrorHandle(cudaFree(lambda_dev));
    cudaErrorHandle(cudaFree(indR_dev));
    cudaErrorHandle(cudaFree(ind_interp_dev));
    cudaErrorHandle(cudaFree(coeff_interp_dev));
    cudaErrorHandle(cudaFree(size_f_dev));
    cudaErrorHandle(cudaFree(df_dev));
    cudaErrorHandle(cudaFree(lambda_indx_dev));
}

__global__ void get_fold_noise(myReal* f_old, const myReal* f, const int* indR, const short* fcL_indx, const Size_f* size_f)
{
    int ind_fcL = threadIdx.x + blockIdx.y*blockDim.x + (threadIdx.y+blockIdx.x*blockDim.y)*blockDim.x*gridDim.y;
    int ind_f = indR[blockIdx.y] + fcL_indx[ind_fcL]*size_f->nR;

    f_old[ind_fcL] = f[ind_f];
}

__global__ void get_fin_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int* indR, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    int indfR = indR[blockIdx.y];
    int indf = indfR + indx*size_f->nR;
    int indInterp = 4*(blockIdx.y + indx*gridDim.y);

    if (isnan(coeff_interp[indInterp])) {
        df[indf] = 0;
    } else {
        int indf_interp[4];
        for (int i = 0; i < 4; i++) {
            indf_interp[i] = indfR + ind_interp[indInterp+i]*size_f->nR;
        }

        myReal f_interp = 0.0;
        for (int i = 0; i < 4; i++) {
            f_interp += f[indf_interp[i]]*coeff_interp[indInterp+i];
        }

        df[indf] = f_interp*lambda[blockIdx.y];
    }
}

__global__ void get_fout_noise(myReal* df, const myReal* fin, const myReal* f, const myReal* lambda, const int* indR, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    int indf = indR[blockIdx.y] + indx*size_f->nR;
    int indfin = blockIdx.y + indx*gridDim.y;

    df[indf] = fin[indfin] - f[indf]*lambda[indfin];
}

__global__ void get_fout_nonoise(myReal* df, const myReal* f, const myReal lambda, const int* lambda_indx, const int lambda_numx, const Size_f* size_f)
{
    int indx = threadIdx.x + blockIdx.x*blockDim.x;
    if (indx < lambda_numx) {
        int indf = lambda_indx[indx]*size_f->nR;
        df[indf] = df[indf] - f[indf]*lambda;
    }
}


