#include "setup.cuh"

#include <stdio.h>
#include <iostream>


__host__ void cudaErrorHandle(const cudaError_t& err)
{
	if (err != cudaSuccess) {
		mexPrintf("cuda Error: ");
		mexPrintf(cudaGetErrorString(err));
		mexPrintf("\n");
	}
}

__host__ void cutensorErrorHandle(const cutensorStatus_t& err)
{
	if (err != CUTENSOR_STATUS_SUCCESS) {
		mexPrintf("cuTensor Error: ");
		mexPrintf(cutensorGetErrorString(err));
		mexPrintf("\n");
	}
}

__host__ void cublasErrorHandle(const cublasStatus_t& err)
{
	if (err != CUFFT_SUCCESS) {
		mexPrintf("cuBlas Error %i\n", err);
	}
}

__host__ void cufftErrorHandle(const cufftResult_t& err)
{
	if (err != CUFFT_SUCCESS) {
		mexPrintf("cufft Error %i\n", err);
	}
}

__host__ void cutensor_initConv(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const void* Fold_dev, const void* X_dev, const void* dF_dev, const Size_F* size_F)
{
	int mode_Fold[3] = {'r','i','j'};
	int mode_X[3] = {'i','j','p'};
	int mode_dF[2] = {'r','p'};

	int64_t extent_Fold[3] = {size_F->nR_compact, size_F->const_2Bx, size_F->const_2Bx};
	int64_t extent_X[3] = {size_F->const_2Bx, size_F->const_2Bx, 2};
	int64_t extent_dF[2] = {size_F->nR_compact, 2};

	cutensorTensorDescriptor_t desc_Fold;
	cutensorTensorDescriptor_t desc_X;
	cutensorTensorDescriptor_t desc_dF;
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_Fold,
		3, extent_Fold, NULL, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_X,
		3, extent_X, NULL, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_dF,
		2, extent_dF, NULL, mycutensor_Complextype, CUTENSOR_OP_IDENTITY));

	uint32_t alignmentRequirement_Fold;
	uint32_t alignmentRequirement_X;
	uint32_t alignmentRequirement_temp;
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		Fold_dev, &desc_Fold, &alignmentRequirement_Fold));
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		X_dev, &desc_X, &alignmentRequirement_X));
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		dF_dev, &desc_dF, &alignmentRequirement_temp));

	cutensorContractionDescriptor_t desc;
	cutensorErrorHandle(cutensorInitContractionDescriptor(handle, &desc,
		&desc_Fold, mode_Fold, alignmentRequirement_Fold,
		&desc_X, mode_X, alignmentRequirement_X,
		&desc_dF, mode_dF, alignmentRequirement_temp,
		&desc_dF, mode_dF, alignmentRequirement_temp,
		mycutensor_computetype));

	cutensorContractionFind_t find;
	cutensorErrorHandle(cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

	cutensorErrorHandle(cutensorContractionGetWorkspace(handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, worksize));

	cutensorErrorHandle(cutensorInitContractionPlan(handle, plan, &desc, &find, *worksize));
}

__host__ void init_Size_F(Size_F* size_F, int BR, int Bx)
{
	size_F->BR = BR;
	size_F->Bx = Bx;
	size_F->lmax = BR-1;

	size_F->nR = (2*size_F->lmax+1) * (2*size_F->lmax+1) * (size_F->lmax+1);
	size_F->nx = (2*Bx) * (2*Bx);
	size_F->nTot = size_F->nR * size_F->nx;

	size_F->nR_compact = (size_F->lmax+1) * (2*size_F->lmax+1) * (2*size_F->lmax+3) / 3;
	size_F->nTot_compact = size_F->nR_compact * size_F->nx;

	size_F->const_2Bx = 2*Bx;
	size_F->const_2lp1 = 2*size_F->lmax+1;
	size_F->const_lp1 = size_F->lmax+1;
	size_F->const_2lp1s = (2*size_F->lmax+1) * (2*size_F->lmax+1);

	size_F->l_cum0 = size_F->const_2lp1;
	size_F->l_cum1 = size_F->l_cum0*size_F->const_2lp1;
	size_F->l_cum2 = size_F->l_cum1*size_F->const_lp1;
	size_F->l_cum3 = size_F->l_cum2*size_F->const_2Bx;
}

__host__ void init_Size_f(Size_f* size_f, const int BR, const int Bx)
{
	size_f->BR = BR;
	size_f->Bx = Bx;

	size_f->nR = (2*BR) * (2*BR) * (2*BR);
	size_f->nx = (2*Bx) * (2*Bx);
	size_f->nTot = size_f->nR * size_f->nx;

	size_f->const_2BR = 2*BR;
	size_f->const_2Bx = 2*Bx;
	size_f->const_2BRs = (2*BR) * (2*BR);
	size_f->const_2Bxs = (2*Bx) * (2*Bx);
}

