#include "setup.cuh"

#include <stdio.h>
#include <iostream>


__host__ void cudaErrorHandle(const cudaError_t& err)
{
	if (err != cudaSuccess) {
		std::cout << "Cuda Error: " << cudaGetErrorString(err) << std::endl;
	}
}

__host__ void cutensorErrorHandle(const cutensorStatus_t& err)
{
	if (err != CUTENSOR_STATUS_SUCCESS) {
		std::cout << "cuTensor Error: " << cutensorGetErrorString(err) << std::endl;
	}
}

__host__ void cublasErrorHandle(const cublasStatus_t& err)
{
	if (err != CUBLAS_STATUS_SUCCESS) {
		std::cout << "cuBlas Error: " << err << std::endl;
	}
}

__host__ void cutensor_initConv(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const void* Fold_dev, const void* X_dev, const void* dF_dev, const int nR_split, const bool issmall, const Size_F* size_F)
{
	int mode_Fold[4];
	int mode_X[4] = {'i','j','k','p'};
	int mode_dF[2] = {'r','p'};

	int64_t extent_Fold[4];
	int64_t extent_X[4] = {size_F->const_2Bx, size_F->const_2Bx, size_F->const_2Bx, 3};
	int64_t extent_dF[2] = {nR_split, 3};

	if (issmall) {
		mode_Fold[0] = 'r';
		mode_Fold[1] = 'i';
		mode_Fold[2] = 'j';
		mode_Fold[3] = 'k';

		extent_Fold[0] = nR_split;
		extent_Fold[1] = size_F->const_2Bx;
		extent_Fold[2] = size_F->const_2Bx;
		extent_Fold[3] = size_F->const_2Bx;
	} else {
		mode_Fold[0] = 'i';
		mode_Fold[1] = 'j';
		mode_Fold[2] = 'k';
		mode_Fold[3] = 'r';

		extent_Fold[0] = size_F->const_2Bx;
		extent_Fold[1] = size_F->const_2Bx;
		extent_Fold[2] = size_F->const_2Bx;
		extent_Fold[3] = nR_split;
	}

	cutensorTensorDescriptor_t desc_Fold;
	cutensorTensorDescriptor_t desc_X;
	cutensorTensorDescriptor_t desc_dF;
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_Fold,
		4, extent_Fold, NULL, mycutensor_datatype, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_X,
		4, extent_X, NULL, mycutensor_datatype, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_dF,
		2, extent_dF, NULL, mycutensor_datatype, CUTENSOR_OP_IDENTITY));

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

__host__ void cutensor_initFMR(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const void* Fold_dev, const void* MR_dev, const void* FMR_dev, const int l, const bool issmall, const Size_F* size_F)
{
	int mode_Fold[2] = {'r','x'};
	int mode_MR[2] = {'r','p'};
	int mode_FMR[2] = {'x','p'};

	int m = (2*l+1)*(2*l+1);

	int64_t extent_Fold[2];
	int64_t extent_MR[2] = {m, 3};
	int64_t extent_FMR[2];

	if (issmall) {
		extent_Fold[0] = m;
		extent_Fold[1] = size_F->nx;

		extent_FMR[0] = size_F->nx;
		extent_FMR[1] = 3;
	} else {
		extent_Fold[0] = m;
		extent_Fold[1] = size_F->const_2Bxs;

		extent_FMR[0] = size_F->const_2Bxs;
		extent_FMR[1] = 3;
	}

	cutensorTensorDescriptor_t desc_Fold;
	cutensorTensorDescriptor_t desc_MR;
	cutensorTensorDescriptor_t desc_FMR;
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_Fold,
		2, extent_Fold, NULL, mycutensor_datatype, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_MR,
		2, extent_MR, NULL, mycutensor_datatype, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_FMR,
		2, extent_FMR, NULL, mycutensor_datatype, CUTENSOR_OP_IDENTITY));

	uint32_t alignmentRequirement_Fold;
	uint32_t alignmentRequirement_MR;
	uint32_t alignmentRequirement_FMR;
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		Fold_dev, &desc_Fold, &alignmentRequirement_Fold));
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		MR_dev, &desc_MR, &alignmentRequirement_MR));
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		FMR_dev, &desc_FMR, &alignmentRequirement_FMR));

	cutensorContractionDescriptor_t desc;
	cutensorErrorHandle(cutensorInitContractionDescriptor(handle, &desc,
		&desc_Fold, mode_Fold, alignmentRequirement_Fold,
		&desc_MR, mode_MR, alignmentRequirement_MR,
		&desc_FMR, mode_FMR, alignmentRequirement_FMR,
		&desc_FMR, mode_FMR, alignmentRequirement_FMR,
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
	size_F->nx = (2*Bx) * (2*Bx) * (2*Bx);
	size_F->nTot = size_F->nR * size_F->nx;

	size_F->nR_compact = (size_F->lmax+1) * (2*size_F->lmax+1) * (2*size_F->lmax+3) / 3;
	size_F->nTot_compact = size_F->nR_compact * size_F->nx;

	size_F->const_2Bx = 2*Bx;
	size_F->const_2Bxs = (2*Bx) * (2*Bx);
	size_F->const_2lp1 = 2*size_F->lmax+1;
	size_F->const_lp1 = size_F->lmax+1;
	size_F->const_2lp1s = (2*size_F->lmax+1) * (2*size_F->lmax+1);

	size_F->l_cum0 = size_F->const_2lp1;
	size_F->l_cum1 = size_F->l_cum0*size_F->const_2lp1;
	size_F->l_cum2 = size_F->l_cum1*size_F->const_lp1;
	size_F->l_cum3 = size_F->l_cum2*size_F->const_2Bx;
	size_F->l_cum4 = size_F->l_cum3*size_F->const_2Bx;

	size_F->ns = size_F->const_2lp1;
	size_F->nR_split = (int) size_F->nR_compact / (size_F->ns-1);
	size_F->nR_remainder = size_F->nR_compact % (size_F->ns-1);

	size_F->nTot_splitx = size_F->nR_compact * size_F->const_2Bxs;
}

