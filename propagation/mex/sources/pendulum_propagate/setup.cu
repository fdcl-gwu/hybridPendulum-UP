#include "setup.cuh"

#include <stdio.h>
#include <iostream>

__host__ void modify_F(const cuDoubleComplex* F, cuDoubleComplex* F_modify, bool reduce,Size_F* size_F)
{
	if (reduce) {
		int ind_F_reduced = 0;
		for (int k = 0; k < size_F[0].const_2Bx; k++) {
			for (int j = 0; j < size_F[0].const_2Bx; j++) {
				for (int i = 0; i < size_F[0].const_2Bx; i++) {
					for (int l = 0; l <= size_F[0].lmax; l++) {
						for (int m = -l; m <= l; m++) {
							for (int n = -l; n <= l; n++) {
								int ind_F = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + 
									l*size_F[0].l_cum1 + i*size_F[0].l_cum2 + j*size_F[0].l_cum3 + k*size_F[0].l_cum4;
								F_modify[ind_F_reduced] = F[ind_F];

								ind_F_reduced++;
							}
						}
					}
				}
			}
		}
	} else {
		int ind_F_reduced = 0;
		for (int k = 0; k < size_F[0].const_2Bx; k++) {
			for (int j = 0; j < size_F[0].const_2Bx; j++) {
				for (int i = 0; i < size_F[0].const_2Bx; i++) {
					for (int l = 0; l <= size_F[0].lmax; l++) {
						for (int m = -l; m <= l; m++) {
							for (int n = -l; n <= l; n++) {
								int ind_F = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + 
									l*size_F[0].l_cum1 + i*size_F[0].l_cum2 + j*size_F[0].l_cum3 + k*size_F[0].l_cum4;
								F_modify[ind_F] = F[ind_F_reduced];

								ind_F_reduced++;
							}
						}
					}
				}
			}
		}
	}
}

__host__ void modify_u(const cuDoubleComplex* u, cuDoubleComplex* u_modify, Size_F* size_F)
{
	int ind_u_reduced = 0;
	for (int i = 0; i < 3; i++) {
		for (int l = 0; l <= size_F[0].lmax; l++) {
			for (int m = -l; m <= l; m++) {
				for (int n = -l; n <= l; n++) {
					int ind_u = n+size_F[0].lmax + (m+size_F[0].lmax)*size_F[0].l_cum0 + l*size_F[0].l_cum1 + i*size_F[0].l_cum2;
					u_modify[ind_u_reduced] = u[ind_u];

					ind_u_reduced++;
				}
			}
		}
	}
}


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
	const cuDoubleComplex* Fold_dev, const cuDoubleComplex* X_ijk_dev, const cuDoubleComplex* dF_temp_dev, const Size_F* size_F)
{
	int mode_Fold[4] = {'r','i','j','k'};
	int mode_X[4] = {'i','j','k','p'};
	int mode_dF[2] = {'r','p'};

	int64_t extent_Fold[4] = {size_F->nR_compact, size_F->const_2Bx, size_F->const_2Bx, size_F->const_2Bx};
	int64_t extent_X[4] = {size_F->const_2Bx, size_F->const_2Bx, size_F->const_2Bx, 3};
	int64_t extent_dF[2] = {size_F->nR_compact, 3};

	cutensorTensorDescriptor_t desc_Fold;
	cutensorTensorDescriptor_t desc_X;
	cutensorTensorDescriptor_t desc_temp;
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_Fold,
		4, extent_Fold, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_X,
		4, extent_X, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_temp,
		2, extent_dF, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));

	uint32_t alignmentRequirement_Fold;
	uint32_t alignmentRequirement_X;
	uint32_t alignmentRequirement_temp;
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		Fold_dev, &desc_Fold, &alignmentRequirement_Fold));
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		X_ijk_dev, &desc_X, &alignmentRequirement_X));
	cutensorErrorHandle(cutensorGetAlignmentRequirement(handle,
		dF_temp_dev, &desc_temp, &alignmentRequirement_temp));

	cutensorContractionDescriptor_t desc;
	cutensorErrorHandle(cutensorInitContractionDescriptor(handle, &desc,
		&desc_Fold, mode_Fold, alignmentRequirement_Fold,
		&desc_X, mode_X, alignmentRequirement_X,
		&desc_temp, mode_dF, alignmentRequirement_temp,
		&desc_temp, mode_dF, alignmentRequirement_temp,
		CUTENSOR_COMPUTE_64F));

	cutensorContractionFind_t find;
	cutensorErrorHandle(cutensorInitContractionFind(handle, &find, CUTENSOR_ALGO_DEFAULT));

	cutensorErrorHandle(cutensorContractionGetWorkspace(handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, worksize));

	cutensorErrorHandle(cutensorInitContractionPlan(handle, plan, &desc, &find, *worksize));
}

__host__ void cutensor_initFMR(cutensorHandle_t* handle, cutensorContractionPlan_t* plan, size_t* worksize,
	const cuDoubleComplex* Fold_dev, const cuDoubleComplex* MR_dev, const cuDoubleComplex* FMR_dev, const int l, const Size_F* size_F)
{
	int mode_Fold[4] = {'r','i','j','k'};
	int mode_MR[2] = {'r','p'};
	int mode_FMR[4] = {'i','j','k','p'};

	int m = (2*l+1)*(2*l+1);

	int64_t extent_Fold[4] = {m, size_F->const_2Bx, size_F->const_2Bx, size_F->const_2Bx};
	int64_t extent_MR[2] = {m, 3};
	int64_t extent_FMR[4] = {size_F->const_2Bx, size_F->const_2Bx, size_F->const_2Bx, 3};

	cutensorTensorDescriptor_t desc_Fold;
	cutensorTensorDescriptor_t desc_MR;
	cutensorTensorDescriptor_t desc_FMR;
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_Fold,
		4, extent_Fold, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_MR,
		2, extent_MR, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));
	cutensorErrorHandle(cutensorInitTensorDescriptor(handle, &desc_FMR,
		4, extent_FMR, NULL, CUDA_C_64F, CUTENSOR_OP_IDENTITY));

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
		CUTENSOR_COMPUTE_64F));

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
}
