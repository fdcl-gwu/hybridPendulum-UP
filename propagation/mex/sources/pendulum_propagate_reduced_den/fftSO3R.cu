#include "fftSO3R.cuh"

__global__ void supplement_R(myComplex* F, const Size_f* size_f)
{
	unsigned int i = threadIdx.x + 1 + size_f->BR;
	unsigned int j = blockIdx.x;

	unsigned int indf_t = i + blockIdx.y*size_f->const_2BR + j*size_f->const_2BRs + blockIdx.z*size_f->nR;
	unsigned int indf_s;
	if (j == 0) {
		indf_s = size_f->const_2BR-i + blockIdx.y*size_f->const_2BR + j*size_f->const_2BRs + blockIdx.z*size_f->nR;
	} else {
		indf_s = size_f->const_2BR-i + blockIdx.y*size_f->const_2BR + (size_f->const_2BR-j)*size_f->const_2BRs + blockIdx.z*size_f->nR;
	}

	F[indf_t].x = F[indf_s].x;
	F[indf_t].y = -F[indf_s].y;
}

__global__ void shiftflip_fft(myComplex* F1, const int dim, const Size_f* size_f)
{
	extern __shared__ myComplex F1_temp[];

	unsigned int ind;
	if (dim == 1) {
		ind = threadIdx.x + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->nR;
	} else if (dim == 3) {
		ind = blockIdx.x + threadIdx.x*size_f->const_2BRs + blockIdx.y*size_f->nR;
	}

	F1_temp[threadIdx.x] = F1[ind];
	__syncthreads();

	unsigned int ind_sf;
	unsigned int ind_sf_dim;
	if (threadIdx.x < size_f->BR) {
		ind_sf_dim = size_f->BR - threadIdx.x - 1;
	} else {
		ind_sf_dim = 3*size_f->BR - threadIdx.x - 1;
	}

	if (dim == 1) {
		ind_sf = ind_sf_dim + blockIdx.x*size_f->const_2BR + blockIdx.y*size_f->nR;
	} else if (dim == 3) {
		ind_sf = blockIdx.x + ind_sf_dim*size_f->const_2BRs + blockIdx.y*size_f->nR;
	}
	F1[ind_sf] = F1_temp[threadIdx.x];
}

__global__ void mul_dw(myReal* dw, const myReal* d, const myReal* w, const Size_F* size_F)
{
	unsigned int ind_d = threadIdx.x + blockIdx.x*size_F->const_2lp1 + blockIdx.y*size_F->const_2lp1s + blockIdx.z*size_F->nR;
	unsigned int ind_w = blockIdx.z;

	dw[ind_d] = d[ind_d] * w[ind_w];
}

__global__ void mul_dl(myReal* dl, const myReal* d, const Size_F* size_F)
{
	unsigned int ind_d = threadIdx.x + blockIdx.x*size_F->const_2lp1 + blockIdx.y*size_F->const_2lp1s + blockIdx.z*size_F->nR;
	dl[ind_d] = d[ind_d] * (2*blockIdx.y+1);
}

__host__ void fftSO3R_forward(myComplex* F_dev, const myReal* f_dev, const myReal* dw_dev, const Size_F* size_F, const Size_F* size_F_dev, const Size_f* size_f, const Size_f* size_f_dev)
{
	// set up arryas
	myComplex* F1_dev;
	cudaErrorHandle(cudaMalloc(&F1_dev, size_f->nTot*sizeof(myComplex)));

	myComplex* F_temp_dev;
	cudaErrorHandle(cudaMalloc(&F_temp_dev, size_f->nx*sizeof(myComplex)));

	// Fourier transform for R1 and R3
	cufftHandle cufft_plan;
	int n[3] = {size_f->const_2BR, size_f->const_2BR};
	int inembed[3] = {size_f->nR, size_f->const_2BRs};
	int onembed[3] = {size_f->nR, size_f->const_2BRs};
	int istride = 1;
	int ostride = 1;
	int idist = size_f->nR;
	int odist = size_f->nR;

	cufftErrorHandle(cufftPlanMany(&cufft_plan, 2, n, inembed, istride, idist, onembed, ostride, odist, myfftForwardType_R, size_f->nx));
	for (int j = 0; j < size_f->const_2BR; j++) {
		int ind_f = j*size_f->const_2BR;
		cufftErrorHandle(myfftForwardExec_R(cufft_plan, (myfftReal*) f_dev+ind_f, (myfftComplex*) F1_dev+ind_f));
	}

	dim3 blocksize_supplement_R(size_f->BR-1, 1, 1);
	dim3 gridsize_supplement_R(size_f->const_2BR, size_f->const_2BR, size_f->nx);
	supplement_R <<<gridsize_supplement_R, blocksize_supplement_R>>> (F1_dev, size_f_dev);

	// Fourier transform for x
	n[0] = size_f->const_2Bx; n[1] = size_f->const_2Bx; n[2] = size_f->const_2Bx;
	inembed[0] = size_f->const_2Bx; inembed[1] = size_f->const_2Bx; inembed[2] = size_f->const_2Bx;
	onembed[0] = size_f->const_2Bx; onembed[1] = size_f->const_2Bx; onembed[2] = size_f->const_2Bx;
	istride = size_f->nR;
	ostride = size_f->nR;
	idist = 1;
	odist = 1;

	cufftErrorHandle(cufftPlanMany(&cufft_plan, 2, n, inembed, istride, idist, onembed, ostride, odist, myfftForwardType_x, 1));
	for (int i = 0; i < size_f->const_2BR; i++) {
		for (int j = 0; j < size_f->const_2BR; j++) {
			for (int k = 0; k < size_f->const_2BR; k++) {
				int ind_f = i + k*size_f->const_2BR + j*size_f->const_2BRs;
				cufftErrorHandle(myfftForwardExec_x(cufft_plan, (myfftComplex*) F1_dev+ind_f, (myfftComplex*) F1_dev+ind_f, CUFFT_FORWARD));
			}
		}
	}

	// fftshift and flip
	dim3 blocksize_flip(size_f->const_2BR, 1, 1);
	dim3 gridsize_flip(size_f->const_2BRs, size_f->nx, 1);

	shiftflip_fft <<<gridsize_flip, blocksize_flip, size_f->const_2BR*sizeof(myComplex)>>> (F1_dev, 1, size_f_dev);
	shiftflip_fft <<<gridsize_flip, blocksize_flip, size_f->const_2BR*sizeof(myComplex)>>> (F1_dev, 3, size_f_dev);

	// Fourier transform for R2
	int mode_F1[2] = {'k','x'};
	int mode_dw[1] = {'k'};
	int mode_F[1] = {'x'};

	int64_t extent_F1[2] = {size_f->const_2BR, size_f->nx};
	int64_t extent_dw[1] = {size_f->const_2BR};
	int64_t extent_F[1] = {size_f->nx};

	int64_t stride_F1[2] = {size_f->const_2BR, size_f->nR};
	int64_t stride_dw[1] = {size_F->nR};
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
	cutensorErrorHandle(cutensorGetAlignmentRequirement(&cutensor_handle, F_temp_dev, &desc_F, &alignment_F));

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

	for (int l = 0; l <= size_F->lmax; l++) {
		for (int m = -l; m <= l; m++) {
			for (int n = -l; n <= l; n++) {
				int ind_F1 = m+size_F->lmax + (n+size_F->lmax)*size_f->const_2BRs;
				int ind_dw = m+size_F->lmax + (n+size_F->lmax)*size_F->const_2lp1 + l*size_F->const_2lp1s;
				int ind_F = m+l + (n+l)*(2*l+1) + l*(2*l-1)*(2*l+1)/3;

				cutensorErrorHandle(cutensorContraction(&cutensor_handle, &cutensor_plan, &alpha, F1_dev+ind_F1, dw_dev+ind_dw,
					&beta, F_temp_dev, F_temp_dev, work, worksize, 0));

				cudaErrorHandle(cudaMemcpy2D(F_dev+ind_F, size_F->nR_compact*sizeof(myComplex), F_temp_dev, sizeof(myComplex), sizeof(myComplex), size_F->nx, cudaMemcpyDeviceToDevice));
			}
		}
	}

	// free memory
	cudaErrorHandle(cudaFree(F1_dev));
	cudaErrorHandle(cudaFree(F_temp_dev));
	if (worksize > 0) {
		cudaErrorHandle(cudaFree(work));
	}
}

__host__ void fftSO3R_backward(myReal* f, const myComplex* F, const myReal* dl_dev, const Size_F* size_F, const Size_F* size_F_dev, const Size_f* size_f, const Size_f* size_f_dev)
{
	// inverse Fourier transform for R2
	myComplex* F_dev;
	cudaErrorHandle(cudaMalloc(&F_dev, size_F->nTot*sizeof(myComplex)));
	cudaErrorHandle(cudaMemcpy(F_dev, F, size_F->nTot*sizeof(myComplex), cudaMemcpyHostToDevice));

	myComplex* F1_dev;
	cudaErrorHandle(cudaMalloc(&F1_dev, size_f->nTot*sizeof(myComplex)));
	cudaErrorHandle(cudaMemset(F1_dev, 0, size_f->nTot*sizeof(myComplex)));

	myComplex* F1_temp_dev;
	cudaErrorHandle(cudaMalloc(&F1_temp_dev, size_f->nx*sizeof(myComplex)));
	cudaErrorHandle(cudaMemset(F1_temp_dev, 0, size_f->nx*sizeof(myComplex)));

	cutensorHandle_t cutensor_handle;
	cutensorInit(&cutensor_handle);

	int mode_F[2] = {'l','x'};
	int mode_dl[1] = {'l'};
	int mode_F1[1] = {'x'};

	size_t* worksize = new size_t[size_F->BR];
	cutensorContractionPlan_t* cutensor_plan = new cutensorContractionPlan_t[size_F->BR];

	for (int l = 0; l <= size_F->lmax; l++) {
		int64_t extent_F[2] = {size_F->BR-l, size_f->nx};
		int64_t extent_dl[1] = {size_F->BR-l};
		int64_t extent_F1[1] = {size_f->nx};

		int64_t stride_F[2] = {size_F->const_2lp1s, size_F->nR};
		int64_t stride_dl[1] = {size_F->const_2lp1s};
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
	for (int l = 0; l <= size_F->lmax; l++) {
		worksize_max = (worksize_max < worksize[l]) ? worksize[l] : worksize_max;
	}

	void* work = nullptr;
	if (worksize_max > 0) {
		cudaErrorHandle(cudaMalloc(&work, worksize_max));
	}

	myComplex alpha = make_myComplex((myReal) 1.0/size_f->nx, 0.0);
	myComplex beta = make_myComplex(0.0, 0.0);

	for (int m = -size_F->lmax; m <= size_F->lmax; m++) {
		for (int n = -size_F->lmax; n <= size_F->lmax; n++) {
			int mp = (m>=0) ? m : -m;
			int np = (n>=0) ? n : -n;
			int lmin = (mp >= np) ? mp : np;

			for (int k = 0; k < size_f->const_2BR; k++) {
				int ind_F = m+size_F->lmax + (n+size_F->lmax)*size_F->const_2lp1 + lmin*size_F->const_2lp1s;
				int ind_dl = ind_F + k*size_F->nR;
				int ind_F1 = m+size_F->lmax + k*size_f->const_2BR + (n+size_F->lmax)*size_f->const_2BRs;

				cutensorErrorHandle(cutensorContraction(&cutensor_handle, cutensor_plan+lmin, &alpha, F_dev+ind_F, dl_dev+ind_dl, &beta, F1_temp_dev, F1_temp_dev, work, worksize[lmin], 0));
				cudaErrorHandle(cudaMemcpy2D(F1_dev+ind_F1, size_f->nR*sizeof(myComplex), F1_temp_dev, sizeof(myComplex), sizeof(myComplex), size_F->nx, cudaMemcpyDeviceToDevice));
			}
		}
	}

	// fftshift and flip
	dim3 blocksize_flip(size_f->const_2BR, 1, 1);
	dim3 gridsize_flip(size_f->const_2BRs, size_f->nx, 1);

	shiftflip_fft <<<gridsize_flip, blocksize_flip, size_f->const_2BR*sizeof(myComplex)>>> (F1_dev, 1, size_f_dev);
	shiftflip_fft <<<gridsize_flip, blocksize_flip, size_f->const_2BR*sizeof(myComplex)>>> (F1_dev, 3, size_f_dev);

	// inverse Fourier transform for x
	cufftHandle cufft_plan;
	int n[3] = {size_f->const_2Bx, size_f->const_2Bx, size_f->const_2Bx};
	int inembed[3] = {size_f->const_2Bx, size_f->const_2Bx, size_f->const_2Bx};
	int onembed[3] = {size_f->const_2Bx, size_f->const_2Bx, size_f->const_2Bx};
	int istride = size_f->nR;
	int ostride = size_f->nR;
	int idist = 1;
	int odist = 1;

	cufftErrorHandle(cufftPlanMany(&cufft_plan, 2, n, inembed, istride, idist, onembed, ostride, odist, myfftBackwardType_x, 1));
	for (int i = 0; i < size_f->const_2BR; i++) {
		for (int j = 0; j < size_f->const_2BR; j++) {
			for (int k = 0; k < size_f->const_2BR; k++) {
				int ind_f = i + k*size_f->const_2BR + j*size_f->const_2BRs;
				cufftErrorHandle(myfftBackwardExec_x(cufft_plan, (myfftComplex*) F1_dev+ind_f, (myfftComplex*) F1_dev+ind_f, CUFFT_INVERSE));
			}
		}
	}

	// Fourier transform for R1 and R3
	myReal* f_dev;
	cudaErrorHandle(cudaMalloc(&f_dev, size_f->nTot*sizeof(myReal)));

	n[0] = size_f->const_2BR; n[1] = size_f->const_2BR;
	inembed[0] = size_f->nR; inembed[1] = size_f->const_2BRs;
	onembed[0] = size_f->nR; onembed[1] = size_f->const_2BRs;
	istride = 1;
	ostride = 1;
	idist = size_f->nR;
	odist = size_f->nR;

	cufftErrorHandle(cufftPlanMany(&cufft_plan, 2, n, inembed, istride, idist, onembed, ostride, odist, myfftBackwardType_R, size_f->nx));
	for (int j = 0; j < size_f->const_2BR; j++) {
		int ind_f = j*size_f->const_2BR;
		cufftErrorHandle(myfftBackwardExec_R(cufft_plan, (myfftComplex*) F1_dev+ind_f, (myfftReal*) f_dev+ind_f));
	}

	cudaErrorHandle(cudaMemcpy(f, f_dev, size_f->nTot*sizeof(myReal), cudaMemcpyDeviceToHost));

	cudaErrorHandle(cudaFree(F_dev));
	cudaErrorHandle(cudaFree(F1_dev));
	cudaErrorHandle(cudaFree(F1_temp_dev));
	cudaErrorHandle(cudaFree(f_dev));
	if (worksize_max > 0) {
		cudaErrorHandle(cudaFree(work));
	}
}

