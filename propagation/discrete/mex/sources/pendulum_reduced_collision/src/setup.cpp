#include "setup.hpp"

#include <stdio.h>

void cudaErrorHandle(const cudaError_t& err)
{
    if (err != cudaSuccess) {
        printf("cuda Error: ");
        printf(cudaGetErrorString(err));
        printf("\n");
    }
}

void cublasErrorHandle(const cublasStatus_t& err)
{
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf("cuBlas Error %i\n", err);
    }
}

void init_Size_f(Size_f* size_f, const int BR, const int Bx)
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

