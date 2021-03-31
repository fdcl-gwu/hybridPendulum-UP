#include "setup.cuh"

__global__ void supplement_R(myComplex* F, const Size_f* size_f);
__global__ void shiftflip_fft(myComplex* F1, const int dim, const Size_f* size_f);
__global__ void mul_dw(myReal* dw, const myReal* d, const myReal* w, const Size_F* size_F);
__global__ void mul_dl(myReal* dl, const myReal* d, const Size_F* size_F);

__host__ void fftSO3R_forward(myComplex* F_dev, const myReal* f_dev, const myReal* dw_dev, const Size_F* size_F, const Size_F* size_F_dev, const Size_f* size_f, const Size_f* size_f_dev);
__host__ void fftSO3R_backward(myReal* f, const myComplex* F_dev, const myReal* dl_dev, const Size_F* size_F, const Size_F* size_F_dev, const Size_f* size_f, const Size_f* size_f_dev);

