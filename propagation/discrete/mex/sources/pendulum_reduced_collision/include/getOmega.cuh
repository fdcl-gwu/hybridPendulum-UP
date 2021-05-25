#include "setup.hpp"

void getOmega(myReal* Omega, const myReal* R, const myReal* x, const int* indR, const int numR, const myReal* epsilon, const myReal* PC, const char* direction, const Size_f* size_f);

__global__ void getT(myReal* t, const myReal* R, const Size_f* size_f);

__global__ void compute_Omega_new(myReal* Omega, const myReal* R, const myReal* x, const myReal* t, const int* indR, const myReal epsilon, const Size_f* size_f);

__global__ void compute_Omega_old(myReal* Omega, const myReal* R, const myReal* x, const myReal* t, const int* indR, const myReal epsilon, const myReal* PC, const myReal x_lower, const myReal x_upper, const Size_f* size_f);

