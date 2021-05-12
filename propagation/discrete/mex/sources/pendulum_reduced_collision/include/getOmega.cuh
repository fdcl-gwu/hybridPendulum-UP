#include "setup.hpp"

void getOmega(myReal* Omega, const myReal* R, const myReal* x, const int* ind_n0, const int nn0, const myReal* epsilon, const Size_f* size_f);

__global__ void getT(myReal* t, const myReal* R, const Size_f* size_f);

__global__ void compute_Omega(myReal* Omega, const myReal* R, const myReal* x, const myReal* t, const int* ind_n0, const myReal epsilon, const Size_f* size_f); 

