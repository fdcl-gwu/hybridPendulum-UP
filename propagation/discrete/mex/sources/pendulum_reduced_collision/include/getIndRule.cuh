#include "setup.hpp"

void getIndRule(int* ind_interp, myReal* coeff_interp, const myReal* x, const myReal* Omega, const int numR, const Size_f* size_f);

__global__ void compute_indRule(int* ind_interp, myReal* coeff_interp, const myReal* x, const myReal* Omega, const myReal L, const myReal dx2, const Size_f* size_f);

