#include "setup.hpp"

void getFcL(myReal*** fcL, int** fcL_indx1, int* fcL_numx1, int*** fcL_indx2, int** fcL_numx2, const myReal* x, const myReal* Omega, const myReal* lambda, const int nn0, int* const* lambda_indx, const int* lambda_numbx, const myReal* Gd, const Size_f* size_f);

__global__ void get_fcL_x(myReal* fcL_x, const myReal* x, const myReal* Omega, const myReal lambda, const int* lambda_indx, const myReal* invGd, const int nn0, const int lambda_numx, const myReal dx2, const myReal c_normal);

