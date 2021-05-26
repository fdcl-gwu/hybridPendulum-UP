#include "setup.hpp"

void getFcL(myReal* fcL, short* fcL_indx, const myReal* x, const myReal* Omega, const myReal* lambda, const int numR, const myReal* Gd, const int nD, const Size_f* size_f);

__global__ void get_fc(myReal* fc_x2, const myReal* x, const myReal* Omega, const myReal* invGd, const int numR, const myReal c_normal);

__global__ void get_fc_normal(myReal* fc, const myReal* fc_normal, const short* fc_indx, const int numR);

__global__ void get_fcL(myReal* fcL, const myReal* lambda, const short* fcL_indx, const myReal dx2);

