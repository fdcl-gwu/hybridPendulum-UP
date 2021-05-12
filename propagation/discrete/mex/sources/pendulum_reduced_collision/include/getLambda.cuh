#include "setup.hpp"

void getLambda(myReal* lambda, char* lambda_cat, const myReal* R, const myReal* x, const myReal* d, const myReal* h, const myReal* r, const myReal* thetat, const myReal* lmabda_max, const Size_f* size_f);

__global__ void getTheta(myReal* theta, const myReal* R, const Size_f* size_f);
__global__ void getPC(myReal* PC, const myReal* R, const myReal* theta, const myReal h, const myReal r, const Size_f* size_f);
__global__ void compute_lambda(myReal* lambda, char* lambda_cat, const myReal* theta, const myReal theta0, const myReal thetat, const myReal lambda_max, const Size_f* size_f);
__global__ void expand_lambda(myReal* lambda, const myReal* R, const myReal* x, const myReal* PC, const int* ind_n0, const Size_f* size_f);

