#include "setup.hpp"

void getLambda(myReal* lambda, int* lambda_indR, int* lambda_numR, myReal* PC, const myReal* R, const myReal* x, const myReal* d, const myReal* h, const myReal* r, const myReal* thetat, const myReal* lambda_max, const Size_f* size_f);

void getLambda_compact(myReal* lambda, int* lambda_indR, int* lambda_numR, int** lambda_indx, int* lambda_numx, myReal* PC, const myReal* R, const myReal* x, const myReal* d, const myReal* h, const myReal* r, const myReal* thetat, const myReal* lmabda_max, const Size_f* size_f);

__global__ void getTheta(myReal* theta, char* lambda_cat, const myReal* R, const myReal theta0, const myReal thetat, const Size_f* size_f);

__global__ void getPC(myReal* PC, const myReal* R, const myReal* theta, const myReal h, const myReal r, const Size_f* size_f);

__global__ void compute_lambda(myReal* lambda, const char* lambda_cat, const int* lambda_indR, const myReal* theta, const myReal theta0, const myReal thetat, const myReal lambda_max, const int numR);

__global__ void get_indx(int* lambda_indx, const myReal* R, const myReal* x, const myReal* PC, const Size_f* size_f);

__global__ void expand_lambda(myReal* lambda, const myReal* R, const myReal* x, const myReal* PC, const int* indR);

