#include "setup.hpp"

void get_df(myReal* df, const myReal* f, const myReal* x, const myReal* lambda, const myReal* Omega, const char* lambda_cat, const myReal* Gd, Size_f* size_f);

__global__ void get_fc(myReal* fc, const myReal* x, const myReal* Omega, const myReal* invGd, const myReal c, const int nn0, const Size_f* size_f);

__global__ void get_flambda(myReal* flambda, const myReal* f, const myReal* lambda, const myReal dx2, const Size_f* size_f);

__global__ void sub_lambda_f(myReal* df, const myReal* lambda, const myReal* f, const Size_f* size_f);

