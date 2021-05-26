#include "setup.hpp"

void get_df_noise(myReal* df, const myReal* f, const myReal* lambda, const myReal* fcL, const int numR, const int* indR, const short* fcL_indx, const int nD, const Size_f* size_f);

void get_df_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f);

__global__ void get_fold_noise(myReal* f_old, const myReal* f, const int* indR, const short* fcL_indx, const Size_f* size_f);

__global__ void get_fin_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int* indR, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f);

__global__ void get_fout_noise(myReal* df, const myReal* fin, const myReal* f, const myReal* lambda, const int* indR, const Size_f* size_f);

__global__ void get_fout_nonoise(myReal* df, const myReal* f, const myReal lambda, const int* lambda_indx, const int lambda_numx, const Size_f* size_f);

