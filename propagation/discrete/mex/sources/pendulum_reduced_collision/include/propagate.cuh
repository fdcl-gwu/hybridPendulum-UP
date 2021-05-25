#include "setup.hpp"

void get_df_noise(myReal* df, const myReal* f, const myReal* lambda, myReal* const* fcL, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, int* const* fcL_indx1, const int* fcL_numx1, int* const* fcL_indx2, int* const* fcL_numx2, const Size_f* size_f);

void get_df_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int numR, const int* indR, int* const* lambda_indx, const int* lambda_numx, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f);

__global__ void get_fold_noise(myReal* f_temp, const myReal* f, const int* fcL_indx2, const int fcL_numx2, const Size_f* size_f);

__global__ void get_fin_nonoise(myReal* df, const myReal* f, const myReal* lambda, const int* indR, const int* ind_interp, const myReal* coeff_interp, const Size_f* size_f);

__global__ void get_fout(myReal* df, const myReal* f, const myReal lambda, const int* lambda_indx, const int lambda_numx, const Size_f* size_f);

