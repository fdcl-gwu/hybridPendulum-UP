#include "setup.hpp"

void get_color(myReal* c, const myReal* f, const myReal* e, const myReal* L, const Size_F* size_F, const Size_f* size_f, const Size_e* size_e);

void get_margin(myReal* fR, const myReal* f, const myReal* L, const Size_f* size_f);

void fftSO3(myComplex* FR, myReal* fR, const Size_F* size_F, const Size_f* size_f);

myReal ifftSO3_single(const myComplex* F, const myReal* e, const Size_F* size_F, const Size_f* size_f);

void wigner_d(myReal* d, const myReal beta, const int lmax);

void wigner_D(myComplex* D, const myReal* d, const myReal alpha, const myReal gamma, const int lmax);

void shiftflip(myComplex* F, const Size_f* size_f);

