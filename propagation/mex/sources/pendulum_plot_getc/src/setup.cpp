#include "setup.hpp"

void init_Size_F(Size_F* size_F, const int BR, const int Bx, const int ndims)
{
	size_F->ndims = ndims;
        size_F->BR = BR;
        size_F->Bx = Bx;
        size_F->lmax = BR-1;

        size_F->nR = (2*size_F->lmax+1) * (2*size_F->lmax+1) * (size_F->lmax+1);
        if (ndims == 5) {
                size_F->nx = (2*Bx) * (2*Bx);
        } else {
                size_F->nx = (2*Bx) * (2*Bx) * (2*Bx);
        }

        size_F->nTot = size_F->nR * size_F->nx;

        size_F->nR_compact = (size_F->lmax+1) * (2*size_F->lmax+1) * (2*size_F->lmax+3) / 3;
        size_F->nTot_compact = size_F->nR_compact * size_F->nx;

        size_F->const_2Bx = 2*Bx;
        size_F->const_2Bxs = (2*Bx) * (2*Bx);
        size_F->const_2lp1 = 2*size_F->lmax+1;
        size_F->const_lp1 = size_F->lmax+1;
        size_F->const_2lp1s = (2*size_F->lmax+1) * (2*size_F->lmax+1);

        size_F->l_cum0 = size_F->const_2lp1;
        size_F->l_cum1 = size_F->l_cum0*size_F->const_2lp1;
        size_F->l_cum2 = size_F->l_cum1*size_F->const_lp1;
        size_F->l_cum3 = size_F->l_cum2*size_F->const_2Bx;
        size_F->l_cum4 = size_F->l_cum3*size_F->const_2Bx;
}

void init_Size_f(Size_f* size_f, const int BR, const int Bx, const int ndims)
{
	size_f->ndims = ndims;
        size_f->BR = BR;
        size_f->Bx = Bx;

        size_f->nR = (2*BR) * (2*BR) * (2*BR);
        if (ndims == 5) {
                size_f->nx = (2*Bx) * (2*Bx);
        } else {
                size_f->nx = (2*Bx) * (2*Bx) * (2*Bx);
        }
        size_f->nTot = size_f->nR * size_f->nx;

        size_f->const_2BR = 2*BR;
        size_f->const_2Bx = 2*Bx;
        size_f->const_2BRs = (2*BR) * (2*BR);
        size_f->const_2Bxs = (2*Bx) * (2*Bx);
}

void init_Size_e(Size_e* size_e, const int ns, const int na)
{
    size_e->ns = ns;
    size_e->na = na;
}


