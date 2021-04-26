#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "omp.h"
#include "fftw3.h"
#include "getc.hpp"

void get_color(myReal* c, const myReal* f, const myReal* e, const myReal* L, const Size_F* size_F, const Size_f* size_f, const Size_e* size_e)
{
    // get marginal distribution
    myReal* fR = (myReal*) malloc(size_f->nR*sizeof(myReal));
    get_margin(fR, f, L, size_f);

    // get Fourier coefficient
    myComplex* FR = (myComplex*) malloc(size_F->nR_compact*sizeof(myComplex));
    fftSO3(FR, fR, size_F, size_f);

    // test
    int ind_gap = 3*size_e->ns;
    memset(c, 0, size_e->ns*3*sizeof(myReal));

    #pragma omp parallel for num_threads(30)
    for (int ns = 0; ns < size_e->ns; ns++) {
        int ind_e = 3*ns;
        int ind_c = ns;
        for (int nd = 0; nd < 3; nd++) {
            for (int na = 0; na < size_e->na; na++) {
                c[ind_c] += ifftSO3_single(FR, e+ind_e, size_F, size_f);

                ind_e += ind_gap;
            }
            ind_c += size_e->ns;
        }
    }

    // free memory
    free(fR);
    free(FR);
}

void get_margin(myReal* fR, const myReal* f, const myReal* L, const Size_f* size_f)
{
    // marginal distribution
    memset(fR, 0, size_f->nR*sizeof(myReal));

    myReal gridsize = *L/(myReal)size_f->const_2Bx;

    #pragma omp parallel for num_threads(30)
    for (int iR = 0; iR < size_f->nR; iR++) {
        int ind_f = iR;
        for (int ix = 0; ix < size_f->nx; ix++) {
            fR[iR] += f[ind_f];
            ind_f += size_f->nR;
        }

        for (int id = 0; id < size_f->ndims-3; id++) {
            fR[iR] *= gridsize;
        }
    }
}

void fftSO3(myComplex* FR, myReal* fR, const Size_F* size_F, const Size_f* size_f)
{
    // fft for R1 and R3
    int rank = 2;
    int n[2] = {size_f->const_2BR, size_f->const_2BR};
    int howmany = size_f->const_2BR;
    int inembed[2] = {size_f->const_2BR, size_f->const_2BRs};
    int onembed[2] = {size_f->const_2BR, size_f->const_2BRs};
    int istride = 1;
    int ostride = 1;
    int idist = size_f->const_2BR;
    int odist = size_f->const_2BR;
    
    myComplex* FR1 = (myComplex*) fftw_malloc(size_f->nR*sizeof(myComplex));
    fftw_plan plan = fftw_plan_many_dft_r2c(2, n, howmany, fR, inembed, istride, idist,
        FR1, onembed, ostride, odist, FFTW_MEASURE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    for (int i = 1+size_f->BR; i < size_f->const_2BR; i++) {
        for (int j = 0; j < size_f->const_2BR; j++) {
            for (int k = 0; k < size_f->const_2BR; k++) {
                int indf_t = i + j*size_f->const_2BR + k*size_f->const_2BRs;
                int indf_s;
                if (k==0) {
                    indf_s = size_f->const_2BR-i + j*size_f->const_2BR;
                } else {
                    indf_s = size_f->const_2BR-i + j*size_f->const_2BR + (size_f->const_2BR-k)*size_f->const_2BRs;
                }
                
                FR1[indf_t][0] = FR1[indf_s][0];
                FR1[indf_t][1] = -FR1[indf_s][1];
            }
        }
    }

    shiftflip(FR1, size_f);

    // fft for R2
    myReal* beta = (myReal*) malloc(size_f->const_2BR*sizeof(myReal));
    for (int i = 0; i < size_f->const_2BR; i++) {
        beta[i] = (myReal) PI/(4*size_f->BR)*(2*i+1);
    }

    myReal* d = (myReal*) malloc(size_F->nR_compact*size_f->const_2BR*sizeof(myReal));
    #pragma omp parallel for num_threads(30)
    for (int i = 0; i < size_f->const_2BR; i++) {
        wigner_d(d+i*size_F->nR_compact, beta[i], size_F->lmax);
    }

    myReal* w = (myReal*) malloc(size_f->const_2BR*sizeof(myReal));
    for (int i = 0; i < size_f->const_2BR; i++) {
        myReal sum = 0;
        for (int l = 0; l < size_f->BR; l++) {
            sum += (myReal) 1/(2*l+1)*mysin((2*l+1)*beta[i]);
        }
        w[i] = (myReal) 1/(4*size_f->BR*size_f->BR*size_f->BR)*mysin(beta[i])*sum;
    }

    memset(FR, 0, size_F->nR_compact*sizeof(myComplex));
    int indFd = 0;
    for (int l = 0; l <= size_F->lmax; l++) {
        #pragma omp parallel for
        for (int m = -l; m <= l; m++) {
            #pragma omp parallel for
            for (int n = -l; n <= l; n++) {
                int indFd_mn = indFd + m+l + (n+l)*(2*l+1);
                int indFR1 = m+size_F->lmax + (n+size_F->lmax)*size_f->const_2BRs;
                for (int i = 0; i < size_f->const_2BR; i++) {
                    FR[indFd_mn][0] += w[i]*FR1[indFR1+i*size_f->const_2BR][0]*d[indFd_mn+i*size_F->nR_compact];
                    FR[indFd_mn][1] += w[i]*FR1[indFR1+i*size_f->const_2BR][1]*d[indFd_mn+i*size_F->nR_compact];
                }
            }
        }
        
        indFd += (2*l+1)*(2*l+1);
    }

    // free memory
    fftw_free(FR1);
    free(beta);
    free(d);
    free(w);
}

myReal ifftSO3_single(const myComplex* F, const myReal* e, const Size_F* size_F, const Size_f* size_f)
{
    // wigner matrices
    myReal* d = (myReal*) malloc(size_F->nR_compact*sizeof(myReal));
    myComplex* D = (myComplex*) malloc(size_F->nR_compact*sizeof(myComplex));
    wigner_d(d, e[1], size_F->lmax);
    wigner_D(D, d, e[0], e[2], size_F->lmax);

    // summation
    myReal f = 0;
    int indF = 0;
    for (int l = 0; l <= size_F->lmax; l++) {
        int const_2lp1 = 2*l+1;
        for (int n = 0; n < const_2lp1; n++) {
            for (int m = 0; m < const_2lp1; m++) {
                f += const_2lp1*(F[indF][0]*D[indF][0]-F[indF][1]*D[indF][1]);

                indF++;
            }
        }
    }

    // free memory
    free(d);
    free(D);

    return f;
}

void wigner_d(myReal* d, const myReal beta, const int lmax)
{
    myReal cb = mycos(beta);
    myReal cb2 = mycos(beta/2);
    myReal sb2 = mysin(beta/2);

    // pre-calculation
    int* indl = (int*) malloc((lmax+1)*sizeof(int));
    indl[0] = 0;
    for (int l = 0; l < lmax; l++) {
        indl[l+1] = indl[l] + (2*l+1)*(2*l+1);
    }

    myReal* factorial = (myReal*) malloc((2*lmax+1)*sizeof(myReal));
    factorial[0] = 1.0;
    for (int l = 1; l <= 2*lmax; l++) {
        factorial[l] = factorial[l-1]*(myReal)l;
    }

    // l=0
    d[0] = 1;
    
    // l=1
    d[1] = cb2*cb2;
    d[2] = -mysqrt(2.0)*cb2*sb2;
    d[3] = sb2*sb2;
    d[5] = cb;
    
    d[4] = -d[2];
    d[6] = d[2];
    d[7] = d[3];
    d[8] = -d[2];
    d[9] = d[1];

    // l >= 2
    for (int l = 2; l <= lmax; l++) {
        myReal* dl = d+indl[l];
        myReal* dlp = d+indl[l-1];
        myReal* dlpp = d+indl[l-2];
        int const_2lp1 = 2*l+1;
        
        int ind = 0;
        for (int n = -l; n <= 0; n++) {
            for (int m = n; m <= -n; m++) {
                if (n == -l) {
                    dl[ind] = mysqrt(factorial[2*l]/factorial[l+m]/factorial[l-m])*
                        (myReal)pow((double)cb2,(double)(l-m))*(myReal)pow((double)-sb2,(double)(l+m));
                } else if (n == -l+1) {
                    dl[ind] = (myReal) l*(const_2lp1-2)/mysqrt((myReal)(l*l-m*m)*(l*l-n*n))*
                        (cb-(myReal)m*n/(l-1)/l)*dlp[m+l-1+(n+l-1)*(const_2lp1-2)];
                } else {
                    dl[ind] = (myReal) l*(const_2lp1-2)/mysqrt((myReal)(l*l-m*m)*(l*l-n*n))*
                        (cb-(myReal)m*n/(l-1)/l)*dlp[m+l-1+(n+l-1)*(const_2lp1-2)] - 
                        mysqrt((myReal)((l-1)*(l-1)-m*m)*((l-1)*(l-1)-n*n))/mysqrt((myReal)(l*l-m*m)*(l*l-n*n))*
                        (myReal)l/(l-1)*dlpp[m+l-2+(n+l-2)*(const_2lp1-4)];
                }

                ind += 1;
            }
            ind += 2*(n+l)+1;
        }

        for (int m = 1; m <= l; m++) {
            for (int n = -m+1; n <= m; n++) {
                dl[m+l+(n+l)*const_2lp1] = dl[-n+l+(-m+l)*const_2lp1];
            }
        }

        for (int m = -l; m <= l-1; m++) {
            for (int n = m+1; n <= l; n++) {
                dl[m+l+(n+l)*const_2lp1] = dl[n+l+(m+l)*const_2lp1]*((m-n)%2 ? -1 : 1);
            }
        }
    }

    // free memory
    free(indl);
    free(factorial);
}

void wigner_D(myComplex* D, const myReal* d, const myReal alpha, const myReal gamma, const int lmax)
{
    int const_2lps = 2*lmax+1;
    myComplex* Fa = (myComplex*) malloc(const_2lps*sizeof(myComplex));
    myComplex* Fg = (myComplex*) malloc(const_2lps*sizeof(myComplex));
    myComplex* FaFg = (myComplex*) malloc(const_2lps*const_2lps*sizeof(myComplex));

    int ind = 0;
    for (int m = -lmax; m <= lmax; m++) {
        Fa[ind][0] = mycos(m*alpha);
        Fa[ind][1] = -mysin(m*alpha);
        Fg[ind][0] = mycos(m*gamma);
        Fg[ind][1] = -mysin(m*gamma);
        ind++;
    }

    int indmn = 0;
    int indn = 0;
    for (int n = 0; n < const_2lps; n++) {
        int indm = 0;
        for (int m = 0; m < const_2lps; m++) {
            FaFg[indmn][0] = Fa[indm][0]*Fg[indn][0]-Fa[indm][1]*Fg[indn][1];
            FaFg[indmn][1] = Fa[indm][0]*Fg[indn][1]+Fa[indm][1]*Fg[indn][0];

            indm++;
            indmn++;
        }
        indn++;
    }

    int indD = 0;
    for (int l = 0; l <= lmax; l++) {
        for (int n = -l; n <= l; n++) {
            int indn = (n+lmax)*const_2lps;
            for (int m = -l; m <= l; m++) {
                int indmn = m+lmax+indn;
                D[indD][0] = d[indD]*FaFg[indmn][0];
                D[indD][1] = d[indD]*FaFg[indmn][1];

                indD++;
            }
        }
    }

    // free memory
    free(Fa);
    free(Fg);
    free(FaFg);
}

void shiftflip(myComplex* F, const Size_f* size_f)
{
    myComplex* FF = (myComplex*) malloc(size_f->nR*sizeof(myComplex));

    int indF = 0;
    for (int k = 0; k < size_f->const_2BR; k++) {
        int indk_sf = k < size_f->BR ? size_f->BR-k-1 : 3*size_f->BR-k-1;
        int indFF_k = indk_sf*size_f->const_2BRs;
        
        for (int j = 0; j < size_f->const_2BR; j++) {
            int indFF_jk = indFF_k + j*size_f->const_2BR;

            for (int i = 0; i < size_f->const_2BR; i++) {
                int indi_sf = i < size_f->BR ? size_f->BR-i-1 : 3*size_f->BR-i-1;
                int indFF = indFF_jk+indi_sf;

                FF[indFF][0] = F[indF][0];
                FF[indFF][1] = F[indF][1];
                indF++;
            }
        }
    }

    memcpy(F, FF, size_f->nR*sizeof(myComplex));
    free(FF);
}

