#ifndef SETUP
#define SETUP

#define FP32 false
#if FP32

#else
	typedef double myReal;
    typedef double myComplex[2];

    #define mysin sin
    #define mycos cos
    #define mysqrt sqrt

	#define mymxGetComplex mxGetComplexDoubles
	#define mymxGetReal mxGetDoubles
	#define mymxRealClass mxDOUBLE_CLASS
#endif

constexpr myReal PI = 3.141592653589793;

struct Size_F {
	int ndims;

	int BR;
	int Bx;
	int lmax;

	int nR;
	int nx;
	int nTot;
	int nR_compact;
	int nTot_compact;

	int const_2Bx;
	int const_2Bxs;
	int const_2lp1;
	int const_lp1;
	int const_2lp1s;

	int l_cum0;
	int l_cum1;
	int l_cum2;
	int l_cum3;
	int l_cum4;
};

struct Size_f {
	int ndims;

	int BR;
	int Bx;

	int nR;
	int nx;
	int nTot;

	int const_2Bx;
	int const_2BR;
	int const_2Bxs;
	int const_2BRs;
};

struct Size_e {
    int ns;
    int na;
};

void init_Size_F(Size_F* size_F, const int BR, const int Bx, const int ndims);
void init_Size_f(Size_f* size_f, const int BR, const int Bx, const int ndims);
void init_Size_e(Size_e* size_e, const int ns, const int na);

#endif // !setup

