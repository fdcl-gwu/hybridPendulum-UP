function [ dF, df, F, f, R ] = fftSO3_deriv( func, B, eta, isreal )

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

lmax = B-1;

% forward transform
[F,f,R] = fftSO3(func,B,isreal);

% u
u = getu(lmax,isreal);

% derivatives
dF = zeros(2*lmax+1,2*lmax+1,lmax+1);

for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    dF(ind,ind,l+1) = eta(1)*F(ind,ind,l+1)*u(ind,ind,l+1,1).' + ...
        eta(2)*F(ind,ind,l+1)*u(ind,ind,l+1,2).' + ...
        eta(3)*F(ind,ind,l+1)*u(ind,ind,l+1,3).';
end

% evaluate derivative values
df = diff_func(func,eta,R);

end


function [ df ] = diff_func( func, eta, R )

addpath('rotation3d');

normEta = sqrt(sum(eta.^2));

syms t;
expt = eye(3) + sin(t*normEta)/normEta*hat(eta) + ...
    (1-cos(t*normEta))/normEta^2*hat(eta)^2;

syms R11 R12 R13 R21 R22 R23 R31 R32 R33
Rsyms = [R11,R12,R13;R21,R22,R23;R31,R32,R33];

d_func = diff(func(Rsyms*expt),t);
d_func = subs(d_func,t,0);
d_func = matlabFunction(d_func,'Var',[R11,R12,R13,R21,R22,R23,R31,R32,R33]);

df = d_func(R(1,1,:,:,:),R(1,2,:,:,:),R(1,3,:,:,:),...
    R(2,1,:,:,:),R(2,2,:,:,:),R(2,3,:,:,:),...
    R(3,1,:,:,:),R(3,2,:,:,:),R(3,3,:,:,:));

df = permute(df,[3,4,5,1,2]);

rmpath('rotation3d');

end

