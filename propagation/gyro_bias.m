function [ f, stat, MFG ] = gyro_bias( isreal, use_mex )
close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

if ~exist('use_mex','var') || isempty(use_mex)
    use_mex = false;
end

if use_mex
    addpath('mex');
end

% time
sf = 100;
T = 1;
Nt = T*sf+1;

% band limit
BR = 10;
Bx = 15;
lmax = BR-1;

% grid over SO(3)
alpha = reshape(pi/BR*(0:(2*BR-1)),1,1,[]);
beta = reshape(pi/(4*BR)*(2*(0:(2*BR-1))+1),1,1,[]);
gamma = reshape(pi/BR*(0:(2*BR-1)),1,1,[]);

ca = cos(alpha);
sa = sin(alpha);
cb = cos(beta);
sb = sin(beta);
cg = cos(gamma);
sg = sin(gamma);

Ra = [ca,-sa,zeros(1,1,2*BR);sa,ca,zeros(1,1,2*BR);zeros(1,1,2*BR),zeros(1,1,2*BR),ones(1,1,2*BR)];
Rb = [cb,zeros(1,1,2*BR),sb;zeros(1,1,2*BR),ones(1,1,2*BR),zeros(1,1,2*BR);-sb,zeros(1,1,2*BR),cb];
Rg = [cg,-sg,zeros(1,1,2*BR);sg,cg,zeros(1,1,2*BR);zeros(1,1,2*BR),zeros(1,1,2*BR),ones(1,1,2*BR)];

R = zeros(3,3,2*BR,2*BR,2*BR);
for i = 1:2*BR
    for j = 1:2*BR
        for k = 1:2*BR
            R(:,:,i,j,k) = Ra(:,:,i)*Rb(:,:,j)*Rg(:,:,k);
        end
    end
end

% grid over R^3
x = zeros(3,2*Bx,2*Bx,2*Bx);
L = 2;
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            x(:,i,j,k) = [-L/2+2/(2*Bx)*(i-1);-L/2+2/(2*Bx)*(j-1);-L/2+2/(2*Bx)*(k-1)];
        end
    end
end

% weights
w = zeros(1,2*BR);
for j = 1:2*BR
    w(j) = 1/(4*BR^3)*sin(beta(j))*sum(1./(2*(0:BR-1)+1).*sin((2*(0:BR-1)+1)*beta(j)));
end

% Wigner_d
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*BR);
for j = 1:2*BR
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% derivatives
u = getu(lmax,isreal);
if ~use_mex
    u = gpuArray(u);
end

% Fourier transform of x
X = zeros(2*Bx,2*Bx,2*Bx,3);
for i = 1:3
    X(:,:,:,i) = fftn(x(i,:,:,:));
end

if ~use_mex
    X = gpuArray(X);
end

% initial condition
s = [4,4,4];
S = diag(s);
Miu = [0;0;0.2];
Sigma = 0.2^2*eye(3);
P = [0,0,0;0,0,0;0,0,0]*0.2/sqrt(10);

c = pdf_MF_normal(s);

vR = @(R) permute(cat(1,s(2)*R(3,2,:)-s(3)*R(2,3,:),...
    s(3)*R(1,3,:)-s(1)*R(3,1,:),...
    s(1)*R(2,1,:)-s(2)*R(1,2,:)),[1,3,2]);
SigmaC = Sigma-P*(trace(S)*eye(3)-S)*P.';

func = @(R,x) 1/(c*sqrt((2*pi)^3*det(SigmaC)))*...
    permute(exp(sum(sum(S.'.*R,1),2)),[3,2,1]).*...
    permute(exp(-0.5*sum(sum(permute(x-Miu-P*vR(R),[1,3,2]).*...
        permute(x-Miu-P*vR(R),[1,3,2]).*SigmaC^-1,1),2)),[3,2,1]);

R_linInd = reshape(R,3,3,(2*BR)^3);
f = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx,(Nt-1)/10+1);
for k = 1:2*Bx
    for j = 1:2*Bx
        for i = 1:2*Bx
            f(:,:,:,i,j,k,1) = reshape(func(R_linInd,x(:,i,j,k)),2*BR,2*BR,2*BR);
        end
    end
end

% noise
H1 = diag([0.1,0.1,0.1]);
H2 = diag([0.05,0.05,0.05]);

G1 = 0.5*(H1*H1.');
G2 = 0.5*(H2*H2.');

%% propagation
F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx,(Nt-1)/10+1);

f_temp = f(:,:,:,:,:,:,1);
F_temp = F(:,:,:,:,:,:,1);
for nt = 1:Nt-1
    tic;
    
    % forward
    F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
    for k = 1:2*BR
        F1(:,k,:,:,:,:) = fftn(f_temp(:,k,:,:,:,:));
    end
    F1 = fftshift(fftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);

    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                F_temp(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            end
        end
    end
    
    % propagating Fourier coefficients
    if use_mex
        F_temp = propagate_mex(F_temp,X,1/sf,u,G1,G2);
    else
        F_temp = integrate(F_temp,X,1/sf,u,G1,G2);
    end
    
    % backward
    F1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = F_temp(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);

            for k = 1:2*BR
                d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
                F1(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                    [1,3,2])+1).*F_mn.*d_jk_betak,3);
            end
        end
    end

    F1 = cat(1,F1,zeros(1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx));
    F1 = cat(3,F1,zeros(2*BR,2*BR,1,2*Bx,2*Bx,2*Bx));
    F1 = ifftshift(ifftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    for k = 1:2*BR
        f_temp(:,k,:,:,:,:) = ifftn(F1(:,k,:,:,:,:),'symmetric')*(2*BR)^2;
    end
    
    if rem(nt,10)==0
        f(:,:,:,:,:,:,nt/10+1) = f_temp;
        F(:,:,:,:,:,:,nt/10+1) = F_temp;
    end
    
    toc;
end

%% statistics
MFG.U = zeros(3,3,(Nt-1)/10+1);
MFG.V = zeros(3,3,(Nt-1)/10+1);
MFG.S = zeros(3,3,(Nt-1)/10+1);
MFG.Miu = zeros(3,(Nt-1)/10+1);
MFG.Sigma = zeros(3,3,(Nt-1)/10+1);
MFG.P = zeros(3,3,(Nt-1)/10+1);

stat.ER = zeros(3,3,(Nt-1)/10+1);
for nt = 1:(Nt-1)/10+1
    fR = sum(f(:,:,:,:,:,:,nt),[4,5,6])*(L/(2*Bx))^3;
    stat.ER(:,:,nt) = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);
end

stat.Ex = zeros(3,(Nt-1)/10+1);
stat.Varx = zeros(3,3,(Nt-1)/10+1);
for nt = 1:(Nt-1)/10+1
    fx = permute(sum(f(:,:,:,:,:,:,nt).*w,[1,2,3]),[1,4,5,6,2,3]);
    stat.Ex(:,nt) = sum(x.*fx,[2,3,4])*(L/(2*Bx))^3;
    stat.Varx(:,:,nt) = sum(permute(x,[1,5,2,3,4]).*permute(x,[5,1,2,3,4]).*...
        permute(fx,[1,5,2,3,4]),[3,4,5])*(L/(2*Bx))^3 - stat.Ex(:,nt)*stat.Ex(:,nt).';
end

stat.EvR = zeros(3,(Nt-1)/10+1);
stat.ExvR = zeros(3,3,(Nt-1)/10+1);
stat.EvRvR = zeros(3,3,(Nt-1)/10+1);
for nt = 1:(Nt-1)/10+1
    [U,D,V] = psvd(stat.ER(:,:,nt));
    s = pdf_MF_M2S(diag(D),diag(S));
    
    MFG.U(:,:,nt) = U;
    MFG.V(:,:,nt) = V;
    MFG.S(:,:,nt) = diag(s);
    
    Q = gather(pagefun(@mtimes,U.',pagefun(@mtimes,gpuArray(R),V)));
    vR = permute(cat(1,s(2)*Q(3,2,:,:,:)-s(3)*Q(2,3,:,:,:),...
        s(3)*Q(1,3,:,:,:)-s(1)*Q(3,1,:,:,:),...
        s(1)*Q(2,1,:,:,:)-s(2)*Q(1,2,:,:,:)),[1,3,4,5,2]);
    fR = sum(f(:,:,:,:,:,:,nt),[4,5,6])*(L/(2*Bx))^3;
    
    stat.EvR(:,nt) = sum(vR.*permute(w,[1,3,2]).*permute(fR,[4,1,2,3]),[2,3,4]);
    stat.EvRvR(:,:,nt) = sum(permute(vR,[1,5,2,3,4]).*permute(vR,[5,1,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(fR,[4,5,1,2,3]),[3,4,5]);
    stat.ExvR(:,:,nt) = sum(permute(vR,[5,1,2,3,4]).*permute(x,[1,5,6,7,8,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(f(:,:,:,:,:,:,nt),[7,8,1,2,3,4,5,6]),[3,4,5,6,7,8])*(L/(2*Bx))^3;
end

for nt = 1:(Nt-1)/10+1
    covxx = stat.Varx(:,:,nt);
    covxvR = stat.ExvR(:,:,nt)-stat.Ex(:,nt)*stat.EvR(:,nt).';
    covvRvR = stat.EvRvR(:,:,nt)-stat.EvR(:,nt)*stat.EvR(:,nt).';
    
    MFG.P(:,:,nt) = covxvR*covvRvR^-1;
    MFG.Miu(:,nt) = stat.Ex(:,nt)-MFG.P(:,:,nt)*stat.EvR(:,nt);
    MFG.Sigma(:,:,nt) = covxx-MFG.P(:,:,nt)*covxvR.'+...
        MFG.P(:,:,nt)*(trace(MFG.S(:,:,nt))*eye(3)-MFG.S(:,:,nt))*MFG.P(:,:,nt).';
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');
if use_mex
    addpath('mex');
end

end


function [ Fnew ] = integrate( Fold, X, dt, u, G1, G2 )

BR = size(Fold,3);
Bx = size(Fold,4)/2;
lmax = BR-1;

Fold = gpuArray(Fold);

% gyro bias
temp1 = zeros(size(Fold));
temp2 = zeros(size(Fold));
temp3 = zeros(size(Fold));
for ix = 1:2*Bx
    for jx = 1:2*Bx
        for kx = 1:2*Bx
            X_ijk = flip(flip(flip(X,1),2),3);
            X_ijk = circshift(X_ijk,ix,1);
            X_ijk = circshift(X_ijk,jx,2);
            X_ijk = circshift(X_ijk,kx,3);
            X_ijk = permute(X_ijk,[5,6,7,1,2,3,4]);
            
            temp1(:,:,:,ix,jx,kx) = gather(sum(X_ijk(:,:,:,:,:,:,1).*Fold,[4,5,6])/(2*Bx)^3);
            temp2(:,:,:,ix,jx,kx) = gather(sum(X_ijk(:,:,:,:,:,:,2).*Fold,[4,5,6])/(2*Bx)^3);
            temp3(:,:,:,ix,jx,kx) = gather(sum(X_ijk(:,:,:,:,:,:,3).*Fold,[4,5,6])/(2*Bx)^3);
        end
    end
end

dF1 = gpuArray.zeros(size(Fold));
for l = 0:lmax
    indmn = -l+lmax+1:l+lmax+1;
    dF1(indmn,indmn,l+1,:,:,:) = dF1(indmn,indmn,l+1,ix,jx,kx)-...
        pagefun(@mtimes,gpuArray(temp1(indmn,indmn,l+1,:,:,:)),u(indmn,indmn,l+1,1).')-...
        pagefun(@mtimes,gpuArray(temp2(indmn,indmn,l+1,:,:,:)),u(indmn,indmn,l+1,2).')-...
        pagefun(@mtimes,gpuArray(temp3(indmn,indmn,l+1,:,:,:)),u(indmn,indmn,l+1,3).');
end

clear temp1 temp2 temp3

% gyro random walk
for l = 0:lmax
    indmn = -l+lmax+1:l+lmax+1;
    for i = 1:3
        for j = 1:3
            dF1(indmn,indmn,l+1,:,:,:) = dF1(indmn,indmn,l+1,:,:,:) + ...
                G1(i,j)*pagefun(@mtimes,pagefun(@mtimes,Fold(indmn,indmn,l+1,:,:,:),...
                u(indmn,indmn,l+1,i).'),u(indmn,indmn,l+1,j).');
        end
    end
end

% bias randomwalk
for i = 1:3
    for j = 1:3
        if i==j
            c = pi^2*[0:Bx-1,-Bx:-1].^2;
            c = -shiftdim(c,-(i+1));
        else
            c = pi*[0:Bx-1,0,-Bx+1:-1];
            c = -shiftdim(c,-(i+1)).*shiftdim(c,-(j+1));
        end
        
        dF1 = dF1 + G2(i,j)*Fold.*c;
    end
end

Fnew = gather(Fold+dF1*dt);

end

