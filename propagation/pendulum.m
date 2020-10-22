function [ f, stat, MFG ] = pendulum( isreal )
close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

% time
sf = 100;
T = 0.1;
Nt = T*sf+1;

% parameters
Jd = diag([1,2,3]);
J = trace(Jd)*eye(3)-Jd;

rho = [0;0;1];
m = 10;
g = 9.8;

% band limit
B = 10;
lmax = B-1;

% grid over SO(3)
alpha = reshape(pi/B*(0:(2*B-1)),1,1,[]);
beta = reshape(pi/(4*B)*(2*(0:(2*B-1))+1),1,1,[]);
gamma = reshape(pi/B*(0:(2*B-1)),1,1,[]);

ca = cos(alpha);
sa = sin(alpha);
cb = cos(beta);
sb = sin(beta);
cg = cos(gamma);
sg = sin(gamma);

Ra = [ca,-sa,zeros(1,1,2*B);sa,ca,zeros(1,1,2*B);zeros(1,1,2*B),zeros(1,1,2*B),ones(1,1,2*B)];
Rb = [cb,zeros(1,1,2*B),sb;zeros(1,1,2*B),ones(1,1,2*B),zeros(1,1,2*B);-sb,zeros(1,1,2*B),cb];
Rg = [cg,-sg,zeros(1,1,2*B);sg,cg,zeros(1,1,2*B);zeros(1,1,2*B),zeros(1,1,2*B),ones(1,1,2*B)];

R = zeros(3,3,2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            R(:,:,i,j,k) = Ra(:,:,i)*Rb(:,:,j)*Rg(:,:,k);
        end
    end
end

% grid over R^3
L = 10;
x = zeros(3,2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            x(:,i,j,k) = [-L/2+L/(2*B)*(i-1);-L/2+L/(2*B)*(j-1);-L/2+L/(2*B)*(k-1)];
        end
    end
end

% weights
w = zeros(1,2*B);
for j = 1:2*B
    w(j) = 1/(4*B^3)*sin(beta(j))*sum(1./(2*(0:B-1)+1).*sin((2*(0:B-1)+1)*beta(j)));
end

% Wigner_d
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
for j = 1:2*B
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% derivatives
u = getu(lmax,isreal);

% CG coefficient
warning('off','MATLAB:nearlySingularMatrix');

CG = cell(B,B);
for l1 = 0:lmax
    for l2 = 0:lmax
        CG{l1+1,l2+1} = clebsch_gordan(l1,l2);
    end
end

warning('on','MATLAB:nearlySingularMatrix');

% Fourier transform of x
X = gpuArray.zeros(2*B,2*B,2*B,3);
for i = 1:3
    X(:,:,:,i) = fftn(x(i,:,:,:));
end

% Fourier transform of Omega cross JOmega
ojo = -cross(x,permute(pagefun(@mtimes,J,permute(gpuArray(x),...
    [1,5,2,3,4])),[1,3,4,5,2]));
ojo = permute(pagefun(@mtimes,J^-1,permute(ojo,[1,5,2,3,4])),[1,3,4,5,2]);

OJO = gpuArray.zeros(2*B,2*B,2*B,3);
for i = 1:3
    OJO(:,:,:,i) = fftn(ojo(i,:,:,:));
end

clear ojo;

% Fourier transform of M(R)
mR = -m*g*cross(repmat(rho,1,2*B,2*B,2*B),permute(R(3,:,:,:,:),[2,3,4,5,1]));
mR = permute(pagefun(@mtimes,J^-1,permute(gpuArray(mR),[1,5,2,3,4])),[1,3,4,5,2]);

MR = gpuArray.zeros(2*lmax+1,2*lmax+1,lmax+1,3);
for i = 1:3
    F1 = gpuArray.zeros(2*B,2*B,2*B);
    for k = 1:2*B
        F1(:,k,:) = fftn(mR(i,:,k,:));
    end
    F1 = fftshift(fftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);

    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                MR(m+lmax+1,n+lmax+1,l+1,i) = sum(w.*F1(m+lmax+1,:,n+lmax+1).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]));
            end
        end
    end
end
clear mR F1;

% initial conditions
S = diag([4,4,4]);
U = expRot([pi*2/3,0,0]);
Miu = [0;0;0];
Sigma = 1^2*eye(3);

c = pdf_MF_normal(diag(S));

f = zeros(2*B,2*B,2*B,2*B,2*B,2*B,Nt);
f(:,:,:,:,:,:,1) = permute(exp(sum(U*S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,5,2,3,4]).*permute((x-Miu),...
    [5,1,2,3,4]).*Sigma^-1,[1,2])),[1,2,6,3,4,5])/c/sqrt((2*pi)^3*det(Sigma));

%% propagation
F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B,2*B,2*B,Nt);

for nt = 1:Nt-1
    tic;
    
    % forward
    F1 = zeros(2*B,2*B,2*B,2*B,2*B,2*B);
    for k = 1:2*B
        F1(:,k,:,:,:,:) = fftn(f(:,k,:,:,:,:,nt));
    end
    F1 = fftshift(fftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    
    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                F(m+lmax+1,n+lmax+1,l+1,:,:,:,nt) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            end
        end
    end
    
    % propagating Fourier coefficients
    F(:,:,:,:,:,:,nt+1) = integrate(F(:,:,:,:,:,:,nt),X,OJO,MR,1/sf,L,u,CG);
    
    % backward
    F1 = zeros(2*B-1,2*B,2*B-1,2*B,2*B,2*B);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:,nt+1);

            for k = 1:2*B
                d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
                F1(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                    [1,3,2])+1).*F_mn.*d_jk_betak,3);
            end
        end
    end

    F1 = cat(1,F1,zeros(1,2*B,2*B-1,2*B,2*B,2*B));
    F1 = cat(3,F1,zeros(2*B,2*B,1,2*B,2*B,2*B));
    F1 = ifftshift(ifftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    for k = 1:2*B
        f(:,k,:,:,:,:,nt+1) = ifftn(F1(:,k,:,:,:,:),'symmetric')*(2*B)^2;
    end
    
    toc;
end

%% statistics
MFG.U = zeros(3,3,Nt);
MFG.V = zeros(3,3,Nt);
MFG.S = zeros(3,3,Nt);
MFG.Miu = zeros(3,Nt);
MFG.Sigma = zeros(3,3,Nt);
MFG.P = zeros(3,3,Nt);

stat.ER = zeros(3,3,Nt);
for nt = 1:Nt
    fR = sum(f(:,:,:,:,:,:,nt),[4,5,6])*(L/2/B)^3;
    stat.ER(:,:,nt) = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);
end

stat.Ex = zeros(3,Nt);
stat.Varx = zeros(3,3,Nt);
for nt = 1:Nt
    fx = permute(sum(f(:,:,:,:,:,:,nt).*w,[1,2,3]),[1,4,5,6,2,3]);
    stat.Ex(:,nt) = sum(x.*fx,[2,3,4])*(L/2/B)^3;
    stat.Varx(:,:,nt) = sum(permute(x,[1,5,2,3,4]).*permute(x,[5,1,2,3,4]).*...
        permute(fx,[1,5,2,3,4]),[3,4,5])*(L/2/B)^3 - stat.Ex(:,nt)*stat.Ex(:,nt).';
end

stat.EvR = zeros(3,Nt);
stat.ExvR = zeros(3,3,Nt);
stat.EvRvR = zeros(3,3,Nt);
for nt = 1:Nt
    [U,D,V] = psvd(stat.ER(:,:,nt));
    s = pdf_MF_M2S(diag(D),diag(S));
    
    MFG.U(:,:,nt) = U;
    MFG.V(:,:,nt) = V;
    MFG.S(:,:,nt) = diag(s);
    
    Q = gather(pagefun(@mtimes,U.',pagefun(@mtimes,gpuArray(R),V)));
    vR = permute(cat(1,s(2)*Q(3,2,:,:,:)-s(3)*Q(2,3,:,:,:),...
        s(3)*Q(1,3,:,:,:)-s(1)*Q(3,1,:,:,:),...
        s(1)*Q(2,1,:,:,:)-s(2)*Q(1,2,:,:,:)),[1,3,4,5,2]);
    fR = sum(f(:,:,:,:,:,:,nt),[4,5,6])*(L/2/B)^3;
    
    stat.EvR(:,nt) = sum(vR.*permute(w,[1,3,2]).*permute(fR,[4,1,2,3]),[2,3,4]);
    stat.EvRvR(:,:,nt) = sum(permute(vR,[1,5,2,3,4]).*permute(vR,[5,1,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(fR,[4,5,1,2,3]),[3,4,5]);
    stat.ExvR(:,:,nt) = sum(permute(vR,[5,1,2,3,4]).*permute(x,[1,5,6,7,8,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(f(:,:,:,:,:,:,nt),[7,8,1,2,3,4,5,6]),[3,4,5,6,7,8])*(L/2/B)^3;
end

for nt = 1:Nt
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

end


function [ Fnew ] = integrate( Fold, X, OJO, MR, dt, L, u, CG )

B = size(Fold,3);
lmax = B-1;

Fold = gpuArray(Fold);
u = gpuArray(u);

dF1 = gpuArray.zeros(size(Fold));

% Omega hat
temp1 = gpuArray.zeros(size(dF1));
temp2 = gpuArray.zeros(size(dF1));
temp3 = gpuArray.zeros(size(dF1));
for ix = 1:2*B
    for jx = 1:2*B
        for kx = 1:2*B
            X_ijk = flip(flip(flip(X,1),2),3);
            X_ijk = circshift(X_ijk,ix,1);
            X_ijk = circshift(X_ijk,jx,2);
            X_ijk = circshift(X_ijk,kx,3);
            X_ijk = permute(X_ijk,[5,6,7,1,2,3,4]);
            
            temp1(:,:,:,ix,jx,kx) = sum(X_ijk(:,:,:,:,:,:,1).*Fold,[4,5,6])/(2*B)^3;
            temp2(:,:,:,ix,jx,kx) = sum(X_ijk(:,:,:,:,:,:,2).*Fold,[4,5,6])/(2*B)^3;
            temp3(:,:,:,ix,jx,kx) = sum(X_ijk(:,:,:,:,:,:,3).*Fold,[4,5,6])/(2*B)^3;
        end
    end
end

for l = 0:lmax
    indmn = -l+lmax+1:l+lmax+1;
    dF1(indmn,indmn,l+1,:,:,:) = dF1(indmn,indmn,l+1,ix,jx,kx)-...
        pagefun(@mtimes,temp1(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,1).')-...
        pagefun(@mtimes,temp2(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,2).')-...
        pagefun(@mtimes,temp3(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,3).');
end

clear temp1 temp2 temp3

% cross(Omega,J*Omega)
for ix = 1:2*B
    for jx = 1:2*B
        for kx = 1:2*B
            OJO_ijk = flip(flip(flip(OJO,1),2),3);
            OJO_ijk = circshift(OJO_ijk,ix,1);
            OJO_ijk = circshift(OJO_ijk,jx,2);
            OJO_ijk = circshift(OJO_ijk,kx,3);
            OJO_ijk = permute(OJO_ijk,[5,6,7,1,2,3,4]);
            
            temp1 = sum(OJO_ijk(:,:,:,:,:,:,1).*Fold,[4,5,6])/(2*B)^3;
            temp2 = sum(OJO_ijk(:,:,:,:,:,:,2).*Fold,[4,5,6])/(2*B)^3;
            temp3 = sum(OJO_ijk(:,:,:,:,:,:,3).*Fold,[4,5,6])/(2*B)^3;
            
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx)-...
                temp1*deriv_x(ix,B,L)-temp2*deriv_x(jx,B,L)-temp3*deriv_x(kx,B,L);
        end
    end
end

clear temp1 temp2 temp3

% -mg*cross(rho,R'*e3)
temp1 = zeros(size(Fold));
temp2 = zeros(size(Fold));
temp3 = zeros(size(Fold));
for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    for l2 = 0:lmax
        ind2 = -l2+lmax+1:l2+lmax+1;
        l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
        for l1 = l1_all
            cl = (2*l1+1)*(2*l2+1)/(2*l+1);
            col = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
            ind1 = -l1+lmax+1:l1+lmax+1;
            for ix = 1:2*B
                kron_FMR1 = reshape(permute(Fold(ind1,ind1,l1+1,:,:,ix),[3,1,6,2,4,5]).*...
                    permute(MR(ind2,ind2,l2+1,1),[1,3,2,4]),[(2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1),2*B,2*B]);
                kron_FMR2 = reshape(permute(Fold(ind1,ind1,l1+1,:,:,ix),[3,1,6,2,4,5]).*...
                    permute(MR(ind2,ind2,l2+1,2),[1,3,2,4]),[(2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1),2*B,2*B]);
                kron_FMR3 = reshape(permute(Fold(ind1,ind1,l1+1,:,:,ix),[3,1,6,2,4,5]).*...
                    permute(MR(ind2,ind2,l2+1,3),[1,3,2,4]),[(2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1),2*B,2*B]);
                
                kron_FMR1_gpu = gpuArray(kron_FMR1);
                temp1(ind,ind,l+1,:,:,ix) = temp1(ind,ind,l+1,:,:,ix) + ...
                    permute(gather(cl*pagefun(@mtimes,CG{l1+1,l2+1}(:,col).',...
                    pagefun(@mtimes,kron_FMR1_gpu,CG{l1+1,l2+1}(:,col)))),[1,2,5,3,4]);
                clear kron_FMR1_gpu;
                
                kron_FMR2_gpu = gpuArray(kron_FMR2);
                temp2(ind,ind,l+1,:,:,ix) = temp2(ind,ind,l+1,:,:,ix) + ...
                    permute(gather(cl*pagefun(@mtimes,CG{l1+1,l2+1}(:,col).',...
                    pagefun(@mtimes,kron_FMR2_gpu,CG{l1+1,l2+1}(:,col)))),[1,2,5,3,4]);
                clear kron_FMR2_gpu;
                
                kron_FMR3_gpu = gpuArray(kron_FMR3);
                temp3(ind,ind,l+1,:,:,ix) = temp3(ind,ind,l+1,:,:,ix) + ...
                    permute(gather(cl*pagefun(@mtimes,CG{l1+1,l2+1}(:,col).',...
                    pagefun(@mtimes,kron_FMR3_gpu,CG{l1+1,l2+1}(:,col)))),[1,2,5,3,4]);
                clear kron_FMR3_gpu;
            end
        end
    end
end

c = 2*pi*1i*[0:B-1,0,-B+1:-1]/L;
dF1 = dF1 - temp1.*permute(c,[1,3,4,2]) - temp2.*permute(c,[1,3,4,5,2]) - ...
    temp3.*permute(c,[1,3,4,5,6,2]);

Fnew = gather(Fold+dF1*dt);

end


function [ c ] = deriv_x( n, B, L )

n = n-1;

if n < B
    c = 2*pi*1i*n/L;
elseif n == B
    c = 0;
elseif n > B && n < 2*B
    c = 2*pi*1i*(n-2*B)/L;
else
    error('n out of range');
end

end

