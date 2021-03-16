function [ stat, MFG ] = pendulum_reduced( use_mex, method, FP, path )

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('use_mex','var') || isempty(use_mex)
    use_mex = false;
end

if use_mex 
    addpath('mex');
end

if ~exist('method','var') || isempty(method)
    method = 'euler';
end

if ~(strcmpi(method,'euler') || strcmpi(method,'midpoint') || strcmpi(method,'RK2') || strcmpi(method,'RK4'))
    error('''method'' must be one of ''euler'',''midpoint'', ''RK2'', or ''RK4''');
end

if ~exist('FP','var') || isempty(FP)
    FP = 64;
end

if FP == 64
    precision = 'double';
elseif FP == 32
    precision = 'single';
else
    error('FP must be 32 or 64');
end

if exist('path','var') && ~isempty(path)
    saveToFile = true;
else
    saveToFile = false;
end

% parameters
J = 0.0152492;
rho = 0.0679878;
m = 1.85480;
g = 9.8;

% scaled parameters
tscale = sqrt(J/(m*g*rho));

% time
sf = 400;
T = 1;
Nt = T*sf+1;

% scaled time
dtt = 1/sf/tscale;

% band limit
BR = 12;
Bx = 10;
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

R = zeros(3,3,2*BR,2*BR,2*BR,precision);
for i = 1:2*BR
    for j = 1:2*BR
        for k = 1:2*BR
            R(:,:,i,j,k) = Ra(:,:,i)*Rb(:,:,j)*Rg(:,:,k);
        end
    end
end

% grid over R^3
L = 1.6*2;
x = zeros(2,2*Bx,2*Bx,precision);
for i = 1:2*Bx
    for j = 1:2*Bx
        x(:,i,j) = [-L/2+L/(2*Bx)*(i-1);-L/2+L/(2*Bx)*(j-1)];
    end
end

% weights
w = zeros(1,2*BR,precision);
for j = 1:2*BR
    w(j) = 1/(4*BR^3)*sin(beta(j))*sum(1./(2*(0:BR-1)+1).*sin((2*(0:BR-1)+1)*beta(j)));
end

% Wigner_d
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*BR,precision);
for j = 1:2*BR
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% derivatives
u = getu(lmax);
if FP == 32
    u = single(u);
end

% CG coefficient
warning('off','MATLAB:nearlySingularMatrix');

CG = cell(BR,BR);
for l1 = 0:lmax
    for l2 = 0:lmax
        CG{l1+1,l2+1} = clebsch_gordan(l1,l2);
        if FP == 32
            CG{l1+1,l2+1} = single(CG{l1+1,l2+1});
        end
    end
end

warning('on','MATLAB:nearlySingularMatrix');

% Fourier transform of x
if use_mex
    X = zeros(2*Bx,2*Bx,2,precision);
else
    X = gpuArray.zeros(2*Bx,2*Bx,2,precision);
end

for i = 1:2
    X(:,:,i) = fftn(x(i,:,:));
end

% Fourier transform of J^-1*M(R)
mR = [permute(R(3,2,:,:,:),[1,3,4,5,2]);permute(-R(3,1,:,:,:),[1,3,4,5,2])];

if use_mex
    MR = zeros(2*lmax+1,2*lmax+1,lmax+1,2,precision);
else
    MR = gpuArray.zeros(2*lmax+1,2*lmax+1,lmax+1,2,precision);
end

for i = 1:2
    F1 = gpuArray.zeros(2*BR,2*BR,2*BR,precision);
    for k = 1:2*BR
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

% damping
b = [0.2;0.2];
bt = b*tscale;
if FP == 32
    bt = single(bt);
end

% noise
H = eye(2)*1;
Ht = H*tscale^(3/2);
G = 0.5*(Ht*Ht.');
if FP == 32
    G = single(G);
end

% initial conditions
S = diag([15,15,15]);
U = expRot([pi*2/3,0,0]);
Miu = [0;0]*tscale;
Sigma = (2*tscale)^2*eye(2);

c = pdf_MF_normal(diag(S));

f = permute(exp(sum(U*S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,4,2,3]).*permute((x-Miu),...
    [4,1,2,3]).*Sigma^-1,[1,2])),[1,2,5,3,4])/c/sqrt((2*pi)^2*det(Sigma));

if saveToFile
    save(strcat(path,'\f1'),'f');
end

% initial Fourier transform
F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,precision);
for k = 1:2*BR
    F1(:,k,:,:,:,:) = fftn(f(:,k,:,:,:,:));
end
F1 = fftshift(fftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,precision);
for l = 0:lmax
    for m = -l:l
        for n = -l:l
            F(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
        end
    end
end

% pre-allocate memory
U = zeros(3,3,Nt);
V = zeros(3,3,Nt);
S = zeros(3,3,Nt);
Miu = zeros(2,Nt);
Sigma = zeros(2,2,Nt);
P = zeros(2,3,Nt);

ER = zeros(3,3,Nt);
Ex = zeros(2,Nt);
Varx = zeros(2,2,Nt);
EvR = zeros(3,Nt);
ExvR = zeros(2,3,Nt);
EvRvR = zeros(3,3,Nt);

[ER(:,:,1),Ex(:,1),Varx(:,:,1),EvR(:,1),ExvR(:,:,1),EvRvR(:,:,1),...
    U(:,:,1),S(:,:,1),V(:,:,1),P(:,:,1),Miu(:,1),Sigma(:,:,1)]...
    = get_stat(double(f),double(R),double(x),double(w));

%% propagation
if FP == 32
    dtt = single(dtt);
    L = single(L);
end

for nt = 1:Nt-1
    tic;
    
    % propagating Fourier coefficients
    if use_mex
        if FP == 32
            F = pendulum_propagate_reduced32(F,X,MR,bt,G,dtt,L,u,CG,method);
        elseif FP == 64
            F = pendulum_propagate_reduced(F,X,MR,bt,G,dtt,L,u,CG,method);
        end
    else
        F = integrate(F,X,MR,bt,G,dtt,L,u,CG,method,precision);
    end
    
    % backward
    F1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,precision);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:);

            for k = 1:2*BR
                d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
                F1(m+lmax+1,k,n+lmax+1,:,:) = sum((2*permute(lmin:lmax,...
                    [1,3,2])+1).*F_mn.*d_jk_betak,3);
            end
        end
    end

    F1 = cat(1,F1,zeros(1,2*BR,2*BR-1,2*Bx,2*Bx,precision));
    F1 = cat(3,F1,zeros(2*BR,2*BR,1,2*Bx,2*Bx,precision));
    F1 = ifftshift(ifftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    for k = 1:2*BR
        f(:,k,:,:,:) = ifftn(F1(:,k,:,:,:),'symmetric')*(2*BR)^2;
    end
    
    [ER(:,:,nt+1),Ex(:,nt+1),Varx(:,:,nt+1),EvR(:,nt+1),ExvR(:,:,nt+1),EvRvR(:,:,nt+1),...
        U(:,:,nt+1),S(:,:,nt+1),V(:,:,nt+1),P(:,:,nt+1),Miu(:,nt+1),Sigma(:,:,nt+1)]...
        = get_stat(double(f),double(R),double(x),double(w));
    
    if saveToFile
        save(strcat(path,'\f',num2str(nt+1)),'f');
    end
    
    toc;
end

stat.ER = ER;
stat.Ex = Ex;
stat.Varx = Varx;
stat.EvR = EvR;
stat.ExvR = ExvR;
stat.EvRvR = EvRvR;

MFG.U = U;
MFG.S = S;
MFG.V = V;
MFG.Miu = Miu;
MFG.Sigma = Sigma;
MFG.P = P;

if saveToFile
    save(strcat(path,'\stat'),'stat');
    save(strcat(path,'\MFG'),'MFG');
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');
if use_mex
    addpath('mex');
end

end


function [ Fnew ] = integrate( Fold, X, MR, b, G, dt, L, u, CG, method, precision )

u = gpuArray(u);
dF1 = derivative(gpuArray(Fold),X,MR,b,G,L,u,CG,precision);

if strcmpi(method,'euler')
    Fnew = Fold+dt*dF1;
    return;
end

% midpoint method, RK2 method
if strcmpi(method,'midpoint') || strcmpi(method,'RK4')
    F2 = Fold+dF1*dt/2;
    dF2 = derivative(gpuArray(F2),X,MR,b,G,L,u,CG,precision);

    if strcmpi(method,'midpoint')
        Fnew = Fold+dt*dF2;
        return;
    end
elseif strcmp(method,'RK2')
    F2 = Fold+dF1*dt;
    dF2 = derivative(gpuArray(F2),X,MR,b,G,L,u,CG,precision);
    
    Fnew = Fold+dt/2*(dF1+dF2);
    return;
end

% RK4 method
F3 = Fold + dF2*dt/2;
dF3 = derivative(gpuArray(F3),X,MR,b,G,L,u,CG,precision);

F4 = Fold + dF3*dt;
dF4 = derivative(gpuArray(F4),X,MR,b,G,L,u,CG,precision);

if strcmpi(method,'RK4')
    Fnew = Fold+1/6*dt*(dF1+2*dF2+2*dF3+dF4);
    return;
end

end


function [ dF ] = derivative( F, X, MR, b, G, L, u, CG, precision )

BR = size(F,3);
Bx = size(F,4)/2;
lmax = BR-1;

dF = gpuArray.zeros(size(F),precision);

% Omega hat
temp1 = gpuArray.zeros(size(dF),precision);
temp2 = gpuArray.zeros(size(dF),precision);
for ix = 1:2*Bx
    for jx = 1:2*Bx
        X_ijk = flip(flip(X,1),2);
        X_ijk = circshift(X_ijk,ix,1);
        X_ijk = circshift(X_ijk,jx,2);
        X_ijk = permute(X_ijk,[4,5,6,1,2,3]);

        temp1(:,:,:,ix,jx) = sum(X_ijk(:,:,:,:,:,1).*F,[4,5])/(2*Bx)^2;
        temp2(:,:,:,ix,jx) = sum(X_ijk(:,:,:,:,:,2).*F,[4,5])/(2*Bx)^2;
    end
end

for l = 0:lmax
    indmn = -l+lmax+1:l+lmax+1;
    dF(indmn,indmn,l+1,:,:) = dF(indmn,indmn,l+1,:,:)-...
        pagefun(@mtimes,temp1(indmn,indmn,l+1,:,:),u(indmn,indmn,l+1,1).')-...
        pagefun(@mtimes,temp2(indmn,indmn,l+1,:,:),u(indmn,indmn,l+1,2).');
end

% b*Omega
bX(:,:,1) = -b(1)*X(:,:,1);
bX(:,:,2) = -b(2)*X(:,:,2);

for ix = 1:2*Bx
    for jx = 1:2*Bx
        bX_ijk = flip(flip(bX,1),2);
        bX_ijk = circshift(bX_ijk,ix,1);
        bX_ijk = circshift(bX_ijk,jx,2);
        bX_ijk = permute(bX_ijk,[4,5,6,1,2,3]);

        temp1 = sum(bX_ijk(:,:,:,:,:,1).*F,[4,5])/(2*Bx)^2;
        temp2 = sum(bX_ijk(:,:,:,:,:,2).*F,[4,5])/(2*Bx)^2;

        dF(:,:,:,ix,jx) = dF(:,:,:,ix,jx)-...
            temp1*deriv_x(ix,Bx,L)-temp2*deriv_x(jx,Bx,L);
    end
end

clear temp1 temp2 temp3

% -mg*cross(rho,R'*e3)
temp1 = gpuArray.zeros(size(F),precision);
temp2 = gpuArray.zeros(size(F),precision);
for l = 0:lmax
    for l2 = 0:lmax
        ind2 = -l2+lmax+1:l2+lmax+1;
        l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
        for l1 = l1_all
            cl = (2*l1+1)*(2*l2+1)/(2*l+1);
            col = l^2-(l1-l2)^2;
            ind1 = -l1+lmax+1:l1+lmax+1;
            
            for m = -l:l
                col_m = col+m+l+1;
                CG_m = reshape(CG{l1+1,l2+1}(:,col_m),2*l2+1,2*l1+1);
                for n = -l:l
                    col_n = col+n+l+1;
                    CG_n = reshape(CG{l1+1,l2+1}(:,col_n),2*l2+1,2*l1+1);
                    
                    temp1(m+lmax+1,n+lmax+1,l+1,:,:) = temp1(m+lmax+1,n+lmax+1,l+1,:,:) + ...
                        cl*sum(F(ind1,ind1,l1+1,:,:).*(CG_m.'*MR(ind2,ind2,l2+1,1)*CG_n),[1,2]);

                    temp2(m+lmax+1,n+lmax+1,l+1,:,:) = temp2(m+lmax+1,n+lmax+1,l+1,:,:) + ...
                        cl*sum(F(ind1,ind1,l1+1,:,:).*(CG_m.'*MR(ind2,ind2,l2+1,2)*CG_n),[1,2]);
                end
            end
        end
    end
end

c = 2*pi*1i*[0:Bx-1,0,-Bx+1:-1]/L;
dF = dF - temp1.*permute(c,[1,3,4,2]) - temp2.*permute(c,[1,3,4,5,2]);

clear temp1 temp2 temp3

% noise
for i = 1:2
    for j = 1:2
        if i==j
            c = 4*pi^2*[0:Bx-1,-Bx:-1].^2/L^2;
            c = -shiftdim(c,-(i+1));
        else
            c = 2*pi*[0:Bx-1,0,-Bx+1:-1]/L;
            c = -shiftdim(c,-(i+1)).*shiftdim(c,-(j+1));
        end
        
        dF = dF + G(i,j)*F.*c;
    end
end

dF = gather(dF);

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


function [ ER, Ex, Varx, EvR, ExvR, EvRvR, U, S, V, P, Miu, Sigma ] = get_stat( f, R, x, w )

Bx = size(x,2)/2;
L = x(1,end,1,1)+x(1,2,1,1)-2*x(1,1,1,1);

fR = sum(f(:,:,:,:,:),[4,5])*(L/2/Bx)^2;
ER = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);

fx = permute(sum(f(:,:,:,:,:).*w,[1,2,3]),[1,4,5,2,3]);
Ex = sum(x.*fx,[2,3])*(L/2/Bx)^2;
Varx = sum(permute(x,[1,4,2,3]).*permute(x,[4,1,2,3]).*...
    permute(fx,[1,4,2,3]),[3,4,5])*(L/2/Bx)^2 - Ex*Ex.';

[U,D,V] = psvd(ER);
try
    s = pdf_MF_M2S(diag(D),[0;0;0]);
    S = diag(s);

    Q = gather(pagefun(@mtimes,U.',pagefun(@mtimes,gpuArray(R),V)));
    vR = permute(cat(1,s(2)*Q(3,2,:,:,:)-s(3)*Q(2,3,:,:,:),...
        s(3)*Q(1,3,:,:,:)-s(1)*Q(3,1,:,:,:),...
        s(1)*Q(2,1,:,:,:)-s(2)*Q(1,2,:,:,:)),[1,3,4,5,2]);

    EvR = sum(vR.*permute(w,[1,3,2]).*permute(fR,[4,1,2,3]),[2,3,4]);
    EvRvR = sum(permute(vR,[1,5,2,3,4]).*permute(vR,[5,1,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(fR,[4,5,1,2,3]),[3,4,5]);
    
    ExvR = sum(permute(vR,[5,1,2,3,4]).*permute(w,[1,3,4,2]).*permute(f,[6,7,1,2,3,4,5]),[3,4,5]);
    ExvR = sum(permute(x,[1,4,5,6,7,2,3]).*ExvR,[5,6,7])*(L/2/Bx)^2;

    covxx = Varx;
    covxvR = ExvR-Ex*EvR.';
    covvRvR = EvRvR-EvR*EvR.';

    P = covxvR*covvRvR^-1;
    Miu = Ex-P*EvR;
    Sigma = covxx-P*covxvR.'+P*(trace(S)*eye(3)-S)*P.';
catch
    S = NaN(3,3);
    EvR = NaN(3,1);
    EvRvR = NaN(3,3);
    ExvR = NaN(3,3);
    P = NaN(3,3);
    Miu = NaN(3,1);
    Sigma = NaN(3,3);
end

end

