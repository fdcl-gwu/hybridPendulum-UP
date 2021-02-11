function [ stat, MFG ] = pendulum( use_mex, method, FP, path )

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

% time
sf = 100;
T = 1;
Nt = T*sf+1;

% parameters
Jd = diag([1,2,3]);
J = trace(Jd)*eye(3)-Jd;

rho = [0;0;1];
m = 10;
g = 3;

% band limit
BR = 10;
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
L = 10;
x = zeros(3,2*Bx,2*Bx,2*Bx,precision);
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            x(:,i,j,k) = [-L/2+L/(2*Bx)*(i-1);-L/2+L/(2*Bx)*(j-1);-L/2+L/(2*Bx)*(k-1)];
        end
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
    X = zeros(2*Bx,2*Bx,2*Bx,3,precision);
else
    X = gpuArray.zeros(2*Bx,2*Bx,2*Bx,3,precision);
end

for i = 1:3
    X(:,:,:,i) = fftn(x(i,:,:,:));
end

% Fourier transform of J^-1*cross(Omega,J*Omega)
ojo = -cross(x,permute(pagefun(@mtimes,J,permute(gpuArray(x),...
    [1,5,2,3,4])),[1,3,4,5,2]));
ojo = permute(pagefun(@mtimes,J^-1,permute(ojo,[1,5,2,3,4])),[1,3,4,5,2]);

if use_mex
    OJO = zeros(2*Bx,2*Bx,2*Bx,3,precision);
else
    OJO = gpuArray.zeros(2*Bx,2*Bx,2*Bx,3,precision);
end

for i = 1:3
    OJO(:,:,:,i) = fftn(ojo(i,:,:,:));
end

if isreal(OJO)
    OJO = complex(OJO,OJO);
end

clear ojo;

% Fourier transform of J^-1*M(R)
mR = -m*g*cross(repmat(rho,1,2*BR,2*BR,2*BR),permute(R(3,:,:,:,:),[2,3,4,5,1]));
mR = permute(pagefun(@mtimes,J^-1,permute(gpuArray(mR),[1,5,2,3,4])),[1,3,4,5,2]);

if use_mex
    MR = zeros(2*lmax+1,2*lmax+1,lmax+1,3,precision);
else
    MR = gpuArray.zeros(2*lmax+1,2*lmax+1,lmax+1,3,precision);
end

for i = 1:3
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

% initial conditions
S = diag([10,10,10]);
U = expRot([pi*2/3,0,0]);
Miu = [0;0;0];
Sigma = 1^2*eye(3);

c = pdf_MF_normal(diag(S));

f = permute(exp(sum(U*S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,5,2,3,4]).*permute((x-Miu),...
    [5,1,2,3,4]).*Sigma^-1,[1,2])),[1,2,6,3,4,5])/c/sqrt((2*pi)^3*det(Sigma));

if saveToFile
    save(strcat(path,'\f1'),'f');
end

% pre-allocate memory
U = zeros(3,3,Nt);
V = zeros(3,3,Nt);
S = zeros(3,3,Nt);
Miu = zeros(3,Nt);
Sigma = zeros(3,3,Nt);
P = zeros(3,3,Nt);

ER = zeros(3,3,Nt);
Ex = zeros(3,Nt);
Varx = zeros(3,3,Nt);
EvR = zeros(3,Nt);
ExvR = zeros(3,3,Nt);
EvRvR = zeros(3,3,Nt);

[ER(:,:,1),Ex(:,1),Varx(:,:,1),EvR(:,1),ExvR(:,:,1),EvRvR(:,:,1),...
    U(:,:,1),S(:,:,1),V(:,:,1),P(:,:,1),Miu(:,1),Sigma(:,:,1)]...
    = get_stat(double(f),double(R),double(x),double(w));

%% propagation
if FP == 32
    sf = single(sf);
    L = single(L);
end

F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx,precision);
for nt = 1:Nt-1
    tic;
    
    % forward
    F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx,precision);
    for k = 1:2*BR
        F1(:,k,:,:,:,:) = fftn(f(:,k,:,:,:,:));
    end
    F1 = fftshift(fftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    
    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                F(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            end
        end
    end
    
    % propagating Fourier coefficients
    if use_mex
        if FP == 32
            F = pendulum_propagate32(F,X,OJO,MR,1/sf,L,u,CG,method);
        elseif FP == 64
            F = pendulum_propagate(F,X,OJO,MR,1/sf,L,u,CG,method);
        end
    else
        F = integrate(F,X,OJO,MR,1/sf,L,u,CG,method,precision);
    end
    
    % backward
    F1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx,precision);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);

            for k = 1:2*BR
                d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
                F1(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                    [1,3,2])+1).*F_mn.*d_jk_betak,3);
            end
        end
    end

    F1 = cat(1,F1,zeros(1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx,precision));
    F1 = cat(3,F1,zeros(2*BR,2*BR,1,2*Bx,2*Bx,2*Bx,precision));
    F1 = ifftshift(ifftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    for k = 1:2*BR
        f(:,k,:,:,:,:) = ifftn(F1(:,k,:,:,:,:),'symmetric')*(2*BR)^2;
    end
    
    fmin = min(f);
    if fmin<0
        f(f<-1.01*fmin) = 0;
    end
    f = f/(sum(f.*w,'all')*(L/2/Bx)^3);
    
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


function [ Fnew ] = integrate( Fold, X, OJO, MR, dt, L, u, CG, method, precision )

u = gpuArray(u);
dF1 = derivative(gpuArray(Fold),X,OJO,MR,L,u,CG,precision);

if strcmpi(method,'euler')
    Fnew = Fold+dt*dF1;
    return;
end

% midpoint method, RK2 method
if strcmpi(method,'midpoint') || strcmpi(method,'RK4')
    F2 = Fold+dF1*dt/2;
    dF2 = derivative(gpuArray(F2),X,OJO,MR,L,u,CG,precision);

    if strcmpi(method,'midpoint')
        Fnew = Fold+dt*dF2;
        return;
    end
elseif strcmp(method,'RK2')
    F2 = Fold+dF1*dt;
    dF2 = derivative(gpuArray(F2),X,OJO,MR,L,u,CG,precision);
    
    Fnew = Fold+dt/2*(dF1+dF2);
    return;
end

% RK4 method
F3 = Fold + dF2*dt/2;
dF3 = derivative(gpuArray(F3),X,OJO,MR,L,u,CG,precision);

F4 = Fold + dF3*dt;
dF4 = derivative(gpuArray(F4),X,OJO,MR,L,u,CG,precision);

if strcmpi(method,'RK4')
    Fnew = Fold+1/6*dt*(dF1+2*dF2+2*dF3+dF4);
    return;
end

end


function [ dF ] = derivative( F, X, OJO, MR, L, u, CG, precision )

BR = size(F,3);
Bx = size(F,4)/2;
lmax = BR-1;

dF = gpuArray.zeros(size(F),precision);

% Omega hat
temp1 = gpuArray.zeros(size(dF),precision);
temp2 = gpuArray.zeros(size(dF),precision);
temp3 = gpuArray.zeros(size(dF),precision);
for ix = 1:2*Bx
    for jx = 1:2*Bx
        for kx = 1:2*Bx
            X_ijk = flip(flip(flip(X,1),2),3);
            X_ijk = circshift(X_ijk,ix,1);
            X_ijk = circshift(X_ijk,jx,2);
            X_ijk = circshift(X_ijk,kx,3);
            X_ijk = permute(X_ijk,[5,6,7,1,2,3,4]);
            
            temp1(:,:,:,ix,jx,kx) = sum(X_ijk(:,:,:,:,:,:,1).*F,[4,5,6])/(2*Bx)^3;
            temp2(:,:,:,ix,jx,kx) = sum(X_ijk(:,:,:,:,:,:,2).*F,[4,5,6])/(2*Bx)^3;
            temp3(:,:,:,ix,jx,kx) = sum(X_ijk(:,:,:,:,:,:,3).*F,[4,5,6])/(2*Bx)^3;
        end
    end
end

for l = 0:lmax
    indmn = -l+lmax+1:l+lmax+1;
    dF(indmn,indmn,l+1,:,:,:) = dF(indmn,indmn,l+1,:,:,:)-...
        pagefun(@mtimes,temp1(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,1).')-...
        pagefun(@mtimes,temp2(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,2).')-...
        pagefun(@mtimes,temp3(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,3).');
end

clear temp1 temp2 temp3

% cross(Omega,J*Omega)
for ix = 1:2*Bx
    for jx = 1:2*Bx
        for kx = 1:2*Bx
            OJO_ijk = flip(flip(flip(OJO,1),2),3);
            OJO_ijk = circshift(OJO_ijk,ix,1);
            OJO_ijk = circshift(OJO_ijk,jx,2);
            OJO_ijk = circshift(OJO_ijk,kx,3);
            OJO_ijk = permute(OJO_ijk,[5,6,7,1,2,3,4]);
            
            temp1 = sum(OJO_ijk(:,:,:,:,:,:,1).*F,[4,5,6])/(2*Bx)^3;
            temp2 = sum(OJO_ijk(:,:,:,:,:,:,2).*F,[4,5,6])/(2*Bx)^3;
            temp3 = sum(OJO_ijk(:,:,:,:,:,:,3).*F,[4,5,6])/(2*Bx)^3;
            
            dF(:,:,:,ix,jx,kx) = dF(:,:,:,ix,jx,kx)-...
                temp1*deriv_x(ix,Bx,L)-temp2*deriv_x(jx,Bx,L)-temp3*deriv_x(kx,Bx,L);
        end
    end
end

clear temp1 temp2 temp3

% -mg*cross(rho,R'*e3)
temp1 = gpuArray.zeros(size(F),precision);
temp2 = gpuArray.zeros(size(F),precision);
temp3 = gpuArray.zeros(size(F),precision);
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
                    
                    temp1(m+lmax+1,n+lmax+1,l+1,:,:,:) = temp1(m+lmax+1,n+lmax+1,l+1,:,:,:) + ...
                        cl*sum(F(ind1,ind1,l1+1,:,:,:).*(CG_m.'*MR(ind2,ind2,l2+1,1)*CG_n),[1,2]);

                    temp2(m+lmax+1,n+lmax+1,l+1,:,:,:) = temp2(m+lmax+1,n+lmax+1,l+1,:,:,:) + ...
                        cl*sum(F(ind1,ind1,l1+1,:,:,:).*(CG_m.'*MR(ind2,ind2,l2+1,2)*CG_n),[1,2]);
                    
                    temp3(m+lmax+1,n+lmax+1,l+1,:,:,:) = temp3(m+lmax+1,n+lmax+1,l+1,:,:,:) + ...
                        cl*sum(F(ind1,ind1,l1+1,:,:,:).*(CG_m.'*MR(ind2,ind2,l2+1,3)*CG_n),[1,2]);
                end
            end
        end
    end
end

c = 2*pi*1i*[0:Bx-1,0,-Bx+1:-1]/L;
dF = dF - temp1.*permute(c,[1,3,4,2]) - temp2.*permute(c,[1,3,4,5,2]) - ...
    temp3.*permute(c,[1,3,4,5,6,2]);

clear temp1 temp2 temp3

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

fR = sum(f(:,:,:,:,:,:),[4,5,6])*(L/2/Bx)^3;
ER = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);

fx = permute(sum(f(:,:,:,:,:,:).*w,[1,2,3]),[1,4,5,6,2,3]);
Ex = sum(x.*fx,[2,3,4])*(L/2/Bx)^3;
Varx = sum(permute(x,[1,5,2,3,4]).*permute(x,[5,1,2,3,4]).*...
    permute(fx,[1,5,2,3,4]),[3,4,5])*(L/2/Bx)^3 - Ex*Ex.';

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
    
    ExvR = sum(permute(vR,[5,1,2,3,4]).*permute(w,[1,3,4,2]).*permute(f,[7,8,1,2,3,4,5,6]),[3,4,5]);
    ExvR = sum(permute(x,[1,5,6,7,8,2,3,4]).*ExvR,[6,7,8])*(L/2/Bx)^3;

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

