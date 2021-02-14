function [ stat, MFG ] = pendulum_sep( path )

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

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

R = zeros(3,3,2*BR,2*BR,2*BR);
for i = 1:2*BR
    for j = 1:2*BR
        for k = 1:2*BR
            R(:,:,i,j,k) = Ra(:,:,i)*Rb(:,:,j)*Rg(:,:,k);
        end
    end
end

% grid over R^3
L = 10;
x = zeros(3,2*Bx,2*Bx,2*Bx);
x1 = -L/2 : L/(2*Bx) : L/2-L/(2*Bx);
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            x(:,i,j,k) = [x1(i);x1(j);x1(k)];
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

% CG coefficient
warning('off','MATLAB:nearlySingularMatrix');

CG = cell(BR,1);
for l = 0:lmax
    CG{l+1} = clebsch_gordan(l,l);
end

warning('on','MATLAB:nearlySingularMatrix');

% D(exp(x))
Dx = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            e = rot2eul(expRot(x(:,i,j,k)/sf),'zyz');
            dijk = Wigner_d(e(2),lmax);
            
            for l = 0:lmax
                ind = -l+lmax+1:l+lmax+1;
                Dx(ind,ind,l+1,i,j,k) = dijk(ind,ind,l+1).*...
                    exp(-1i*e(1)*(-l:l)).'.*exp(-1i*e(3)*(-l:l));
            end
        end
    end
end

% Fourier transform of J^-1*cross(Omega,J*Omega)
ojo = -cross(x,permute(pagefun(@mtimes,J,permute(gpuArray(x),...
    [1,5,2,3,4])),[1,3,4,5,2]));
ojo = permute(pagefun(@mtimes,J^-1,permute(ojo,[1,5,2,3,4])),[1,3,4,5,2]);

OJO = zeros(2*Bx,2*Bx,2*Bx,3);
for i = 1:3
    OJO(:,:,:,i) = fftn(ojo(i,:,:,:));
end

clear ojo;

% Fourier transform of J^-1*M(R)
mR = -m*g*cross(repmat(rho,1,2*BR,2*BR,2*BR),permute(R(3,:,:,:,:),[2,3,4,5,1]));
mR = permute(pagefun(@mtimes,J^-1,permute(gpuArray(mR),[1,5,2,3,4])),[1,3,4,5,2]);

MR = zeros(2*lmax+1,2*lmax+1,lmax+1,3);
for i = 1:3
    F1 = gpuArray.zeros(2*BR,2*BR,2*BR);
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
    save(strcat(path,'\f1'),'f','-v7.3');
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
    U(:,:,1),S(:,:,1),V(:,:,1),P(:,:,1),Miu(:,1),Sigma(:,:,1)] = get_stat(f,R,x,w);

%% propagation
for nt = 1:Nt-1
    tic;
    f = integrate(f,d,w,CG,Dx,OJO,MR,L,1/sf);
    
    [ER(:,:,nt+1),Ex(:,nt+1),Varx(:,:,nt+1),EvR(:,nt+1),ExvR(:,:,nt+1),EvRvR(:,:,nt+1),...
        U(:,:,nt+1),S(:,:,nt+1),V(:,:,nt+1),P(:,:,nt+1),Miu(:,nt+1),Sigma(:,:,nt+1)] = get_stat(f,R,x,w);
    
    if saveToFile
        save(strcat(path,'\f',num2str(nt+1)),'f','-v7.3');
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

end


function [ f ] = integrate( f, d, w, CG, Dx, OJO, MR, L, dt )

BR = size(f,1)/2;
Bx = size(f,4)/2;
lmax = BR-1;
const_2BR = 2*BR;
const_2Bx = 2*Bx;

% forward Fourier transform
F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
for i = 1:const_2Bx
    for j = 1:const_2Bx
        for k = 1:const_2Bx
            for l = 1:const_2BR
                F1(:,l,:,i,j,k) = fftn(f(:,l,:,i,j,k));
            end
        end
    end
end
F1 = fftshift(fftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
FR = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
for l = 0:lmax
    for m = -l:l
        for n = -l:l
            FR(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            F(m+lmax+1,n+lmax+1,l+1,:,:,:) = fftn(FR(m+lmax+1,n+lmax+1,l+1,:,:,:));
        end
    end
end

Fx = permute(F(lmax+1,lmax+1,1,:,:,:),[4,5,6,1,2,3]);

fx = sum(f.*w,[1,2,3]);
FR = FR./fx;

% calculate dFx
dFx = zeros(2*Bx,2*Bx,2*Bx);
for ix = 1:2*Bx
    for jx = 1:2*Bx
        for kx = 1:2*Bx
            OJO_ijk = flip(flip(flip(OJO,1),2),3);
            OJO_ijk = circshift(OJO_ijk,ix,1);
            OJO_ijk = circshift(OJO_ijk,jx,2);
            OJO_ijk = circshift(OJO_ijk,kx,3);
            
            dFx(ix,jx,kx) = dFx(ix,jx,kx) - deriv_x(ix,Bx,L)*sum(Fx.*OJO_ijk(:,:,:,1),'all')/(2*Bx)^3 -...
                deriv_x(jx,Bx,L)*sum(Fx.*OJO_ijk(:,:,:,2),'all')/(2*Bx)^3 -...
                deriv_x(kx,Bx,L)*sum(Fx.*OJO_ijk(:,:,:,3),'all')/(2*Bx)^3;
        end
    end
end

c = 2*pi*1i*[0:Bx-1,0,-Bx+1:-1]/L;
for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    c0 = (2*l+1)^2;
    
    CG_mn = reshape(CG{l+1}(:,1),2*l+1,2*l+1);
    temp1 = c0*sum(F(ind,ind,l+1,:,:,:).*(CG_mn.'*MR(ind,ind,l+1,1)*CG_mn),[1,2]);
    temp2 = c0*sum(F(ind,ind,l+1,:,:,:).*(CG_mn.'*MR(ind,ind,l+1,2)*CG_mn),[1,2]);
    temp3 = c0*sum(F(ind,ind,l+1,:,:,:).*(CG_mn.'*MR(ind,ind,l+1,3)*CG_mn),[1,2]);
    
    temp1 = permute(temp1,[4,5,6,1,2,3]);
    temp2 = permute(temp2,[4,5,6,1,2,3]);
    temp3 = permute(temp3,[4,5,6,1,2,3]);
    
    dFx = dFx - temp1.*permute(c,[2,1]) - temp2.*c - temp3.*permute(c,[1,3,2]);
end

% integrate
Fx = Fx + dFx*dt;

% backward Fourier transform for fx
fx = ifftn(Fx,'symmetric');

% calculated rotated FR
FR_r = zeros(size(FR));
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            for l = 0:lmax
                ind = -l+lmax+1:l+lmax+1;
                FR_r(ind,ind,l+1,i,j,k) = FR(ind,ind,l+1,i,j,k)*conj(Dx(ind,ind,l+1,i,j,k));
            end
        end
    end
end

% backward Fourier transfomr for rotated FR
F1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx);
for m = -lmax:lmax
    for n = -lmax:lmax
        lmin = max(abs(m),abs(n));
        F_mn = FR_r(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);
        
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

for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            for l = 1:2*BR
                f(:,l,:,i,j,k) = ifftn(F1(:,l,:,i,j,k),'symmetric')*(2*BR)^2*fx(i,j,k);
            end
        end
    end
end

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

