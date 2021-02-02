function [ stat, MFG ] = pendulum_log( use_mex, path )

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('use_mex','var') || isempty(use_mex)
    use_mex = false;
end

if exist('path','var') && ~isempty(path)
    saveToFile = true;
else
    saveToFile = false;
end

if use_mex 
    addpath('mex');
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
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            x(:,i,j,k) = [-L/2+L/(2*Bx)*(i-1);-L/2+L/(2*Bx)*(j-1);-L/2+L/(2*Bx)*(k-1)];
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
u = getu(lmax);

% CG coefficient
warning('off','MATLAB:nearlySingularMatrix');

CG = cell(BR,BR);
for l1 = 0:lmax
    for l2 = 0:lmax
        CG{l1+1,l2+1} = clebsch_gordan(l1,l2);
    end
end

warning('on','MATLAB:nearlySingularMatrix');

% Fourier transform of x
if use_mex
    X = zeros(2*Bx,2*Bx,2*Bx,3);
else
    X = gpuArray.zeros(2*Bx,2*Bx,2*Bx,3);
end

for i = 1:3
    X(:,:,:,i) = fftn(x(i,:,:,:));
end

% Fourier transform of Omega cross JOmega
ojo = -cross(x,permute(pagefun(@mtimes,J,permute(gpuArray(x),...
    [1,5,2,3,4])),[1,3,4,5,2]));
ojo = permute(pagefun(@mtimes,J^-1,permute(ojo,[1,5,2,3,4])),[1,3,4,5,2]);

if use_mex
    OJO = zeros(2*Bx,2*Bx,2*Bx,3);
else
    OJO = gpuArray.zeros(2*Bx,2*Bx,2*Bx,3);
end

for i = 1:3
    OJO(:,:,:,i) = fftn(ojo(i,:,:,:));
end

clear ojo;

% Fourier transform of M(R)
mR = -m*g*cross(repmat(rho,1,2*BR,2*BR,2*BR),permute(R(3,:,:,:,:),[2,3,4,5,1]));
mR = permute(pagefun(@mtimes,J^-1,permute(gpuArray(mR),[1,5,2,3,4])),[1,3,4,5,2]);

if use_mex
    MR = zeros(2*lmax+1,2*lmax+1,lmax+1,3);
else
    MR = gpuArray.zeros(2*lmax+1,2*lmax+1,lmax+1,3);
end

for i = 1:3
    G1 = gpuArray.zeros(2*BR,2*BR,2*BR);
    for k = 1:2*BR
        G1(:,k,:) = fftn(mR(i,:,k,:));
    end
    G1 = fftshift(fftshift(G1,1),3);
    G1 = flip(flip(G1,1),3);

    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                MR(m+lmax+1,n+lmax+1,l+1,i) = sum(w.*G1(m+lmax+1,:,n+lmax+1).*...
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
    U(:,:,1),S(:,:,1),V(:,:,1),P(:,:,1),Miu(:,1),Sigma(:,:,1)] = get_stat(f,R,x,w);

%% propagation
g = log(f);
G = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
for nt = 1:Nt-1
    tic;
    
    % forward
    G1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
    for k = 1:2*BR
        G1(:,k,:,:,:,:) = fftn(g(:,k,:,:,:,:));
    end
    G1 = fftshift(fftshift(G1,1),3);
    G1 = flip(flip(G1,1),3);
    
    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                G(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*G1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            end
        end
    end
    
    % propagating Fourier coefficients
    if use_mex
        G = pendulum_propagate(G,X,OJO,MR,1/sf,L,u,CG);
    else
        G = integrate(G,X,OJO,MR,1/sf,L,u,CG);
    end
    
    % backward
    G1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = G(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);

            for k = 1:2*BR
                d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
                G1(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                    [1,3,2])+1).*F_mn.*d_jk_betak,3);
            end
        end
    end

    G1 = cat(1,G1,zeros(1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx));
    G1 = cat(3,G1,zeros(2*BR,2*BR,1,2*Bx,2*Bx,2*Bx));
    G1 = ifftshift(ifftshift(G1,1),3);
    G1 = flip(flip(G1,1),3);
    for k = 1:2*BR
        g(:,k,:,:,:,:) = ifftn(G1(:,k,:,:,:,:),'symmetric')*(2*BR)^2;
    end
    
    % normalize
    f = exp(g);
    f = f/(sum(f.*w,'all')*(L/2/Bx)^3);
    % g = log(f);
    
    [ER(:,:,nt+1),Ex(:,nt+1),Varx(:,:,nt+1),EvR(:,nt+1),ExvR(:,:,nt+1),EvRvR(:,:,nt+1),...
        U(:,:,nt+1),S(:,:,nt+1),V(:,:,nt+1),P(:,:,nt+1),Miu(:,nt+1),Sigma(:,:,nt+1)] = get_stat(f,R,x,w);
    
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


function [ Fnew ] = integrate( Fold, X, OJO, MR, dt, L, u, CG )

BR = size(Fold,3);
Bx = size(Fold,4)/2;
lmax = BR-1;

Fold = gpuArray(Fold);
u = gpuArray(u);

dF1 = gpuArray.zeros(size(Fold));

% Omega hat
temp1 = gpuArray.zeros(size(dF1));
temp2 = gpuArray.zeros(size(dF1));
temp3 = gpuArray.zeros(size(dF1));

for l = 0:lmax
    indmn = -l+lmax+1:l+lmax+1;
    temp1(indmn,indmn,l+1,:,:,:) = pagefun(@mtimes,Fold(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,1).');
    temp2(indmn,indmn,l+1,:,:,:) = pagefun(@mtimes,Fold(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,2).');
    temp3(indmn,indmn,l+1,:,:,:) = pagefun(@mtimes,Fold(indmn,indmn,l+1,:,:,:),u(indmn,indmn,l+1,3).');
end

for ix = 1:2*Bx
    for jx = 1:2*Bx
        for kx = 1:2*Bx
            X_ijk = flip(flip(flip(X,1),2),3);
            X_ijk = circshift(X_ijk,ix,1);
            X_ijk = circshift(X_ijk,jx,2);
            X_ijk = circshift(X_ijk,kx,3);
            X_ijk = permute(X_ijk,[5,6,7,1,2,3,4]);
            
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx) - sum(X_ijk(:,:,:,:,:,:,1).*temp1,[4,5,6])/(2*Bx)^3;
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx) - sum(X_ijk(:,:,:,:,:,:,2).*temp2,[4,5,6])/(2*Bx)^3;
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx) - sum(X_ijk(:,:,:,:,:,:,3).*temp3,[4,5,6])/(2*Bx)^3;
        end
    end
end

clear temp1 temp2 temp3

% cross(Omega,J*Omega)
c = 2*pi*1i*[0:Bx-1,0,-Bx+1:-1]/L;

dF1 = dF1 - permute(OJO(:,:,:,1),[4,5,6,1,2,3]).*permute(c,[1,3,4,2])...
          - permute(OJO(:,:,:,2),[4,5,6,1,2,3]).*permute(c,[1,3,4,5,2])...
          - permute(OJO(:,:,:,3),[4,5,6,1,2,3]).*permute(c,[1,3,4,5,6,2]);
      
temp1 = Fold.*permute(c,[1,3,4,2]);
temp2 = Fold.*permute(c,[1,3,4,5,2]);
temp3 = Fold.*permute(c,[1,3,4,5,6,2]);

for ix = 1:2*Bx
    for jx = 1:2*Bx
        for kx = 1:2*Bx
            OJO_ijk = flip(flip(flip(OJO,1),2),3);
            OJO_ijk = circshift(OJO_ijk,ix,1);
            OJO_ijk = circshift(OJO_ijk,jx,2);
            OJO_ijk = circshift(OJO_ijk,kx,3);
            OJO_ijk = permute(OJO_ijk,[5,6,7,1,2,3,4]);
            
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx) - sum(OJO_ijk(:,:,:,:,:,:,1).*temp1,[4,5,6])/(2*Bx)^3;
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx) - sum(OJO_ijk(:,:,:,:,:,:,2).*temp2,[4,5,6])/(2*Bx)^3;
            dF1(:,:,:,ix,jx,kx) = dF1(:,:,:,ix,jx,kx) - sum(OJO_ijk(:,:,:,:,:,:,3).*temp3,[4,5,6])/(2*Bx)^3;
        end
    end
end

% -mg*cross(rho,R'*e3)
temp1 = gather(temp1);
temp2 = gather(temp2);
temp3 = gather(temp3);

for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    for l2 = 0:lmax
        ind2 = -l2+lmax+1:l2+lmax+1;
        l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
        for l1 = l1_all
            cl = (2*l1+1)*(2*l2+1)/(2*l+1);
            col = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
            ind1 = -l1+lmax+1:l1+lmax+1;
            for ix = 1:2*Bx
                kron_FMR1 = reshape(permute(temp1(ind1,ind1,l1+1,:,:,ix),[3,1,6,2,4,5]).*...
                    permute(MR(ind2,ind2,l2+1,1),[1,3,2,4]),[(2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1),2*Bx,2*Bx]);
                kron_FMR2 = reshape(permute(temp2(ind1,ind1,l1+1,:,:,ix),[3,1,6,2,4,5]).*...
                    permute(MR(ind2,ind2,l2+1,2),[1,3,2,4]),[(2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1),2*Bx,2*Bx]);
                kron_FMR3 = reshape(permute(temp3(ind1,ind1,l1+1,:,:,ix),[3,1,6,2,4,5]).*...
                    permute(MR(ind2,ind2,l2+1,3),[1,3,2,4]),[(2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1),2*Bx,2*Bx]);
                
                kron_FMR1_gpu = gpuArray(kron_FMR1);
                dF1(ind,ind,l+1,:,:,ix) = dF1(ind,ind,l+1,:,:,ix) - ...
                    permute(gather(cl*pagefun(@mtimes,CG{l1+1,l2+1}(:,col).',...
                    pagefun(@mtimes,kron_FMR1_gpu,CG{l1+1,l2+1}(:,col)))),[1,2,5,3,4]);
                clear kron_FMR1_gpu;
                
                kron_FMR2_gpu = gpuArray(kron_FMR2);
                dF1(ind,ind,l+1,:,:,ix) = dF1(ind,ind,l+1,:,:,ix) - ...
                    permute(gather(cl*pagefun(@mtimes,CG{l1+1,l2+1}(:,col).',...
                    pagefun(@mtimes,kron_FMR2_gpu,CG{l1+1,l2+1}(:,col)))),[1,2,5,3,4]);
                clear kron_FMR2_gpu;
                
                kron_FMR3_gpu = gpuArray(kron_FMR3);
                dF1(ind,ind,l+1,:,:,ix) = dF1(ind,ind,l+1,:,:,ix) - ...
                    permute(gather(cl*pagefun(@mtimes,CG{l1+1,l2+1}(:,col).',...
                    pagefun(@mtimes,kron_FMR3_gpu,CG{l1+1,l2+1}(:,col)))),[1,2,5,3,4]);
                clear kron_FMR3_gpu;
            end
        end
    end
end

Fnew = gather(Fold+dF1*dt);

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
    ExvR = sum(permute(vR,[5,1,2,3,4]).*permute(x,[1,5,6,7,8,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(f(:,:,:,:,:,:),[7,8,1,2,3,4,5,6]),[3,4,5,6,7,8])*(L/2/Bx)^3;

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

