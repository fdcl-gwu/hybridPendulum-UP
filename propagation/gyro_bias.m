function [ f, ER ] = gyro_bias( isreal )
close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

% time
sf = 10;
T = 1;
Nt = T*sf+1;

% band limit
B = 10;

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
x = zeros(3,2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            x(:,i,j,k) = [-1+2/(2*B)*(i-1);-1+2/(2*B)*(j-1);-1+2/(2*B)*(k-1)];
        end
    end
end

% weights
w = zeros(1,2*B);
for j = 1:2*B
    w(j) = 1/(4*B^3)*sin(beta(j))*sum(1./(2*(0:B-1)+1).*sin((2*(0:B-1)+1)*beta(j)));
end

% Wigner_d
lmax = B-1;
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
for j = 1:2*B
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% derivatives
u = getu(lmax,isreal);

% Fourier transform of x
X = zeros(2*B,2*B,2*B,3);
for i = 1:3
    X(:,:,:,i) = fftn(x(i,:,:,:));
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

R_linInd = reshape(R,3,3,(2*B)^3);
f = zeros(2*B,2*B,2*B,2*B,2*B,2*B,Nt);
for k = 1:2*B
    for j = 1:2*B
        parfor i = 1:2*B
            f(:,:,:,i,j,k,1) = reshape(func(R_linInd,x(:,i,j,k)),2*B,2*B,2*B);
        end
    end
end

% noise
H1 = diag([0.1,0.1,0.1]);
H2 = diag([0.05,0.05,0.05]);

G1 = 0.5*(H1*H1.');
G2 = 0.5*(H2*H2.');

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
    F(:,:,:,:,:,:,nt+1) = integrate(F(:,:,:,:,:,:,nt),X,1/sf,u,G1,G2);
    
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

%% ER, Ex, Varx
ER = zeros(3,3,Nt);
for nt = 1:Nt
    fR = sum(f(:,:,:,:,:,:,nt),[4,5,6])*0.1^3;
    ER(:,:,nt) = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);
end

Ex = zeros(3,Nt);
Varx = zeros(3,3,Nt);
for nt = 1:Nt
    fx = permute(sum(f(:,:,:,:,:,:,nt).*w,[1,2,3]),[1,4,5,6,2,3]);
    Ex(:,nt) = sum(x.*fx,[2,3,4])*0.1^3;
    Varx(:,:,nt) = sum(permute(x,[1,5,2,3,4]).*permute(x,[5,1,2,3,4]).*...
        permute(fx,[1,5,2,3,4]),[3,4,5])*0.1^3 - Ex(:,nt)*Ex(:,nt).';
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');

end


function [ Fnew ] = integrate( Fold, X, dt, u, G1, G2 )

B = size(Fold,3);
lmax = B-1;

Fold = gpuArray(Fold);
X = gpuArray(X);
u = gpuArray(u);

% gyro bias
dF1 = gpuArray.zeros(2*lmax+1,2*lmax+1,lmax+1,2*B,2*B,2*B);
for ix = 1:2*B
    for jx = 1:2*B
        for kx = 1:2*B
            X_ijk = flip(flip(flip(X,1),2),3);
            X_ijk = circshift(X_ijk,ix,1);
            X_ijk = circshift(X_ijk,jx,2);
            X_ijk = circshift(X_ijk,kx,3);
            X_ijk = permute(X_ijk,[5,6,7,1,2,3,4]);
            
            for l = 0:lmax
                indmn = -l+lmax+1:l+lmax+1;
                temp1 = sum(X_ijk(:,:,:,:,:,:,1).*...
                    Fold(indmn,indmn,l+1,:,:,:),[4,5,6])/(2*B)^3;
                temp2 = sum(X_ijk(:,:,:,:,:,:,2).*...
                    Fold(indmn,indmn,l+1,:,:,:),[4,5,6])/(2*B)^3;
                temp3 = sum(X_ijk(:,:,:,:,:,:,3).*...
                    Fold(indmn,indmn,l+1,:,:,:),[4,5,6])/(2*B)^3;
                
                dF1(indmn,indmn,l+1,ix,jx,kx) = -temp1*u(indmn,indmn,l+1,1).'-...
                    temp2*u(indmn,indmn,l+1,2).'-temp3*u(indmn,indmn,l+1,3).';
            end
        end
    end
end

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
            c = pi^2*[0:B-1,-B:-1].^2;
            c = shiftdim(c,-(i+1));
        else
            c = pi*[0:B-1,0,-B+1:-1];
            c = shiftdim(c,-(i+1)).*shiftdim(c,-(j+1));
        end
        
        dF1 = dF1 + G2(i,j)*Fold.*c;
    end
end

Fnew = gather(Fold+dF1*dt);

end

