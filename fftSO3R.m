function [ F, f ] = fftSO3R( func, B, isreal )

addpath('matrix Fisher');
addpath('rotation3d');

if ~exist('func','var') || isempty(func)
    s = [5,5,5];
    S = diag(s);
    Miu = [0;0;0];
    Sigma = 0.2^2*eye(3);
    P = [0.7,0,0;0,0.7,0;0,0,0.7]*0.2/sqrt(10);
    
    c = pdf_MF_normal(s);
    
    vR = @(R) permute(cat(1,s(2)*R(3,2,:)-s(3)*R(2,3,:),...
        s(3)*R(1,3,:)-s(1)*R(3,1,:),...
        s(1)*R(2,1,:)-s(2)*R(1,2,:)),[1,3,2]);
    SigmaC = Sigma-P*(trace(S)*eye(3)-S)*P.';
    
    func = @(R,x) 1/(c*sqrt((2*pi)^3*det(SigmaC)))*...
        permute(exp(sum(sum(S.'.*R,1),2)),[3,2,1]).*...
        permute(exp(-0.5*sum(sum(permute(x-Miu-P*vR(R),[1,3,2]).*...
            permute(x-Miu-P*vR(R),[1,3,2]).*SigmaC^-1,1),2)),[3,2,1]);
end

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

% function values
R_linInd = reshape(R,3,3,(2*B)^3);
f = zeros(2*B,2*B,2*B,2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        parfor k = 1:2*B
            f(:,:,:,i,j,k) = reshape(func(R_linInd,x(:,i,j,k)),2*B,2*B,2*B);
        end
    end
end

% fft
F1 = zeros(2*B,2*B,2*B,2*B,2*B,2*B);
for k = 1:2*B
    F1(:,k,:,:,:,:) = fftn(f(:,k,:,:,:,:));
end
F1 = fftshift(fftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B,2*B,2*B);
for l = 0:lmax
    for m = -l:l
        for n = -l:l
            F(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
        end
    end
end

rmpath('matrix Fisher');
rmpath('rotation3d');

end

