function [ F, f ] = fftSO3R( func, BR, Bx )

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
            x(:,i,j,k) = [-L/2+L/(2*Bx)*(i-1);-L/2+2/(2*Bx)*(j-1);-L/2+2/(2*Bx)*(k-1)];
        end
    end
end

% weights
w = zeros(1,2*BR);
for j = 1:2*BR
    w(j) = 1/(4*BR^3)*sin(beta(j))*sum(1./(2*(0:BR-1)+1).*sin((2*(0:BR-1)+1)*beta(j)));
end

% Wigner_d
lmax = BR-1;
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*BR);
for j = 1:2*BR
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% function values
if isa(func,'function_handle')
    R_linInd = reshape(R,3,3,(2*BR)^3);
    f = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
    for i = 1:2*Bx
        for j = 1:2*Bx
            parfor k = 1:2*Bx
                f(:,:,:,i,j,k) = reshape(func(R_linInd,x(:,i,j,k)),2*BR,2*BR,2*BR);
            end
        end
    end
else
    f = func;
end

% fft
F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx,'like',f);
for k = 1:2*BR
    F1(:,k,:,:,:,:) = fftn(f(:,k,:,:,:,:));
end
F1 = fftshift(fftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx,'like',F1);
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

