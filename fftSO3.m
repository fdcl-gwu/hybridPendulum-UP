function [ F, f, R ] = fftSO3( func, B )

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

% weights
w = zeros(1,2*B);
for j = 1:2*B
    w(j) = 1/(4*B^3)*sin(beta(j))*sum(1./(2*(0:B-1)+1).*sin((2*(0:B-1)+1)*beta(j)));
end

% function values
f = zeros(2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(i,j,k) = func(R(:,:,i,j,k));
        end
    end
end

% Wigner_d
lmax = B-1;

d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
for j = 1:2*B
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% fft
F = zeros(2*lmax+1,2*lmax+1,lmax+1);

S1 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for kk = 1:2*B
        S1(ii,:,kk) = fftshift(ifft(f(:,ii,kk)))*(2*B);
    end
end

S2 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for jj = 1:2*B
        S2(ii,jj,:) = fftshift(ifft(S1(ii,jj,:)))*(2*B);
    end
end

for l = 0:lmax
    for jj = -l:l
        for kk = -l:l
            F(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2)'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
        end
    end
end

end

