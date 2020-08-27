function [ F, f ] = fftSO3( func, B )

addpath('rotation3d');

% grid over SO(3)
alpha = pi/B*(0:(2*B-1));
beta = pi/(4*B)*(2*(0:(2*B-1))+1);
gamma = pi/B*(0:(2*B-1));

R = zeros(3,3,2*B,2*B,2*B);
ca = cos(alpha);
sa = sin(alpha);
cb = cos(beta);
sb = sin(beta);
cg = cos(gamma);
sg = sin(gamma);

for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            R(:,:,i,j,k) = [1,0,0;0,ca(i),-sa(i);0,sa(i),ca(i)]...
                *[cb(j),0,sb(j);0,1,0;-sb(j),0,cb(j)]...
                *[1,0,0;0,cg(k),-sg(k);0,sg(k),cg(k)];
        end
    end
end

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

for l = 0:B-1
    S1 = zeros(2*B,2*l+1,2*B);
    S2 = zeros(2*B,2*l+1,2*l+1);
    
    for ii = 1:2*B
        for kk = 1:2*B
            temp = fftshift(ifft(f(:,ii,kk)));
            S1(ii,:,kk) = temp(-l+B+1:l+B+1)*(2*B);
        end
    end
    
    for ii = 1:2*B
        for jj = 1:2*l+1
            temp = fftshift(ifft(S1(ii,jj,:)));
            S2(ii,jj,:) = temp(-l+B+1:l+B+1)*(2*B);
        end
    end
    
    for jj = -l:l
        for kk = -l:l
            F(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+l+1,kk+l+1)'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
        end
    end
end

rmpath('rotation3d');

end

