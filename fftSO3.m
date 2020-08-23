function [ F, f ] = fftSO3( func, B )

addpath('rotation3d');

% grid over SO(3)
alpha = pi/B*(0:(2*B-1));
beta = pi/(4*B)*(2*(0:(2*B-1))+1);
gamma = pi/B*(0:(2*B-1));

R = zeros(3,3,2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            R(:,:,i,j,k) = expRot(alpha(i)*[0;0;1])*expRot(beta(j)*[0;1;0])*expRot(gamma(k)*[0;0;1]);
        end
    end
end

w = zeros(1,2*B);
for j = 1:2*B
    w(j) = 2/B*sin(beta(j))*sum(1./(2*(0:B-1)+1).*sin((2*(0:B-1)+1)*beta(j)));
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
d = Wigner_d(beta,B-1);

% fft
F = cell(B,1);

for i = 1:B
    l = i-1;
    S1 = zeros(2*B,2*l+1,2*B);
    S2 = zeros(2*B,2*l+1,2*l+1);
    
    for ii = 1:2*B
        for kk = 1:2*B
            temp = fftshift(ifft(f(:,ii,kk)));
            S1(ii,:,kk) = temp(-l+B+1:l+B+1);
        end
    end
    
    for ii = 1:2*B
        for jj = 1:2*l+1
            temp = fftshift(ifft(S1(ii,jj,:)));
            S2(ii,jj,:) = temp(-l+B+1:l+B+1);
        end
    end
    
    F{i} = zeros(2*l+1,2*l+1);
    for jj = 1:2*l+1
        for kk = 1:2*l+1
            F{i}(jj,kk) = sum(w.*S2(:,jj,kk)'.*d{i}{jj,kk});
        end
    end
end

rmpath('rotation3d');

end

