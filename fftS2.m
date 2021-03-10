function [ F, f ] = fftS2( func, B )

lmax = B-1;

% grid over S2
theta = pi*(2*(0:2*B-1)+1)/(4*B);
phi = permute(pi*(0:2*B-1)/B,[1,3,2]);

x = cat(1,cos(phi).*sin(theta),sin(phi).*sin(theta),repmat(cos(theta),[1,1,2*B]));

% weights
w = zeros(2*B,1);
for k = 1:2*B
    j = 0:B-1;
    w(k) = 2*pi/B^2*sin(theta(k))*sum(1./(2*j+1).*sin((2*j+1)*theta(k)));
end

% Legendre
P = zeros(2*lmax+1,lmax,2*B);
for l = 0:lmax
    P(lmax+1:l+lmax+1,l+1,:) = permute(legendre(l,cos(theta),'norm'),[1,3,2]);
    P(-l+lmax+1:lmax,l+1,:) = flip(P(lmax+2:l+lmax+1,l+1,:));
    P(lmax+1:l+lmax+1,l+1,:) = (-1).^(0:l).'.*P(lmax+1:l+lmax+1,l+1,:);
    P(:,l+1,:) = P(:,l+1,:)/sqrt(2*pi);
end

% function values
if isa(func,'function_handle')
    f = zeros(2*B,2*B);
    for k = 1:2*B
        for j = 1:2*B
            f(k,j) = func(x(:,k,j));
        end
    end
else
    f = func;
end

% FFT
F1 = zeros(2*B,2*B);
for k = 1:2*B
    F1(k,:) = fft(f(k,:));
end
F1 = fftshift(F1,2);

F = zeros(2*lmax+1,lmax+1);
for l = 0:lmax
    for m = -l:l
        F(m+lmax+1,l+1) = sum(w.*permute(P(m+lmax+1,l+1,:),[3,2,1]).*F1(:,m+lmax+2));
    end
end

end

