function [ f ] = ifftS2( F )

B = size(F,2);
lmax = B-1;

% grid over S2
theta = pi*(2*(0:2*B-1)+1)/(4*B);

% Legendre
P = zeros(2*lmax+1,lmax,2*B);
for l = 0:lmax
    P(lmax+1:l+lmax+1,l+1,:) = permute(legendre(l,cos(theta),'norm'),[1,3,2]);
    P(-l+lmax+1:lmax,l+1,:) = flip(P(lmax+2:l+lmax+1,l+1,:));
    P(lmax+1:l+lmax+1,l+1,:) = (-1).^(0:l).'.*P(lmax+1:l+lmax+1,l+1,:);
    P(:,l+1,:) = P(:,l+1,:)/sqrt(2*pi);
end

% ifft
F1 = zeros(2*B,2*B-1);
for m = -lmax:lmax
    Fm = F(m+lmax+1,abs(m)+1:lmax+1);
    for k = 1:2*B
        Pmk = P(m+lmax+1,abs(m)+1:lmax+1,k);
        F1(k,m+lmax+1) = sum(Fm.*Pmk);
    end
end

F1 = cat(2,zeros(2*B,1),F1);
F1 = ifftshift(F1,2);

f = zeros(2*B,2*B);
for k = 1:2*B
    f(k,:) = ifft(F1(k,:),'symmetric')*(2*B);
end

end

