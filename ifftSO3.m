function [ f ] = ifftSO3( F )

B = size(F,3);
lmax = B-1;

% grid over SO(3)
alpha = pi/B*(0:(2*B-1));
beta = pi/(4*B)*(2*(0:(2*B-1))+1);
gamma = pi/B*(0:(2*B-1));

% Wigner_d
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
for j = 1:2*B
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% recover S2
S2 = zeros(2*B,2*B-1,2*B-1);
for m = -lmax:lmax
    for n = -lmax:lmax
        lmin = max(abs(m),abs(n));
        F_mn = reshape(F(m+lmax+1,n+lmax+1,lmin+1:lmax+1),1,[]);
        
        for k = 1:2*B
            d_jk_betak = reshape(d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k),1,[]);
            S2(k,m+lmax+1,n+lmax+1) = sum((2*(lmin:lmax)+1).*F_mn.*d_jk_betak);
        end
    end
end

% recover S1
S1 = zeros(2*B,2*B-1,2*B);
for i = 1:2*B
    for j = 1:2*B-1
        for k = 1:2*B
            S1(i,j,k) = sum(exp(-1i*(-B+1:B-1)*gamma(k)).*reshape(S2(i,j,:),1,[]));
        end
    end
end

% recover f
f = zeros(2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(j,i,k) = sum(exp(-1i*(-B+1:B-1)*alpha(j)).*S1(i,:,k));
        end
    end
end

end

