function [ f ] = ifftSO3R( F, isreal )

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

% ifft
F2 = zeros(2*B-1,2*B,2*B-1,2*B,2*B,2*B);
for m = -lmax:lmax
    for n = -lmax:lmax
        lmin = max(abs(m),abs(n));
        F_mn = F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);
        
        for k = 1:2*B
            d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
            F2(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                [1,3,2])+1).*F_mn.*d_jk_betak,3);
        end
    end
end

F1 = zeros(2*B,2*B,2*B,2*B,2*B,2*B);
for j1 = 1:2*B
    for j2 = 1:2*B
        for k = 1:2*B
            F1(j1,k,j2,:,:,:) = sum(sum(permute(exp(-1i*(-B+1:B-1)*alpha(j1)),[2,1]).*...
                permute(exp(-1i*(-B+1:B-1)*gamma(j2)),[1,3,2]).*F2(:,k,:,:,:,:),1),3);
        end
    end
end

f = zeros(2*B,2*B,2*B,2*B,2*B,2*B);
for j1 = 1:2*B
    for j2 = 1:2*B
        for k = 1:2*B
            f(j1,k,j2,:,:,:) = ifftn(F1(j1,k,j2,:,:,:));
        end
    end
end

end

