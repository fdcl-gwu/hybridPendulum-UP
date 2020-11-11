function [ f ] = ifftSO3R( F, isreal )

BR = size(F,3);
Bx = size(F,4)/2;
lmax = BR-1;

% grid over SO(3)
beta = pi/(4*BR)*(2*(0:(2*BR-1))+1);

% Wigner_d
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*BR);
for j = 1:2*BR
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% ifft
F1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx);
for m = -lmax:lmax
    for n = -lmax:lmax
        lmin = max(abs(m),abs(n));
        F_mn = F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);
        
        for k = 1:2*BR
            d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
            F1(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                [1,3,2])+1).*F_mn.*d_jk_betak,3);
        end
    end
end

F1 = cat(1,F1,zeros(1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx));
F1 = cat(3,F1,zeros(2*BR,2*BR,1,2*Bx,2*Bx,2*Bx));
F1 = ifftshift(ifftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

f = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
for k = 1:2*BR
    f(:,k,:,:,:,:) = ifftn(F1(:,k,:,:,:,:))*(2*BR)^2;
end

end

