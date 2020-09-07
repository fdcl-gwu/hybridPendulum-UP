function [ FG ] = fftSO3_prod( func1, func2, B )

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
g = zeros(2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(i,j,k) = func1(R(:,:,i,j,k));
            g(i,j,k) = func2(R(:,:,i,j,k));
        end
    end
end

% Wigner_d
lmax = B-1;

d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
for j = 1:2*B
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% fft for f
F = zeros(2*lmax+1,2*lmax+1,lmax+1);

S1 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for kk = 1:2*B
        S1(ii,:,kk) = ifftshift(ifft(f(:,ii,kk)))*(2*B);
    end
end

S2 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for jj = 1:2*B
        S2(ii,jj,:) = ifftshift(ifft(S1(ii,jj,:)))*(2*B);
    end
end

for l = 0:lmax
    for jj = -l:l
        for kk = -l:l
            F(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
        end
    end
end

% fft for g
G = zeros(2*lmax+1,2*lmax+1,lmax+1);

S1 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for kk = 1:2*B
        S1(ii,:,kk) = ifftshift(ifft(g(:,ii,kk)))*(2*B);
    end
end

S2 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for jj = 1:2*B
        S2(ii,jj,:) = ifftshift(ifft(S1(ii,jj,:)))*(2*B);
    end
end

for l = 0:lmax
    for jj = -l:l
        for kk = -l:l
            G(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
        end
    end
end

% CG coefficient
CG = cell(B,B);
for l1 = 0:lmax
    for l2 = 0:lmax
        CG{l1+1,l2+1} = clebsch_gordan(l1,l2);
    end
end

% fft for product
FG = zeros(2*lmax+1,2*lmax+1,lmax+1);

for l1 = 0:lmax
    for l2 = 0:lmax
        ind1 = -l1+lmax+1:l1+lmax+1;
        ind2 = -l2+lmax+1:l2+lmax+1;
        
        A = CG{l1+1,l2+1}.'*kron(F(ind1,ind1,l1+1),G(ind2,ind2,l2+1))*CG{l1+1,l2+1};
        A = A*(2*l1+1)*(2*l2+1);
        
        for l = 0:lmax
            if l>=abs(l1-l2) && l<=l1+l2
                indA = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
                indFG = -l+lmax+1:l+lmax+1;
                FG(indFG,indFG,l+1) = FG(indFG,indFG,l+1)+A(indA,indA)/(2*l+1);
            end
        end
    end
end

end

