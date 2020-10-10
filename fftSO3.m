function [ F, f, R ] = fftSO3( func, B, isreal )

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

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
if isreal
    % Psi
    Psi = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
    
    for k = 1:2*B
        for l = 0:lmax
            for m = -l:l
                for n = -l:0
                    if m==0 && n==0
                        Psi(m+lmax+1,n+lmax+1,l+1,k) = d(lmax+1,lmax+1,l+1,k);
                    elseif m==0 || n==0
                        Psi(m+lmax+1,n+lmax+1,l+1,k) = (-1)^(m-n)*sqrt(2)...
                            *d(abs(m)+lmax+1,abs(n)+lmax+1,l+1,k);
                    else
                        Psi(m+lmax+1,n+lmax+1,l+1,k) =...
                            (-1)^(m-n)*d(abs(m)+lmax+1,abs(n)+lmax+1,l+1,k)+...
                            (-1)^m*sign(m)*d(abs(m)+lmax+1,-abs(n)+lmax+1,l+1,k);
                    end
                end
            end
            
            if l>0
                Psi(-l+lmax+1:l+lmax+1,1+lmax+1:l+lmax+1,l+1,k) =...
                    flip(Psi(-l+lmax+1:l+lmax+1,-l+lmax+1:-1+lmax+1,l+1,k),2);
            end
        end
    end
    
    % real harmonic analysis
    F = zeros(2*lmax+1,2*lmax+1,lmax+1);
    
    S12 = zeros(2*B,2*B,2*B);
    for n = 1:2*B
        S12(n,:,:) = fftshift(fft2(reshape(f(:,n,:),2*B,2*B)));
    end
    
    for l = 0:lmax
        for m = -l:l
            indm = m+lmax+2;
            for n = -l:l
                indn = n+lmax+2;
                ind_n = -n+lmax+2;
                sin_mang = -imag(S12(:,indm,indn));
                sin_ma_ng = -imag(S12(:,indm,ind_n));
                cos_mang = real(S12(:,indm,indn));
                cos_ma_ng = real(S12(:,indm,ind_n));
                
                if (m>=0 && n>=0) || (m<0 && n<0)
                    F(m+lmax+1,n+lmax+1,l+1) = sum(w.'.*(...
                        -0.5*(cos_ma_ng-cos_mang).*reshape(Psi(-m+lmax+1,n+lmax+1,l+1,:),2*B,1)...
                        +0.5*(cos_ma_ng+cos_mang).*reshape(Psi(m+lmax+1,n+lmax+1,l+1,:),2*B,1)));
                else
                    F(m+lmax+1,n+lmax+1,l+1) = sum(w.'.*(...
                        -0.5*(sin_mang+sin_ma_ng).*reshape(Psi(-m+lmax+1,n+lmax+1,l+1,:),2*B,1)...
                        +0.5*(sin_mang-sin_ma_ng).*reshape(Psi(m+lmax+1,n+lmax+1,l+1,:),2*B,1)));
                end
            end
        end
    end
else
    % complex harmonic analysis
    F1 = zeros(2*B,2*B,2*B);
    for k = 1:2*B
        F1(:,k,:) = fftn(f(:,k,:));
    end
    F1 = fftshift(fftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);

    F = zeros(2*lmax+1,2*lmax+1,lmax+1);
    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                F(m+lmax+1,n+lmax+1,l+1) = sum(w.*F1(m+lmax+1,:,n+lmax+1).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]));
            end
        end
    end
end

end

