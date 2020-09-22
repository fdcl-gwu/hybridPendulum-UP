function [ f ] = ifftSO3( F, isreal )

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

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

% inverse transform
f = zeros(2*B,2*B,2*B);

if isreal
    % W
    W = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
    
    for k = 1:2*B
        for l = 0:lmax
            for m = -l:l
                if m>=0
                    n_all = 0:l;
                else
                    n_all = -l:-1;
                end
                
                for n = n_all
                    if m==0 && n==0
                        W(m+lmax+1,n+lmax+1,l+1,k) = d(lmax+1,lmax+1,l+1,k);
                    elseif m==0 || n==0
                        W(m+lmax+1,n+lmax+1,l+1,k) = (-1)^(m-n)*sqrt(2)...
                            *d(abs(m)+lmax+1,abs(n)+lmax+1,l+1,k);
                    else
                        W(m+lmax+1,n+lmax+1,l+1,k) =...
                            (-1)^(m-n)*d(abs(m)+lmax+1,abs(n)+lmax+1,l+1,k)+...
                            (-1)^m*sign(m)*d(abs(m)+lmax+1,-abs(n)+lmax+1,l+1,k);
                    end
                end
            end
        end
    end
    
    % X
    Xa = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
    Xg = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
    
    for j = 1:2*B
        for l = 0:lmax
            for m = -l:l
                Xa(m+lmax+1,m+lmax+1,l+1,j) = cos(m*alpha(j));
                Xg(m+lmax+1,m+lmax+1,l+1,j) = cos(m*gamma(j));
                if m~=0
                    Xa(m+lmax+1,-m+lmax+1,l+1,j) = -sin(m*alpha(j));
                    Xg(m+lmax+1,-m+lmax+1,l+1,j) = -sin(m*gamma(j));
                end
            end
        end
    end
    
    for j1 = 1:2*B
        for k = 1:2*B
            for j2 = 1:2*B
                for l = 0:lmax
                    ind = -l+lmax+1:l+lmax+1;
                    f(j1,k,j2) = f(j1,k,j2) + (2*l+1)*trace(...
                        F(ind,ind,l+1).'*Xa(ind,ind,l+1,j1)*...
                        W(ind,ind,l+1,k)*Xg(ind,ind,l+1,j2));
                end
            end
        end
    end
else
    % complex harmonic analysis
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

    S1 = zeros(2*B,2*B-1,2*B);
    for i = 1:2*B
        for j = 1:2*B-1
            for k = 1:2*B
                S1(i,j,k) = sum(exp(-1i*(-B+1:B-1)*gamma(k)).*reshape(S2(i,j,:),1,[]));
            end
        end
    end

    for i = 1:2*B
        for j = 1:2*B
            for k = 1:2*B
                f(j,i,k) = sum(exp(-1i*(-B+1:B-1)*alpha(j)).*S1(i,:,k));
            end
        end
    end
end

end

