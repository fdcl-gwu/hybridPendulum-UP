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
    S1 = zeros(2*B-1,2*B,2*B-1);
    F1 = zeros(2*B-1,2*B,2*B-1);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = reshape(F(m+lmax+1,n+lmax+1,lmin+1:lmax+1),1,[]);
            
            for k = 1:2*B
                Psi1_betak = reshape(Psi(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k),1,[]);
                Psi2_betak = reshape(Psi(-m+lmax+1,n+lmax+1,lmin+1:lmax+1,k),1,[]);
                
                S1(m+lmax+1,k,n+lmax+1) = sum((2*(lmin:lmax)+1).*F_mn.*Psi1_betak);
                F1(m+lmax+1,k,n+lmax+1) = sum((2*(lmin:lmax)+1).*F_mn.*Psi2_betak);
            end
        end
    end
    
    for i = 1:2*B
        for j = 1:2*B
            smsn_p = -sin(permute(0:lmax,[2,1])*alpha(i)).*sin(permute(0:lmax,[1,3,2])*gamma(j));
            smsn_n = -sin(permute(-lmax:-1,[2,1])*alpha(i)).*sin(permute(-lmax:-1,[1,3,2])*gamma(j));
            cmcn_p = cos(permute(0:lmax,[2,1])*alpha(i)).*cos(permute(0:lmax,[1,3,2])*gamma(j));
            cmcn_n = cos(permute(-lmax:-1,[2,1])*alpha(i)).*cos(permute(-lmax:-1,[1,3,2])*gamma(j));
            
            smcn_p = -sin(permute(0:lmax,[2,1])*alpha(i)).*cos(permute(-lmax:-1,[1,3,2])*gamma(j));
            smcn_n = -sin(permute(-lmax:-1,[2,1])*alpha(i)).*cos(permute(0:lmax,[1,3,2])*gamma(j));
            cmsn_p = cos(permute(0:lmax,[2,1])*alpha(i)).*sin(permute(-lmax:-1,[1,3,2])*gamma(j));
            cmsn_n = cos(permute(-lmax:-1,[2,1])*alpha(i)).*sin(permute(0:lmax,[1,3,2])*gamma(j));
            
            for k = 1:2*B
                f(i,k,j) = sum(sum(F1(:,k,:).*cat(3,[smsn_n;smcn_p],[smcn_n;smsn_p]),1),3)...
                    + sum(sum(S1(:,k,:).*cat(3,[cmcn_n;cmsn_p],[cmsn_n;cmcn_p]),1),3);
            end
        end
    end
else
    % complex harmonic analysis
    F1 = zeros(2*B-1,2*B,2*B-1);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = reshape(F(m+lmax+1,n+lmax+1,lmin+1:lmax+1),1,[]);

            for k = 1:2*B
                d_jk_betak = reshape(d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k),1,[]);
                F1(m+lmax+1,k,n+lmax+1) = sum((2*(lmin:lmax)+1).*F_mn.*d_jk_betak);
            end
        end
    end
    
    F1 = cat(1,F1,zeros(1,2*B,2*B-1));
    F1 = cat(3,F1,zeros(2*B,2*B,1));
    F1 = ifftshift(ifftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);

    f = zeros(2*B,2*B,2*B);
    for k = 1:2*B
        f(:,k,:) = ifftn(F1(:,k,:))*(2*B)^2;
    end
end

end

