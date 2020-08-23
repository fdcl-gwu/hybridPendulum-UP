function [ f ] = ifftSO3( F )

B = size(F,1);

% grid over SO(3)
alpha = pi/B*(0:(2*B-1));
beta = pi/(4*B)*(2*(0:(2*B-1))+1);
gamma = pi/B*(0:(2*B-1));

% Wigner_d
d = Wigner_d(beta,B-1);

% recover S2
S2 = zeros(2*B,2*B-1,2*B-1);
for j = 1:2*B-1
    for k = 1:2*B-1
        lmin = max(abs(j-B),abs(k-B));
        F_jk = zeros(1,B-lmin);
        for ii = lmin:B-1
            try
                F_jk(ii-lmin+1) = F{ii+1}(j-B+ii+1,k-B+ii+1);
            catch
                pause(1);
            end
        end
        
        for i = 1:2*B
            d_jk_betai = zeros(1,B-lmin);
            for ii = lmin:B-1
                d_jk_betai(ii-lmin+1) = d{ii+1}{j-B+ii+1,k-B+ii+1}(i);
            end
            
            S2(i,j,k) = sum(F_jk.*d_jk_betai);
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

