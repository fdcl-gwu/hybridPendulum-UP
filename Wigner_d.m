function [ d ] = Wigner_d( beta, l, m, n )

cb = cos(beta);
cb2 = cos(beta/2);
sb2 = sin(beta/2);

if exist('m','var') && exist('n','var')
    d = sqrt(factorial(l+n)*factorial(l-n)/factorial(l+m)/factorial(l-m))...
        *sb2^(n-m)*cb2^(n+m)*jacobiP(l-n,n-m,n+m,cb);
else
    lmax = l;
    d = zeros(2*lmax+1,2*lmax+1,lmax+1);
    
    % l=0
    d(lmax+1,lmax+1,1) = 1;
    
    % l=1
    d(lmax,lmax,2) = cb2^2;
    d(lmax+1,lmax,2) = -sqrt(2)*cb2*sb2;
    d(lmax+1,lmax+1,2) = cb;
    d(lmax+2,lmax,2) = sb2^2;
    d = symm(d,1,lmax);

    % l>=2
    for l = 2:lmax
        for n = -l:0
            for m = n:-n
                if n==-l
                    d(m+lmax+1,n+lmax+1,l+1) = sqrt(factorial(2*l)/factorial(l+m)/factorial(l-m))...
                        *cb2^(l-m)*(-sb2)^(l+m);
                elseif n==-l+1
                    d(m+lmax+1,n+lmax+1,l+1) = l*(2*l-1)/sqrt((l^2-m^2)*(l^2-n^2))...
                        *(cb-m*n/(l-1)/l)*d(m+lmax+1,n+lmax+1,l);
                else
                    d(m+lmax+1,n+lmax+1,l+1) = l*(2*l-1)/sqrt((l^2-m^2)*(l^2-n^2))...
                        *(cb-m*n/(l-1)/l)*d(m+lmax+1,n+lmax+1,l)...
                        - sqrt(((l-1)^2-m^2)*((l-1)^2-n^2))/sqrt((l^2-m^2)*(l^2-n^2))...
                        *l/(l-1)*d(m+lmax+1,n+lmax+1,l-1);
                end
            end
        end
        
        d = symm(d,l,lmax);
    end
end

end


function [ d ] = symm( d, l, lmax )

for m = 1:l
    for n = -m+1:m
        d(m+lmax+1,n+lmax+1,l+1) = d(-n+lmax+1,-m+lmax+1,l+1);
    end
end

for m = -l:l-1
    for n = m+1:l
        d(m+lmax+1,n+lmax+1,l+1) = d(n+lmax+1,m+lmax+1,l+1)*(-1)^(m-n);
    end
end

end

