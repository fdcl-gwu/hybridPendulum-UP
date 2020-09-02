function [ u ] = getu( l, m, n, ei )

if exist('m','var') && exist('n','var') && exist('ei','var')
    if ei == 1
        if m-1==n
            u = -0.5*1i*sqrt((l-n)*(l+n+1));
        elseif m+1==n
            u = 0.5*1i*sqrt((1+n)*(l-n+1));
        else
            u = 0;
        end
    elseif ei == 2
        if m-1==n
            u = -0.5*sqrt((l-n)*(l+n+1));
        elseif m+1==n
            u = 0.5*sqrt((1+n)*(l-n+1));
        else
            u = 0;
        end
    elseif ei == 3
        if m==n
            u = -1i*m;
        else
            u = 0;
        end
    end
else
    lmax = l;
    u = zeros(2*lmax+1,2*lmax+1,lmax+1,3);
    
    for l = 0:lmax
        % along e3
        for m = -l:l
            u(m+lmax+1,m+lmax+1,l+1,3) = -1i*m;
        end
        
        % along e2
        for n = -l:l
            if n > -l
                u(n+lmax,n+lmax+1,l+1,2) = 0.5*sqrt((l+n)*(l-n+1));
            end
            
            if n < l
                u(n+lmax+2,n+lmax+1,l+1,2) = -0.5*sqrt((l-n)*(l+n+1));
            end
        end
        
        % along e1
        for n = -l:l
            if n > -l
                u(n+lmax,n+lmax+1,l+1,1) = -1i*u(n+lmax,n+lmax+1,l+1,2);
            end
            
            if n < l
                u(n+lmax+2,n+lmax+1,l+1,1) = 1i*u(n+lmax+2,n+lmax+1,l+1,2);
            end
        end
    end
end

end

