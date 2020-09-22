function [ u ] = getu( l, isreal, m, n, ei )

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

if isreal
    if exist('m','var') && exist('n','var') && exist('ei','var')
        if ei == 1
            if (m>=2 && n==-m+1) || (m<=-2 && n==-m-1)
                u = 0.5*(m+n)*sqrt((l+abs(m))*(l-abs(m)+1));
            elseif (m>=1 && m<=l-1 && n==-m-1) || (m>=-l+1 && m<=-1 && n==-m+1)
                u = -0.5*(m+n)*sqrt((l-abs(m))*(l+abs(m)+1));
            elseif m==-1 && n==0
                u = 1/sqrt(2)*sqrt(l*(l+1));
            elseif m==0 && n==-1
                u = -1/sqrt(2)*sqrt(l*(l+1));
            else
                u = 0;
            end
        elseif ei == 2
            if (m>=2 && n==m-1) || (m<=-2 && n==m+1)
                u = 0.5*sqrt((l+abs(m))*(l-abs(m)+1));
            elseif (m>=1 && m<=l-1 && n==m+1) || (m>=-l+1 && m<=-1 && n==m-1)
                u = -0.5*sqrt((l-abs(m))*(l+abs(m)+1));
            elseif m==1 && n==0
                u = 1/sqrt(2)*sqrt(l*(l+1));
            elseif m==0 && n==1
                u = -1/sqrt(2)*sqrt(l*(l+1));
            else
                u = 0;
            end
        elseif ei == 3
            if m==-n && m~=0
                u = -m;
            else
                u = 0;
            end
        end
    else
        lmax = l;
        u = zeros(2*lmax+1,2*lmax+1,lmax+1,3);
        
        for l = 0:lmax
            % along e1
            for m = -l:l
                if m>=2
                    u(m+lmax+1,-m+1+lmax+1,l+1,1) = 0.5*sqrt((l+abs(m))*(l-abs(m)+1));
                end
                
                if m<=-2
                    u(m+lmax+1,-m-1+lmax+1,l+1,1) = -0.5*sqrt((l+abs(m))*(l-abs(m)+1));
                end
                
                if m>=1 && m<=l-1
                    u(m+lmax+1,-m-1+lmax+1,l+1,1) = 0.5*sqrt((l-abs(m))*(l+abs(m)+1));
                end
                
                if m<=-1 && m>=-l+1
                    u(m+lmax+1,-m+1+lmax+1,l+1,1) = -0.5*sqrt((l-abs(m))*(l+abs(m)+1));
                end
                
                if m==-1
                    u(-1+lmax+1,lmax+1,l+1,1) = -1/sqrt(2)*sqrt(l*(l+1));
                end
                
                if m==0 && l>=1
                    u(lmax+1,-1+lmax+1,l+1,1) = 1/sqrt(2)*sqrt(l*(l+1));
                end
            end

            % along e2
            for m = -l:l
                if m>=2
                    u(m+lmax+1,m-1+lmax+1,l+1,2) = 0.5*sqrt((l+abs(m))*(l-abs(m)+1));
                end
                
                if m<=-2
                    u(m+lmax+1,m+1+lmax+1,l+1,2) = 0.5*sqrt((l+abs(m))*(l-abs(m)+1));
                end
                
                if m>=1 && m<=l-1
                    u(m+lmax+1,m+1+lmax+1,l+1,2) = -0.5*sqrt((l-abs(m))*(l+abs(m)+1));
                end
                
                if m<=-1 && m>=-l+1
                    u(m+lmax+1,m-1+lmax+1,l+1,2) = -0.5*sqrt((l-abs(m))*(l+abs(m)+1));
                end
                
                if m==1
                    u(1+lmax+1,lmax+1,l+1,2) = 1/sqrt(2)*sqrt(l*(l+1));
                end
                
                if m==0
                    u(lmax+1,1+lmax+1,l+1,2) = -1/sqrt(2)*sqrt(l*(l+1));
                end
            end

            % along e3
            for m = -l:l
                if m~=0
                    u(m+lmax+1,-m+lmax+1,l+1,3) = -m;
                end
            end
        end
    end
else
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
            % along e1
            for n = -l:l
                if n > -l
                    u(n+lmax,n+lmax+1,l+1,1) = -0.5i*sqrt((l+n)*(l-n+1));
                end

                if n < l
                    u(n+lmax+2,n+lmax+1,l+1,1) = -0.5i*sqrt((l-n)*(l+n+1));
                end
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

            % along e3
            for m = -l:l
                u(m+lmax+1,m+lmax+1,l+1,3) = -1i*m;
            end
        end
    end
end

end

