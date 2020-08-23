function [ d ] = Wigner_d( beta, l, m, n )

if exist('m','var') && exist('n','var')
    d = sqrt(factorial(l+n)*factorial(l-n)/factorial(l+m)/factorial(l-m))...
        *sin(beta/2)^(n-m)*cos(beta/2)^(n+m)*jacobiP(l-n,n-m,n+m,cos(beta))...
        *sqrt((2*l+1)/2);
else
    d = cell(l,1);
    
    for ii = 1:length(beta)
        % only calculate the upper left matrix
        % l=1
        d{1}{1,1}(ii) = sqrt(3/2)*cos(beta(ii)/2)^2;
        d{1}{1,2}(ii) = sqrt(3)*cos(beta(ii)/2)*sin(beta(ii)/2);
        d{1}{1,3}(ii) = sqrt(3/2)*sin(beta(ii)/2)^2;
        d{1}{2,1}(ii) = -sqrt(3)*cos(beta(ii)/2)*sin(beta(ii)/2);
        d{1}{2,2}(ii) = sqrt(3/2)*cos(beta(ii));
        d{1}{3,1}(ii) = sqrt(3/2)*sin(beta(ii)/2)^2;
        d{1} = symm(d{1},1,ii);

        % l=2
        for j = 1:5
            for k = 1:(6-j)
                m = j-3;
                n = k-3;
                d{2}{j,k}(ii) = sqrt(factorial(2+n)*factorial(2-n)/factorial(2+m)/factorial(2-m))...
                    *sin(beta(ii)/2)^(n-m)*cos(beta(ii)/2)^(n+m)*jacobiP(2-n,n-m,n+m,cos(beta(ii)))...
                    *sqrt((2*2+1)/2);
            end
        end
        d{2} = symm(d{2},2,ii);

        % l>=3
        for i = 3:l
            for j = 1:(2*i+1)
                for k = 1:(2*i+2-j)
                    m = j-i-1;
                    n = k-i-1;

                    if j==1
                        d{i}{j,k}(ii) = sqrt(factorial(2*i)/factorial(i+n)/factorial(i-n))...
                            *cos(beta(ii)/2)^(i-n)*sin(beta(ii)/2)^(i+n)...
                            *sqrt((2*i+1)/2);
                    elseif k==1
                        d{i}{j,k}(ii) = sqrt(factorial(2*i)/factorial(i+m)/factorial(i-m))...
                            *cos(beta(ii)/2)^(i-m)*(-sin(beta(ii)/2))^(i+m)...
                            *sqrt((2*i+1)/2);
                    elseif j==2 || k==2
                        d{i}{j,k}(ii) = sqrt(factorial(i+n)*factorial(i-n)/factorial(i+m)/factorial(i-m))...
                            *sin(beta(ii)/2)^(n-m)*cos(beta(ii)/2)^(n+m)*jacobiP(i-n,n-m,n+m,cos(beta(ii)))...
                            *sqrt((2*i+1)/2);
                    else
                        d{i}{j,k}(ii) = sqrt((2*i+1)/(2*i-1))*i*(2*i-1)/sqrt((i^2-m^2)*(i^2-n^2))...
                            *(cos(beta(ii))-m*n/(i-1)/i)*d{i-1}{j-1,k-1}(ii)...
                            - sqrt((2*i+1)/(2*i-3))*sqrt(((i-1)^2-m^2)*((i-1)^2-n^2))...
                            /sqrt((i^2-m^2)*(i^2-n^2))*i/(i-1)*d{i-2}{j-2,k-2}(ii);
                    end
                end
            end

            d{i} = symm(d{i},i,ii);
        end
    end
    
    d = [{{ones(1,length(beta))}};d];
end

end


function [ d ] = symm( d, l, ii )
% supplement lower right part using symmetry

for i = 2:(2*l+1)
    for j = (2*l+3-i):(2*l+1)
        d{i,j}(ii) = d{2*(l+1)-j,2*(l+1)-i}(ii);
    end
end

end

