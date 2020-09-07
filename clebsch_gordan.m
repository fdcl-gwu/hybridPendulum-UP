function [ CG ] = clebsch_gordan( l1, l2 )

CG = zeros((2*l1+1)*(2*l2+1),(2*l1+1)*(2*l2+1));

for l = abs(l1-l2):l1+l2
    for m = -l:l
        m1 = (m-l1-l2+abs(l1-l2+m))/2 : (m+l1+l2-abs(l1-l2-m))/2;
        m2 = m-m1;
        n_nz = length(m1);

        % solve C
        A = zeros(n_nz,n_nz);
        for k = 1:n_nz
            A(k,k) = l1*(l1+1)+l2*(l2+1)+2*m1(k)*m2(k)-l*(l+1);

            if k>1
                A(k-1,k) = sqrt(l1*(l1+1)-m1(k-1)*m1(k))*...
                    sqrt(l2*(l2+1)-m2(k-1)*m2(k));
            end
        end
        A = A+A.'-diag(diag(A));

        if n_nz == 1
            C = 1;
        else
            CinCn = linsolve(A(1:n_nz-1,1:n_nz-1),...
                [zeros(n_nz-2,1);-A(n_nz-1,n_nz)]);
            Cn = sqrt(1/(1+sum(CinCn.^2)));

            C = [CinCn*Cn;Cn];
        end

        % put the result into the large CG matrix
        for k = 1:n_nz
            col = l^2-(l2-l1)^2+l+m;
            row = (l1+m1(k))*(2*l2+1)+l2+m2(k);

            CG(row+1,col+1) = C(k);
        end
    end
end

end

