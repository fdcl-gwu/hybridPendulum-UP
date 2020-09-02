addpath('rotation3d');

% time
sf = 10;
T = 1;

t = 0:1/sf:T;

% band limit
B = 20;

% grid
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

% weight
w = zeros(1,2*B);
for j = 1:2*B
    w(j) = 1/(4*B^3)*sin(beta(j))*sum(1./(2*(0:B-1)+1).*sin((2*(0:B-1)+1)*beta(j)));
end

% wigner d matrix
lmax = B-1;
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*B);
for j = 1:2*B
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% derivatives
u = getu(lmax);

% initial condition
f = zeros(2*B,2*B,2*B,T*sf+1);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(i,j,k) = exp(trace(diag([2,2,2])*R(:,:,i,j,k)))/11.397299813790130;
        end
    end
end

F = zeros(2*B,2*B,2*B,T*sf+1);

% angular velocity
omega = [0,0,2*pi];

% noise
H = diag([0.5,0.5,0.5]);
G = H*H';

%% propagation
for nt = 1:T*sf
    S1 = zeros(2*B,2*B,2*B);
    for ii = 1:2*B
        for kk = 1:2*B
            S1(ii,:,kk) = fftshift(ifft(f(:,ii,kk,nt)))*(2*B);
        end
    end

    S2 = zeros(2*B,2*B,2*B);
    for ii = 1:2*B
        for jj = 1:2*B
            S2(ii,jj,:) = fftshift(ifft(S1(ii,jj,:)))*(2*B);
        end
    end

    for l = 0:lmax
        for jj = -l:l
            for kk = -l:l
                F(jj+lmax+1,kk+lmax+1,l+1,nt) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2)'.*...
                    reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
            end
        end
    end
    
    % propagating Fourier coefficients
    for l = 0:lmax
        ind = -l+lmax+1:l+lmax+1;
        
        A = zeros(2*l+1,2*l+1);
        for i = 1:3
            A = A-omega(i)*u(ind,ind,l+1,i);
            for j = 1:3
                A = A+0.5*G(i,j)*u(ind,ind,l+1,i)*u(ind,ind,l+1,j);
            end
        end
        
        F(ind,ind,l+1,nt+1) = expm(A/sf)*F(ind,ind,l+1,nt);
    end
    
    % inverse transform
    S2 = zeros(2*B,2*B-1,2*B-1);
    for m = -lmax:lmax
        for n = -lmax:lmax
            lmin = max(abs(m),abs(n));
            F_mn = reshape(F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,nt+1),1,[]);

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
                f(j,i,k,nt+1) = sum(exp(-1i*(-B+1:B-1)*alpha(j)).*S1(i,:,k));
            end
        end
    end
end

f = real(f);

%% ER
ER = zeros(3,3,T*sf+1);

for nt = 1:T*sf+1
    for i = 1:2*B
        for j = 1:2*B
            for k = 1:2*B
                ER(:,:,nt) = ER(:,:,nt)+R(:,:,i,j,k)*f(i,j,k,nt)*w(j);
            end
        end
    end
end

rmpath('rotation3d');


