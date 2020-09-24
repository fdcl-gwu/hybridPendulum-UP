function [ f ] = gyro_nobias( isreal )
close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('isreal','var') || isempty(isreal)
    isreal = false;
end

% time
sf = 10;
T = 1;

Nt = T*sf+1;

% band limit
B = 10;

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
u = getu(lmax,isreal);

% Psi
if isreal
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
end

% initial condition
s0 = [10,10,10];
f = zeros(2*B,2*B,2*B,Nt);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(i,j,k) = exp(trace(diag(s0)*R(:,:,i,j,k)))/...
                pdf_MF_normal(s0);
        end
    end
end

F = zeros(2*lmax+1,2*lmax+1,lmax+1,Nt);

% angular velocity
omega = [0,0,2*pi];

% noise
H = diag([0.25,0.25,0.25]);
G = H*H';

%% coefficient matrix
F_tot = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3;
A = zeros(F_tot,F_tot);
for l = 0:lmax
    for m = -l:l
        for n = -l:l
            row = l*(2*l-1)*(2*l+1)/3 + (n+l)*(2*l+1) + m+l+1;
            col = l*(2*l-1)*(2*l+1)/3 + (0:2*l)*(2*l+1) + m+l+1;
            
            ind = -l+lmax+1:l+lmax+1;
        
            Al = zeros(2*l+1,2*l+1);
            for i = 1:3
                Al = Al-omega(i)*u(ind,ind,l+1,i);
                for j = 1:3
                    Al = Al+0.5*G(i,j)*u(ind,ind,l+1,i)*u(ind,ind,l+1,j);
                end
            end
            
            A(row,col) = Al(n+l+1,:);
        end
    end
end

expA = expm(A/sf);

%% propagation
for nt = 1:T*sf
    if isreal
        S12 = zeros(2*B,2*B,2*B);
        for kk = 1:2*B
            S12(kk,:,:) = fftshift(fft2(reshape(f(:,kk,:,nt),2*B,2*B)));
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
                        F(m+lmax+1,n+lmax+1,l+1,nt) = sum(w.'.*(...
                            -0.5*(cos_ma_ng-cos_mang).*reshape(Psi(-m+lmax+1,n+lmax+1,l+1,:),2*B,1)...
                            +0.5*(cos_ma_ng+cos_mang).*reshape(Psi(m+lmax+1,n+lmax+1,l+1,:),2*B,1)));
                    else
                        F(m+lmax+1,n+lmax+1,l+1,nt) = sum(w.'.*(...
                            -0.5*(sin_mang+sin_ma_ng).*reshape(Psi(-m+lmax+1,n+lmax+1,l+1,:),2*B,1)...
                            +0.5*(sin_mang-sin_ma_ng).*reshape(Psi(m+lmax+1,n+lmax+1,l+1,:),2*B,1)));
                    end
                end
            end
        end
    else
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
                    F(jj+lmax+1,kk+lmax+1,l+1,nt) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                        reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
                end
            end
        end
    end
    
    % propagating Fourier coefficients
    F_col = zeros(F_tot,1);
    
    for l = 0:lmax
        ind_col = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
        ind = -l+lmax+1 : l+lmax+1;
        F_col(ind_col) = reshape(F(ind,ind,l+1,nt),[],1);
    end
    
    F_col = expA*F_col;
    
    for l = 0:lmax
        ind_col = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
        ind = -l+lmax+1 : l+lmax+1;
        F(ind,ind,l+1,nt+1) = reshape(F_col(ind_col),2*l+1,2*l+1);
    end
    
    % inverse transform
    if isreal
        S1 = zeros(2*B-1,2*B,2*B-1);
        S2 = zeros(2*B-1,2*B,2*B-1);
        for m = -lmax:lmax
            for n = -lmax:lmax
                lmin = max(abs(m),abs(n));
                F_mn = reshape(F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,nt+1),1,[]);

                for k = 1:2*B
                    Psi1_betak = reshape(Psi(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k),1,[]);
                    Psi2_betak = reshape(Psi(-m+lmax+1,n+lmax+1,lmin+1:lmax+1,k),1,[]);

                    S1(m+lmax+1,k,n+lmax+1) = sum((2*(lmin:lmax)+1).*F_mn.*Psi1_betak);
                    S2(m+lmax+1,k,n+lmax+1) = sum((2*(lmin:lmax)+1).*F_mn.*Psi2_betak);
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
                    f(i,k,j,nt+1) = sum(sum(S2(:,k,:).*cat(3,[smsn_n;smcn_p],[smcn_n;smsn_p]),1),3)...
                        + sum(sum(S1(:,k,:).*cat(3,[cmcn_n;cmsn_p],[cmsn_n;cmcn_p]),1),3);
                end
            end
        end
    else
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
        
        f(:,:,:,nt+1) = real(f(:,:,:,nt+1));
    end
end

%% match to matrix Fisher and plot
% expectation
ER = zeros(3,3,Nt);
for nt = 1:Nt
    ER(:,:,nt) = sum(sum(sum(R.*permute(f(:,:,:,nt),[4,5,1,2,3]).*...
        permute(w,[1,4,3,2,5]),5),4),3);
end

% match to matrix Fisher
U = zeros(3,3,Nt);
V = zeros(3,3,Nt);
S = zeros(3,Nt);

for nt = 1:Nt
    [U(:,:,nt),D,V(:,:,nt)] = psvd(ER(:,:,nt));
    if nt==1
        S(:,nt) = pdf_MF_M2S(diag(D),s0.');
    else
        S(:,nt) = pdf_MF_M2S(diag(D),S(:,nt-1));
    end
end

% spherical grid
Nt1 = 100;
Nt2 = 100;
theta1 = linspace(-pi,pi,Nt2);
theta2 = linspace(0,pi,Nt1);
s1 = cos(theta1)'.*sin(theta2);
s2 = sin(theta1)'.*sin(theta2);
s3 = repmat(cos(theta2),Nt1,1);

% color map
color = zeros(100,100,3,Nt);
for nt = 1:Nt
    cF = pdf_MF_normal(S(:,nt));
    F = U(:,:,nt)*diag(S(:,nt))*V(:,:,nt).';
    
    for i = 1:3
        j = mod(i,3)+1;
        k = mod(i+1,3)+1;
        
        for m = 1:100
            for n = 1:100
                r = [s1(m,n);s2(m,n);s3(m,n)];
                
                s_p = eig(F(:,[j,k]).'*(eye(3)-r*r.')*F(:,[j,k]));
                s_p1 = sqrt(s_p(1));
                s_p2 = sqrt(s_p(2));
                
                color(m,n,i,nt) = exp(F(:,i).'*r)/cF*besseli(0,...
                    s_p1+sign(r.'*cross(F(:,j),F(:,k)))*s_p2);
            end
        end
    end
end

color = real(color);

% plot
for nt = 1:Nt
    figure(nt);
    surf(s1,s2,s3,sum(color(:,:,:,nt),3),'LineStyle','none');
    
    axis equal;
    view([1,1,1]);
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');

end


