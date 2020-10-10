function [ f, ER, S ] = avR( isreal )
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

% CG coefficient
warning('off','MATLAB:nearlySingularMatrix');

CG = cell(B,B);
for l1 = 0:lmax
    for l2 = 0:lmax
        CG{l1+1,l2+1} = clebsch_gordan(l1,l2);
    end
end

warning('on','MATLAB:nearlySingularMatrix');

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

% cg matrix
if isreal
    T = cell(2*lmax+1,1);
    for l = 0:2*lmax
        T{l+1} = zeros(2*l+1,2*l+1);
        for m = -l:l
            if m<0
                T{l+1}(m+l+1,m+l+1) = 1i;
                T{l+1}(m+l+1,-m+l+1) = -1i*(-1)^(-m);
            elseif m==0
                T{l+1}(l+1,l+1) = sqrt(2);
            else
                T{l+1}(m+l+1,m+l+1) = (-1)^(m);
                T{l+1}(m+l+1,-m+l+1) = 1;
            end
        end
        
        T{l+1} = T{l+1}/sqrt(2);
    end
    
    for l1 = 0:lmax
        for l2 = 0:lmax
            CG{l1+1,l2+1} = kron(conj(T{l1+1}),conj(T{l2+1}))...
                *CG{l1+1,l2+1}*blkdiag(T{abs(l1-l2)+1:l1+l2+1}).';
        end
    end
end

% initial condition
f = zeros(2*B,2*B,2*B,Nt);
s0 = [0,0,0];
cS = pdf_MF_normal(s0);
initR = expRot(pi/2*[0,0,0]);

f(:,:,:,1) = permute(exp(sum(initR'*diag(s0).*R,[1,2]))/cS,[3,4,5,1,2]);

F = zeros(2*lmax+1,2*lmax+1,lmax+1,Nt);

% angular velocity
k_o = -5;
G = diag([1,1,1]);
Rd = expRot([0,0,0]);

omega = zeros(2*B,2*B,2*B,3);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            omega(i,j,k,:) = 0.5*k_o*vee(G*Rd.'*R(:,:,i,j,k)-R(:,:,i,j,k).'*Rd*G,[],false); 
        end
    end
end

% FFT of angular velocity
OMEGA = zeros(2*lmax+1,2*lmax+1,lmax+1,3);

for i = 1:3
    if isreal
        S12 = zeros(2*B,2*B,2*B);
        for kk = 1:2*B
            S12(kk,:,:) = fftshift(fft2(reshape(omega(:,kk,:,i),2*B,2*B)));
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
                        OMEGA(m+lmax+1,n+lmax+1,l+1,i) = sum(w.'.*(...
                            -0.5*(cos_ma_ng-cos_mang).*reshape(Psi(-m+lmax+1,n+lmax+1,l+1,:),2*B,1)...
                            +0.5*(cos_ma_ng+cos_mang).*reshape(Psi(m+lmax+1,n+lmax+1,l+1,:),2*B,1)));
                    else
                        OMEGA(m+lmax+1,n+lmax+1,l+1,i) = sum(w.'.*(...
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
                S1(ii,:,kk) = ifftshift(ifft(omega(:,ii,kk,i)))*(2*B);
            end
        end

        S2 = zeros(2*B,2*B,2*B);
        for ii = 1:2*B
            for jj = 1:2*B
                S2(ii,jj,:) = ifftshift(ifft(S1(ii,jj,:)))*(2*B);
            end
        end

        for l = 0:lmax
            for jj = -l:l
                for kk = -l:l
                    OMEGA(jj+lmax+1,kk+lmax+1,l+1,i) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                        reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
                end
            end
        end
    end
end

% noise
H = diag([pi/4,pi/4,pi/4]);
G = H*H';

%% coefficient matrix
% product of functions
F_tot = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3;
A = zeros(F_tot,F_tot);

for l = 0:lmax
    tic;
    row = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
    ind_u = -l+lmax+1:l+lmax+1;
    
    % drift
    for l2 = 0:lmax
        col = l2*(2*l2-1)*(2*l2+1)/3+1 : (l2+1)*(2*l2+1)*(2*l2+3)/3;
        l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
        
        temp_l2 = zeros((2*l+1)^2,(2*l2+1)^2,3);
        for l1 = l1_all
            cl = (2*l1+1)*(2*l2+1)/(2*l+1);
            col_C = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
            
            temp_l1 = zeros((2*l+1)^2,(2*l2+1)^2,3);
            for j = -l1:l1
                row_C1 = (j+l1)*(2*l2+1)+1 : (j+l1+1)*(2*l2+1);
                for k = -l1:l1
                    row_C2 = (k+l1)*(2*l2+1)+1 : (k+l1+1)*(2*l2+1);
                    CG_kron = kron(CG{l1+1,l2+1}(row_C2,col_C)',...
                            CG{l1+1,l2+1}(row_C1,col_C).');
                    temp_l1 = temp_l1-permute(OMEGA(j+lmax+1,k+lmax+1,l1+1,:),[1,2,4,3]).*CG_kron;
                end
            end
            temp_l2 = temp_l2+cl*temp_l1;
        end
        
        for i = 1:3
            temp_l2(:,:,i) = kron(u(ind_u,ind_u,l+1,i),eye(2*l+1,2*l+1))*temp_l2(:,:,i);
        end
        A(row,col) = sum(temp_l2,3);
    end
    
    % diffusion
    for i = 1:3
        for j = 1:3
            A(row,row) = A(row,row)+0.5*G(i,j)*kron(u(ind_u,ind_u,l+1,i)*...
                u(ind_u,ind_u,l+1,j),eye(2*l+1));
        end
    end
    
    toc;
end

expA = expm(A/sf);

%% propagation
for nt = 1:Nt-1
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
        F1 = zeros(2*B,2*B,2*B);
        for k = 1:2*B
            F1(:,k,:) = fftn(f(:,k,:,nt));
        end
        F1 = fftshift(fftshift(F1,1),3);
        F1 = flip(flip(F1,1),3);

        for l = 0:lmax
            for m = -l:l
                for n = -l:l
                    F(m+lmax+1,n+lmax+1,l+1,nt) = sum(w.*F1(m+lmax+1,:,n+lmax+1).*...
                        permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]));
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
        F1 = zeros(2*B-1,2*B,2*B-1);
        for m = -lmax:lmax
            for n = -lmax:lmax
                lmin = max(abs(m),abs(n));
                F_mn = reshape(F(m+lmax+1,n+lmax+1,lmin+1:lmax+1,nt+1),1,[]);

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

        for k = 1:2*B
            f(:,k,:,nt+1) = ifftn(F1(:,k,:),'symmetric')*(2*B)^2;
        end
    end
end

%% match to matrix Fisher and plot
% expectation
ER = zeros(3,3,Nt);
for nt = 1:Nt
    for i = 1:2*B
        for j = 1:2*B
            for k = 1:2*B
                ER(:,:,nt) = ER(:,:,nt)+R(:,:,i,j,k)*f(i,j,k,nt)*w(j);
            end
        end
    end
end

% match to matrix Fisher
U = zeros(3,3,Nt);
V = zeros(3,3,Nt);
S = zeros(3,Nt);

for nt = 1:Nt
    [U(:,:,nt),D,V(:,:,nt)] = psvd(ER(:,:,nt));
    S(:,nt) = pdf_MF_M2S(diag(D),s0.');
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


