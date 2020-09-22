clear;
close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

% band limit
B = 15;

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

% CG coefficient
warning('off','MATLAB:nearlySingularMatrix');

CG = cell(B,B);
for l1 = 0:lmax
    for l2 = 0:lmax
        CG{l1+1,l2+1} = clebsch_gordan(l1,l2);
    end
end

warning('on','MATLAB:nearlySingularMatrix');

% f
f = zeros(2*B,2*B,2*B);
s0 = [2,2,2];
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(i,j,k) = exp(trace(diag(s0)*R(:,:,i,j,k)))/pdf_MF_normal(s0);
        end
    end
end

% angular velocity
k_o = -5;

omega = zeros(2*B,2*B,2*B,3);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            omega(i,j,k,:) = -k_o*vee(R(:,:,i,j,k)-R(:,:,i,j,k).');
        end
    end
end

% omega1_f
omega_f = omega.*f;

% domega1_f
domega_f = zeros(2*B,2*B,2*B,3);
hate1 = hat([1,0,0]);
hate2 = hat([0,1,0]);
hate3 = hat([0,0,1]);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            domega_f(i,j,k,:) = f(i,j,k)*[-k_o*(R(2,2,i,j,k)+R(3,3,i,j,k))+...
                omega(i,j,k,1)*trace(diag(s0)*R(:,:,i,j,k)*hate1),...
                -k_o*(R(1,1,i,j,k)+R(3,3,i,j,k))+...
                omega(i,j,k,2)*trace(diag(s0)*R(:,:,i,j,k)*hate2),...
                -k_o*(R(1,1,i,j,k)+R(2,2,i,j,k))+...
                omega(i,j,k,3)*trace(diag(s0)*R(:,:,i,j,k)*hate3)];
        end
    end
end

%% DFT of domega1_f directly
DOMEGA_F_direct = zeros(2*lmax+1,2*lmax+1,lmax+1,3);

for i = 1:3
    S1 = zeros(2*B,2*B,2*B);
    for ii = 1:2*B
        for kk = 1:2*B
            S1(ii,:,kk) = ifftshift(ifft(domega_f(:,ii,kk,i)))*(2*B);
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
                DOMEGA_F_direct(jj+lmax+1,kk+lmax+1,l+1,i) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                    reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
            end
        end
    end
end

%% DFT of domega1_f using product rule
% DFT of f
F = zeros(2*lmax+1,2*lmax+1,l+1);

S1 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for kk = 1:2*B
        S1(ii,:,kk) = ifftshift(ifft(f(:,ii,kk)))*(2*B);
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
            F(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
        end
    end
end

% DFT of omega1
OMEGA = zeros(2*lmax+1,2*lmax+1,lmax+1,3);

for i = 1:3
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

% compose
DOMEGA_F = zeros(2*lmax+1,2*lmax+1,l+1,3);

% for l = 0:lmax
%     indo1f = -l+lmax+1:l+lmax+1;
%     indu = indo1f;
%     
%     for i = 1:3
%         for l2 = 0:lmax
%             l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
%             for l1 = l1_all
%                 cl = (2*l1+1)*(2*l2+1)/(2*l+1);
% 
%                 ind_omega = -l1+lmax+1:l1+lmax+1;
%                 ind_f = -l2+lmax+1:l2+lmax+1;
% 
%                 indC = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
% 
%                 DOMEGA_F(indo1f,indo1f,l+1,i) = DOMEGA_F(indo1f,indo1f,l+1,i)+...
%                     cl*CG{l1+1,l2+1}(:,indC).'*kron(OMEGA(ind_omega,ind_omega,l1+1,i)...
%                     ,F(ind_f,ind_f,l2+1))*CG{l1+1,l2+1}(:,indC)*u(indu,indu,l+1,i).';
%             end
%         end
%     end
% end

F_tot = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3;
A = zeros(F_tot,F_tot,3);

for l = 0:lmax
    tic;
    row = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
    ind_u = -l+lmax+1:l+lmax+1;
    for i = 1:3
        for l2 = 0:lmax
            col = l2*(2*l2-1)*(2*l2+1)/3+1 : (l2+1)*(2*l2+1)*(2*l2+3)/3;
            temp = zeros((2*l+1)^2,(2*l2+1)^2);
            
            l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
            for l1 = l1_all
                cl = (2*l1+1)*(2*l2+1)/(2*l+1);
                
                col_C = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
                for j = -l1:l1
                    row_C1 = (j+l1)*(2*l2+1)+1 : (j+l1+1)*(2*l2+1);
                    for k = -l1:l1
                        row_C2 = (k+l1)*(2*l2+1)+1 : (k+l1+1)*(2*l2+1);
                        temp = temp+cl*OMEGA(j+lmax+1,k+lmax+1,l1+1,i)*...
                            kron(CG{l1+1,l2+1}(row_C2,col_C).',...
                            CG{l1+1,l2+1}(row_C1,col_C).');
                    end
                end
            end
            temp = kron(u(ind_u,ind_u,l+1,i),eye(2*l+1,2*l+1))*temp;
            A(row,col,i) = A(row,col,i)+temp;
        end
    end
    toc;
end

F_col = zeros(F_tot,1);

for l = 0:lmax
    ind_col = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
    ind = -l+lmax+1 : l+lmax+1;
    F_col(ind_col) = reshape(F(ind,ind,l+1),[],1);
end

OMEGA_F_col = [A(:,:,1)*F_col,A(:,:,2)*F_col,A(:,:,3)*F_col];

for i = 1:3
    for l = 0:lmax
        ind_col = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
        ind = -l+lmax+1 : l+lmax+1;
        DOMEGA_F(ind,ind,l+1,i) = reshape(OMEGA_F_col(ind_col,i),2*l+1,2*l+1);
    end
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');


