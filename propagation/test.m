clear;
close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

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
k_o = 0.2;

omega = zeros(2*B,2*B,2*B,3);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            v = logRot(R(:,:,i,j,k),'v',false);
            omega(i,j,k,:) = -k_o*v;
%             omega(i,j,k,:) = exp(trace(diag([6,4,2])*R(:,:,i,j,k)))/pdf_MF_normal([6,4,2]);
        end
    end
end

% omega1_f
omega1_f = zeros(2*B,2*B,2*B);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            omega1_f(i,j,k) = omega(i,j,k,1)*f(i,j,k);
        end
    end
end

% domega1_f
domega1_f = zeros(2*B,2*B,2*B);
hate1 = [0,0,0;0,0,-1;0,1,0];
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            [v,~,theta] = logRot(R(:,:,i,j,k),[],false);
            adv = @(x)v*x-x*v;
            dexpv = @(x) x - 1/2*adv(x) +...
                (1/theta^2-(1+cos(theta))/(2*theta*sin(theta)))*adv(adv(x));
            
            dexpve1 = vee(dexpv(hate1));
            domega1_f(i,j,k) = -k_o*dexpve1(1)*f(i,j,k) +...
                omega(i,j,k,1)*f(i,j,k)*trace(diag(s0)*R(:,:,i,j,k)*hate1);
%             domega1_f(i,j,k) = f(i,j,k)*omega(i,j,k,1)*...
%                 trace(diag([8,6,4])*R(:,:,i,j,k)*hate1);
        end
    end
end

%% DFT of domega1_f directly
DOMEGA1_F_direct = zeros(2*lmax+1,2*lmax+1,lmax+1);

S1 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for kk = 1:2*B
        S1(ii,:,kk) = ifftshift(ifft(domega1_f(:,ii,kk)))*(2*B);
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
            DOMEGA1_F_direct(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
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
OMEGA1 = zeros(2*lmax+1,2*lmax+1,l+1);

S1 = zeros(2*B,2*B,2*B);
for ii = 1:2*B
    for kk = 1:2*B
        S1(ii,:,kk) = ifftshift(ifft(omega(:,ii,kk,1)))*(2*B);
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
            OMEGA1(jj+lmax+1,kk+lmax+1,l+1) = sum(w.*S2(:,jj+lmax+2,kk+lmax+2).'.*...
                reshape(d(jj+lmax+1,kk+lmax+1,l+1,:),1,[]));
        end
    end
end

% compose
OMEGA1_F = zeros(2*lmax+1,2*lmax+1,l+1);

for l = 0:lmax
    indo1f = -l+lmax+1:l+lmax+1;
    indu = indo1f;
    
    for l2 = 0:lmax
        l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
        for l1 = l1_all
            cl = (2*l1+1)*(2*l2+1)/(2*l+1);
            
            ind_omega = -l1+lmax+1:l1+lmax+1;
            ind_f = -l2+lmax+1:l2+lmax+1;

            indC = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
            
            OMEGA1_F(indo1f,indo1f,l+1) = OMEGA1_F(indo1f,indo1f,l+1)+...
                cl*CG{l1+1,l2+1}(:,indC).'*kron(OMEGA1(ind_omega,ind_omega,l1+1)...
                ,F(ind_f,ind_f,l2+1))*CG{l1+1,l2+1}(:,indC);
        end
    end
end

% for l = 0:lmax
%     indo1f = -l+lmax+1:l+lmax+1;
%     indu = indo1f;
%     
%     for l2 = 0:lmax
%         l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
%         for l1 = l1_all
%             cl = (2*l1+1)*(2*l2+1)/(2*l+1);
%             
%             indf = -l2+lmax+1:l2+lmax+1;
%             col_C = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
%             
%             X = zeros(2*l+1,2*l+1);
%             for j = -l1:l1
%                 for k = -l1:l1
%                     row_C1 = (j+l1)*(2*l2+1)+1 : (j+l1+1)*(2*l2+1);
%                     row_C2 = (k+l1)*(2*l2+1)+1 : (k+l1+1)*(2*l2+1);
%                     
%                     DOMEGA1_F(indo1f,indo1f,l+1) = DOMEGA1_F(indo1f,indo1f,l+1)+...
%                         cl*OMEGA1(j+lmax+1,k+lmax+1,l1+1)*CG{l1+1,l2+1}(row_C1,col_C).'*...
%                         F(indf,indf,l2+1)*CG{l1+1,l2+1}(row_C2,col_C)*u(indu,indu,l+1,1).';
%                 end
%             end
%         end
%     end
% end

% F_tot = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3;
% A = zeros(F_tot,F_tot);
% 
% for l = 0:lmax
%     tic;
%     indu = -l+lmax+1:l+lmax+1;
%     for m = -l:l
%         for n = -l:l
%             row = l*(2*l-1)*(2*l+1)/3 + (n+l)*(2*l+1) + m+l+1;
%             for l2 = 0:lmax
%                 l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
%                 for p = -l2:l2
%                     for q = -l2:l2
%                         col = l2*(2*l2-1)*(2*l2+1)/3 + (q+l2)*(2*l2+1) + p+l2+1;
%                         for l1 = l1_all
%                             cl = (2*l1+1)*(2*l2+1)/(2*l+1);
%                             
%                             indomega= -l1+lmax+1:l1+lmax+1;
%                             
%                             row_C1 = (0:2*l1)*(2*l2+1)+p+l2+1;
%                             col_C1 = l^2-(l1-l2)^2+m+l+1;
%                             col_C2 = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
%                             
%                             Cu = CG{l1+1,l2+1}(:,col_C2)*u(indu,indu,l+1,1).';
%                             
%                             row_Cu = (0:2*l1)*(2*l2+1)+q+l2+1;
%                             col_Cu = n+l+1;
%                             
%                             A(row,col) = A(row,col)+...
%                                 cl*trace(OMEGA1(indomega,indomega,l1+1)*...
%                                 Cu(row_Cu,col_Cu)*CG{l1+1,l2+1}(row_C1,col_C1).');
%                         end
%                     end
%                 end
%             end
%         end
%     end
%     toc;
% end
% 
% F_col = zeros(F_tot,1);
%     
% for l = 0:lmax
%     ind_col = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
%     ind = -l+lmax+1 : l+lmax+1;
%     F_col(ind_col) = reshape(F(ind,ind,l+1),[],1);
% end
% 
% OMEGA1_F_col = A*F_col;
% 
% for l = 0:lmax
%     ind_col = l*(2*l-1)*(2*l+1)/3+1 : (l+1)*(2*l+1)*(2*l+3)/3;
%     ind = -l+lmax+1 : l+lmax+1;
%     DOMEGA1_F(ind,ind,l+1) = reshape(OMEGA1_F_col(ind_col),2*l+1,2*l+1);
% end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');


