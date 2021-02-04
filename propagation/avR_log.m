function [ f, ER, S ] = avR_log( method )

close all;

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if ~exist('method','var') || isempty(method)
    method = 'euler';
end

% time
sf = 100;
T = 1;

Nt = T*sf+1;

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

% initial condition
f = zeros(2*B,2*B,2*B,Nt);
s0 = [0,0,0];
initR = expRot(pi/2*[0,0,0]);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            f(i,j,k) = exp(trace(initR'*diag(s0)...
                *R(:,:,i,j,k)))/pdf_MF_normal(s0);
        end
    end
end

F = zeros(2*lmax+1,2*lmax+1,lmax+1,Nt);

% angular velocity
k_o = -5;
G = diag([1,1,1]);

omega = zeros(2*B,2*B,2*B,3);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            omega(i,j,k,:) = 0.5*k_o*vee(G*R(:,:,i,j,k)-R(:,:,i,j,k).'*G,[],false); 
        end
    end
end

% FFT of angular velocity
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

% noise
H = diag([pi/4,pi/4,pi/4]);
G = H*H';

%% propagation
for nt = 1:Nt-1
    tic;
    
    logf = log(f(:,:,:,nt));
    
    % forward transform
    F1 = zeros(2*B,2*B,2*B);
    for k = 1:2*B
        F1(:,k,:) = fftn(logf(:,k,:));
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
    
    % propagating Fourier coefficients
    F(:,:,:,nt+1) = integrate(F(:,:,:,nt),OMEGA,1/sf,CG,u,G,method);
    
    % inverse transform
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
        logf(:,k,:) = ifftn(F1(:,k,:),'symmetric')*(2*B)^2;
    end
    
    f(:,:,:,nt+1) = exp(logf);
    f(:,:,:,nt+1) = f(:,:,:,nt+1)/sum(f(:,:,:,nt+1).*w,'all');
        
    toc;
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
for nt = 1:1:Nt
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
for nt = 1:1:Nt
    figure(nt);
    surf(s1,s2,s3,sum(color(:,:,:,nt),3),'LineStyle','none');
    
    axis equal;
    view([1,1,1]);
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');

end


function [ Fnew ] = integrate( Fold, OMEGA, dt, CG, u, G, method )

dF1 = derivative(Fold,OMEGA,CG,u,G);

% euler method
if strcmpi(method,'euler')
    Fnew = Fold+dF1*dt;
    return;
end

% midpoint method
F2 = Fold+dF1*dt/2;
dF2 = derivative(F2,OMEGA,CG,u,G);

if strcmpi(method,'midpoint')
    Fnew = Fold+dt*dF2;
    return;
end

% Runge-Kutta method
F3 = Fold+dF2*dt/2;
dF3 = derivative(F3,OMEGA,CG,u,G);

F4 = Fold+dF3*dt;
dF4 = derivative(F4,OMEGA,CG,u,G);

if strcmpi(method,'runge-kutta')
    Fnew = Fold+1/6*dt*(dF1+2*dF2+2*dF3+dF4);
    return;
end

error('''method'' must be one of ''euler'',''midpoint'', or ''Runge-Kutta''');

end


function [ dF ] = derivative(Fold, OMEGA, CG, u, G)

lmax = size(Fold,3)-1;
dF = zeros(2*lmax+1,2*lmax+1,lmax+1);

% omega*dg
temp1 = zeros(size(dF));
temp2 = zeros(size(dF));
temp3 = zeros(size(dF));
for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    temp1(ind,ind,l+1) = Fold(ind,ind,l+1)*u(ind,ind,l+1,1).';
    temp2(ind,ind,l+1) = Fold(ind,ind,l+1)*u(ind,ind,l+1,2).';
    temp3(ind,ind,l+1) = Fold(ind,ind,l+1)*u(ind,ind,l+1,3).';
end

for l1 = 0:lmax
    for l2 = 0:lmax
        indOMEGA = -l1+lmax+1:l1+lmax+1;
        indF = -l2+lmax+1:l2+lmax+1;

        A1 = CG{l1+1,l2+1}.'*kron(OMEGA(indOMEGA,indOMEGA,l1+1,1),temp1(indF,indF,l2+1))*CG{l1+1,l2+1};
        A2 = CG{l1+1,l2+1}.'*kron(OMEGA(indOMEGA,indOMEGA,l1+1,2),temp2(indF,indF,l2+1))*CG{l1+1,l2+1};
        A3 = CG{l1+1,l2+1}.'*kron(OMEGA(indOMEGA,indOMEGA,l1+1,3),temp3(indF,indF,l2+1))*CG{l1+1,l2+1};

        for l = 0:lmax
            cl = (2*l1+1)*(2*l2+1)/(2*l+1);
            if l>=abs(l1-l2) && l<=l1+l2
                indA = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
                indF = -l+lmax+1:l+lmax+1;
                
                dF(indF,indF,l+1) = dF(indF,indF,l+1)-cl*A1(indA,indA);
                dF(indF,indF,l+1) = dF(indF,indF,l+1)-cl*A2(indA,indA);
                dF(indF,indF,l+1) = dF(indF,indF,l+1)-cl*A3(indA,indA);
            end
        end
    end
end

% domega
for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    dF(ind,ind,l+1) = dF(ind,ind,l+1)-OMEGA(ind,ind,l+1,1)*u(ind,ind,l+1,1).';
    dF(ind,ind,l+1) = dF(ind,ind,l+1)-OMEGA(ind,ind,l+1,2)*u(ind,ind,l+1,2).';
    dF(ind,ind,l+1) = dF(ind,ind,l+1)-OMEGA(ind,ind,l+1,3)*u(ind,ind,l+1,3).';
end

% G*ddg
for l = 0:lmax
    ind = -l+lmax+1:l+lmax+1;
    for i = 1:3
        for j = 1:3
            dF(ind,ind,l+1) = dF(ind,ind,l+1)+0.5*G(i,j)*Fold(ind,ind,l+1)*...
                u(ind,ind,l+1,j).'*u(ind,ind,l+1,i).';
        end
    end
end

% G*dg*dg
for i = 1:3
    for j = 1:3
        for l = 0:lmax
            ind = -l+lmax+1:l+lmax+1;
            temp1(ind,ind,l+1) = Fold(ind,ind,l+1)*u(ind,ind,l+1,i).';
            temp2(ind,ind,l+1) = Fold(ind,ind,l+1)*u(ind,ind,l+1,j).';
        end
        
        for l1 = 0:lmax
            for l2 = 0:lmax
                ind1 = -l1+lmax+1:l1+lmax+1;
                ind2 = -l2+lmax+1:l2+lmax+1;
                A = CG{l1+1,l2+1}.'*kron(temp1(ind1,ind1,l1+1),temp2(ind2,ind2,l2+1))*CG{l1+1,l2+1};
                
                for l = 0:lmax
                    cl = (2*l1+1)*(2*l2+1)/(2*l+1);
                    if l>=abs(l1-l2) && l<=l1+l2
                        indA = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
                        indF = -l+lmax+1:l+lmax+1;

                        dF(indF,indF,l+1) = dF(indF,indF,l+1)+0.5*G(i,j)*cl*A(indA,indA);
                    end
                end
            end
        end
    end
end

end

