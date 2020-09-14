addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

% time
sf = 100;
T = 1;

t = 0:1/sf:T;
Nt = T*sf+1;

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

% initial condition
f = zeros(2*B,2*B,2*B,Nt);
s0 = [2,2,2];
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
k_o = 0.2;

omega = zeros(2*B,2*B,2*B,3);
for i = 1:2*B
    for j = 1:2*B
        for k = 1:2*B
            v = logRot(R(:,:,i,j,k),'v',false);
            omega(i,j,k,:) = -k_o*v; 
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
H = diag([0,0,0]);
G = H*H';

%% propagation
for nt = 1:Nt-1
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
    
    % propagating Fourier coefficients
    dF = zeros(2*lmax+1,2*lmax+1,lmax+1);
    for l = 0:lmax
        inddF = -l+lmax+1:l+lmax+1;
        indu = inddF;
        for l2 = 0:lmax
            l1_all = find(l>=abs((0:lmax)-l2) & l<=(0:lmax)+l2)-1;
            indF = -l2+lmax+1:l2+lmax+1;
            for l1 = l1_all
                cl = (2*l1+1)*(2*l2+1)/(2*l+1);
                
                colC = l^2-(l1-l2)^2+1 : l^2-(l1-l2)^2+2*l+1;
                indomega = -l1+lmax+1:l1+lmax+1;
                
                for i = 1:3
                    dF(inddF,inddF,l+1) = dF(inddF,inddF,l+1) - cl*...
                        CG{l1+1,l2+1}(:,colC).'*kron(OMEGA(indomega,indomega,l1+1,i),...
                        F(indF,indF,l2+1,nt))*CG{l1+1,l2+1}(:,colC)*...
                        u(indu,indu,l+1,i).';
                end
            end
        end
    end
    
    F(:,:,:,nt+1) = F(:,:,:,nt) + dF/sf;
    
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
    
%     f(:,:,:,nt+1) = real(f(:,:,:,nt+1));
%     temp = f(:,:,:,nt+1);
%     temp(temp<0) = 0;
%     f(:,:,:,nt+1) = temp/(sum(sum(temp,3),1)*w.');
end

f = real(f);

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

% color map
color = zeros(100,100,3,Nt);
for nt = 1:10:Nt
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
for nt = 1:10:Nt
    figure((nt-1)/10+1);
    surf(s1,s2,s3,sum(color(:,:,:,nt),3),'LineStyle','none');
    
    axis equal;
    view([1,1,1]);
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');


