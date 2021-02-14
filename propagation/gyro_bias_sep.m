function [ stat, MFG ] = gyro_bias_sep( path )

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if exist('path','var') && ~isempty(path)
    saveToFile = true;
else
    saveToFile = false;
end

% time
sf = 100;
T = 1;
Nt = T*sf+1;

% band limit
BR = 10;
Bx = 10;
lmax = BR-1;

% grid over SO(3)
alpha = reshape(pi/BR*(0:(2*BR-1)),1,1,[]);
beta = reshape(pi/(4*BR)*(2*(0:(2*BR-1))+1),1,1,[]);
gamma = reshape(pi/BR*(0:(2*BR-1)),1,1,[]);

ca = cos(alpha);
sa = sin(alpha);
cb = cos(beta);
sb = sin(beta);
cg = cos(gamma);
sg = sin(gamma);

Ra = [ca,-sa,zeros(1,1,2*BR);sa,ca,zeros(1,1,2*BR);zeros(1,1,2*BR),zeros(1,1,2*BR),ones(1,1,2*BR)];
Rb = [cb,zeros(1,1,2*BR),sb;zeros(1,1,2*BR),ones(1,1,2*BR),zeros(1,1,2*BR);-sb,zeros(1,1,2*BR),cb];
Rg = [cg,-sg,zeros(1,1,2*BR);sg,cg,zeros(1,1,2*BR);zeros(1,1,2*BR),zeros(1,1,2*BR),ones(1,1,2*BR)];

R = zeros(3,3,2*BR,2*BR,2*BR);
for i = 1:2*BR
    for j = 1:2*BR
        for k = 1:2*BR
            R(:,:,i,j,k) = Ra(:,:,i)*Rb(:,:,j)*Rg(:,:,k);
        end
    end
end

% grid over R^3
x = zeros(3,2*Bx,2*Bx,2*Bx);
L = 2;
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            x(:,i,j,k) = [-L/2+2/(2*Bx)*(i-1);-L/2+2/(2*Bx)*(j-1);-L/2+2/(2*Bx)*(k-1)];
        end
    end
end

% weights
w = zeros(1,2*BR);
for j = 1:2*BR
    w(j) = 1/(4*BR^3)*sin(beta(j))*sum(1./(2*(0:BR-1)+1).*sin((2*(0:BR-1)+1)*beta(j)));
end

% Wigner_d
d = zeros(2*lmax+1,2*lmax+1,lmax+1,2*BR);
for j = 1:2*BR
    d(:,:,:,j) = Wigner_d(beta(j),lmax);
end

% D(exp(x))
Dx = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            e = rot2eul(expRot(x(:,i,j,k)/sf),'zyz');
            dijk = Wigner_d(e(2),lmax);
            
            for l = 0:lmax
                ind = -l+lmax+1:l+lmax+1;
                Dx(ind,ind,l+1,i,j,k) = dijk(ind,ind,l+1).*...
                    exp(-1i*e(1)*(-l:l)).'.*exp(-1i*e(3)*(-l:l));
            end
        end
    end
end

% initial condition
s = [4,4,4];
S = diag(s);
Miu = [0;0;0.2];
Sigma = 0.2^2*eye(3);

c = pdf_MF_normal(s);

f = permute(exp(sum(S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,5,2,3,4]).*permute((x-Miu),...
    [5,1,2,3,4]).*Sigma^-1,[1,2])),[1,2,6,3,4,5])/c/sqrt((2*pi)^3*det(Sigma));

if saveToFile
    save(strcat(path,'\f1'),'f');
end

% noise
H1 = diag([0.1,0.1,0.1]);
H2 = diag([0.05,0.05,0.05]);

G1 = 0.5*(H1*H1.');
G2 = 0.5*(H2*H2.');

% pre-allocate memory
U = zeros(3,3,Nt);
V = zeros(3,3,Nt);
S = zeros(3,3,Nt);
Miu = zeros(3,Nt);
Sigma = zeros(3,3,Nt);
P = zeros(3,3,Nt);

ER = zeros(3,3,Nt);
Ex = zeros(3,Nt);
Varx = zeros(3,3,Nt);
EvR = zeros(3,Nt);
ExvR = zeros(3,3,Nt);
EvRvR = zeros(3,3,Nt);

[ER(:,:,1),Ex(:,1),Varx(:,:,1),EvR(:,1),ExvR(:,:,1),EvRvR(:,:,1),...
    U(:,:,1),S(:,:,1),V(:,:,1),P(:,:,1),Miu(:,1),Sigma(:,:,1)] = get_stat(f,R,x,w);

%% propagation
for nt = 1:Nt-1
    tic;
    f = integrate(f,d,w,Dx,G2,L,1/sf);
    
    [ER(:,:,nt+1),Ex(:,nt+1),Varx(:,:,nt+1),EvR(:,nt+1),ExvR(:,:,nt+1),EvRvR(:,:,nt+1),...
        U(:,:,nt+1),S(:,:,nt+1),V(:,:,nt+1),P(:,:,nt+1),Miu(:,nt+1),Sigma(:,:,nt+1)] = get_stat(f,R,x,w);
    
    if saveToFile
        save(strcat(path,'\f',num2str(nt+1)),'f','-v7.3');
    end
    toc;
end

stat.ER = ER;
stat.Ex = Ex;
stat.Varx = Varx;
stat.EvR = EvR;
stat.ExvR = ExvR;
stat.EvRvR = EvRvR;

MFG.U = U;
MFG.S = S;
MFG.V = V;
MFG.Miu = Miu;
MFG.Sigma = Sigma;
MFG.P = P;

if saveToFile
    save(strcat(path,'\stat'),'stat');
    save(strcat(path,'\MFG'),'MFG');
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');

end


function [ f ] = integrate( f, d, w, Dx, G2, L, dt )

BR = size(f,1)/2;
Bx = size(f,4)/2;
lmax = BR-1;

% forward Fourier transform
F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            for l = 1:2*BR
                F1(:,l,:,i,j,k) = fftn(f(:,l,:,i,j,k));
            end
        end
    end
end
F1 = fftshift(fftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
FR = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
for l = 0:lmax
    for m = -l:l
        for n = -l:l
            FR(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            F(m+lmax+1,n+lmax+1,l+1,:,:,:) = fftn(FR(m+lmax+1,n+lmax+1,l+1,:,:,:));
        end
    end
end

Fx = permute(F(lmax+1,lmax+1,1,:,:,:),[4,5,6,1,2,3]);

fx = sum(f.*w);
FR = FR./fx;

% calculate dFx
dFx = zeros(2*Bx,2*Bx,2*Bx);
for i = 1:3
    for j = 1:3
        if i==j
            c = 4*pi^2*[0:Bx-1,-Bx:-1].^2/L^2;
            c = -shiftdim(c,2-i);
        else
            c = 2*pi*[0:Bx-1,0,-Bx+1:-1]/L;
            c = -shiftdim(c,2-i).*shiftdim(c,2-j);
        end
        
        dFx = dFx + G2(i,j)*Fx.*c;
    end
end

% integrate
Fx = Fx + dFx*dt;

% backward Fourier transform for fx
fx = ifftn(Fx,'symmetric');

% calculated rotated FR
FR_r = zeros(size(FR));
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            for l = 0:lmax
                ind = -l+lmax+1:l+lmax+1;
                FR_r(ind,ind,l+1,i,j,k) = FR(ind,ind,l+1,i,j,k)*conj(Dx(ind,ind,l+1,i,j,k));
            end
        end
    end
end

% backward Fourier transfomr for rotated FR
F1 = zeros(2*BR-1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx);
for m = -lmax:lmax
    for n = -lmax:lmax
        lmin = max(abs(m),abs(n));
        F_mn = FR_r(m+lmax+1,n+lmax+1,lmin+1:lmax+1,:,:,:);
        
        for k = 1:2*BR
            d_jk_betak = d(m+lmax+1,n+lmax+1,lmin+1:lmax+1,k);
            F1(m+lmax+1,k,n+lmax+1,:,:,:) = sum((2*permute(lmin:lmax,...
                [1,3,2])+1).*F_mn.*d_jk_betak,3);
        end
    end
end

F1 = cat(1,F1,zeros(1,2*BR,2*BR-1,2*Bx,2*Bx,2*Bx));
F1 = cat(3,F1,zeros(2*BR,2*BR,1,2*Bx,2*Bx,2*Bx));
F1 = ifftshift(ifftshift(F1,1),3);
F1 = flip(flip(F1,1),3);

for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            for l = 1:2*BR
                f(:,l,:,i,j,k) = ifftn(F1(:,l,:,i,j,k),'symmetric')*(2*BR)^2*fx(i,j,k);
            end
        end
    end
end

end


function [ ER, Ex, Varx, EvR, ExvR, EvRvR, U, S, V, P, Miu, Sigma ] = get_stat( f, R, x, w )

Bx = size(x,2)/2;
L = x(1,end,1,1)+x(1,2,1,1)-2*x(1,1,1,1);

fR = sum(f(:,:,:,:,:,:),[4,5,6])*(L/2/Bx)^3;
ER = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);

fx = permute(sum(f(:,:,:,:,:,:).*w,[1,2,3]),[1,4,5,6,2,3]);
Ex = sum(x.*fx,[2,3,4])*(L/2/Bx)^3;
Varx = sum(permute(x,[1,5,2,3,4]).*permute(x,[5,1,2,3,4]).*...
    permute(fx,[1,5,2,3,4]),[3,4,5])*(L/2/Bx)^3 - Ex*Ex.';

[U,D,V] = psvd(ER);
try
    s = pdf_MF_M2S(diag(D),[0;0;0]);
    S = diag(s);

    Q = gather(pagefun(@mtimes,U.',pagefun(@mtimes,gpuArray(R),V)));
    vR = permute(cat(1,s(2)*Q(3,2,:,:,:)-s(3)*Q(2,3,:,:,:),...
        s(3)*Q(1,3,:,:,:)-s(1)*Q(3,1,:,:,:),...
        s(1)*Q(2,1,:,:,:)-s(2)*Q(1,2,:,:,:)),[1,3,4,5,2]);

    EvR = sum(vR.*permute(w,[1,3,2]).*permute(fR,[4,1,2,3]),[2,3,4]);
    EvRvR = sum(permute(vR,[1,5,2,3,4]).*permute(vR,[5,1,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(fR,[4,5,1,2,3]),[3,4,5]);
    
    ExvR = sum(permute(vR,[5,1,2,3,4]).*permute(w,[1,3,4,2]).*permute(f,[7,8,1,2,3,4,5,6]),[3,4,5]);
    ExvR = sum(permute(x,[1,5,6,7,8,2,3,4]).*ExvR,[6,7,8])*(L/2/Bx)^3;

    covxx = Varx;
    covxvR = ExvR-Ex*EvR.';
    covvRvR = EvRvR-EvR*EvR.';

    P = covxvR*covvRvR^-1;
    Miu = Ex-P*EvR;
    Sigma = covxx-P*covxvR.'+P*(trace(S)*eye(3)-S)*P.';
catch
    S = NaN(3,3);
    EvR = NaN(3,1);
    EvRvR = NaN(3,3);
    ExvR = NaN(3,3);
    P = NaN(3,3);
    Miu = NaN(3,1);
    Sigma = NaN(3,3);
end

end

