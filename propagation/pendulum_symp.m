function [ stat, MFG ] = pendulum_symp( path )

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');

if exist('path','var') && ~isempty(path)
    saveToFile = true;
else
    saveToFile = false;
end

% parameters
Jd = diag([1,2,3]);
J = trace(Jd)*eye(3)-Jd;

rho = [0;0;1];
m = 10;
g = 3;

% band limit
BR = 10;
Bx = 10;
lmax = BR-1;

% time
sf = 100;
T = 1;
Nt = T*sf+1;

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
L = 10;
x = zeros(3,2*Bx,2*Bx,2*Bx);
x1 = -L/2 : L/(2*Bx) : L/2-L/(2*Bx);
for i = 1:2*Bx
    for j = 1:2*Bx
        for k = 1:2*Bx
            x(:,i,j,k) = [x1(i);x1(j);x1(k)];
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

% derivatives
u = getu(lmax);

% initial conditions
S = diag([4,4,4]);
U = expRot([pi*2/3,0,0]);
Miu = [0;0;0];
Sigma = 1^2*eye(3);

c = pdf_MF_normal(diag(S));

f = permute(exp(sum(U*S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,5,2,3,4]).*permute((x-Miu),...
    [5,1,2,3,4]).*Sigma^-1,[1,2])),[1,2,6,3,4,5])/c/sqrt((2*pi)^3*det(Sigma));

if saveToFile
    save(strcat(path,'\f1'),'f');
end

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

% grid one step before
Rb = zeros(3,3,(2*BR)^3,2*Bx,2*Bx,2*Bx);
xb = zeros(3,(2*BR)^3,2*Bx,2*Bx,2*Bx);

const_2Bx = 2*Bx;
parfor k = 1:const_2Bx
    for j = 1:const_2Bx
        for i = 1:const_2Bx
            [Rb(:,:,:,i,j,k),xb(:,:,i,j,k)] = LGVI(reshape(R,3,3,(2*BR)^3),x(:,i,j,k),-1/sf,m,rho,J,g);
        end
    end
end

eb = zeros(3,(2*BR)^3,2*Bx,2*Bx,2*Bx);
parfor k = 1:const_2Bx
    for j = 1:const_2Bx
        for i = 1:const_2Bx
            eb(:,:,i,j,k) = rot2eul(Rb(:,:,:,i,j,k),'zyz');
        end
    end
end
eb(1,:,:,:,:) = wrapTo2Pi(eb(1,:,:,:,:));
eb(3,:,:,:,:) = wrapTo2Pi(eb(3,:,:,:,:));

clear Rb;

%% propagation
F = zeros(2*lmax+1,2*lmax+1,lmax+1,2*Bx,2*Bx,2*Bx);
nR = (2*BR)^3;
for nt = 1:Nt-1
    tic;
    
    % Fourier transform
    F1 = zeros(2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
    for k = 1:2*BR
        F1(:,k,:,:,:,:) = fftn(f(:,k,:,:,:,:));
    end
    F1 = fftshift(fftshift(F1,1),3);
    F1 = flip(flip(F1,1),3);
    
    for l = 0:lmax
        for m = -l:l
            for n = -l:l
                F(m+lmax+1,n+lmax+1,l+1,:,:,:) = sum(w.*F1(m+lmax+1,:,n+lmax+1,:,:,:).*...
                    permute(d(m+lmax+1,n+lmax+1,l+1,:),[1,4,3,2]),2);
            end
        end
    end
    
    % propagate
    Fnew = zeros((2*BR)^3,2*Bx,2*Bx,2*Bx);
    parfor k = 1:const_2Bx
        tic;
        for j = 1:const_2Bx
            for i = 1:const_2Bx
                for r = 1:nR
                    Fnew(r,i,j,k) = invFourier(F,eb(:,r,i,j,k),xb(:,r,i,j,k),BR,Bx,L);
                end
            end
        end
        toc;
    end
    
    f = reshape(Fnew,2*BR,2*BR,2*BR,2*Bx,2*Bx,2*Bx);
    
    [ER(:,:,nt+1),Ex(:,nt+1),Varx(:,:,nt+1),EvR(:,nt+1),ExvR(:,:,nt+1),EvRvR(:,:,nt+1),...
        U(:,:,nt+1),S(:,:,nt+1),V(:,:,nt+1),P(:,:,nt+1),Miu(:,nt+1),Sigma(:,:,nt+1)] = get_stat(f,R,x,w);
    
    if saveToFile
        save(strcat(path,'\f',num2str(nt+1)),'f');
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


function [ R, x ] = LGVI( R, x, dt, m, rho, J, g)

Ns = size(R,3);

x = gpuArray(x);
R = gpuArray(R);

M = -m*g*permute(cat(1,rho(2)*R(3,3,:)-rho(3)*R(3,2,:),...
    rho(3)*R(3,1,:)-rho(1)*R(3,3,:),...
    rho(1)*R(3,2,:)-rho(2)*R(3,1,:)),[1,3,2]);

dR = gpuArray.zeros(3,3,Ns);
A = dt*J*x+dt^2/2*M;

% G
normv = @(v) sqrt(sum(v.^2,1));
Gv = @(v) sin(normv(v))./normv(v).*(J*v)+...
    (1-cos(normv(v)))./sum(v.^2,1).*cross(v,J*v);

% Jacobian of G
DGv = @(v) permute((cos(normv(v)).*normv(v)-sin(normv(v)))./normv(v).^3,[1,3,2]).*...
    pagefun(@mtimes,permute(J*v,[1,3,2]),permute(v,[3,1,2])) + ...
    permute(sin(normv(v))./normv(v),[1,3,2]).*J + ...
    permute((sin(normv(v)).*normv(v)-2*(1-cos(normv(v))))./sum(v.^2,1).^2,[1,3,2]).*...
    pagefun(@mtimes,permute(cross(v,J*v),[1,3,2]),permute(v,[3,1,2])) + ...
    permute((1-cos(normv(v)))./sum(v.^2,1),[1,3,2]).*(-hat(J*v)+pagefun(@mtimes,hat(v),J));

% GPU matrix exponential
expRot = @(v) eye(3) + permute(sin(normv(v))./normv(v),[1,3,2]).*hat(v) + ...
    permute((1-cos(normv(v)))./sum(v.^2,1),[1,3,2]).*pagefun(@mtimes,hat(v),hat(v));

% initializa Newton method
v = gpuArray.ones(3,Ns)*1e-5;

% tolerance
epsilon = 1e-5;

% step size
alpha = 1;

% Newton method
n_finished = 0;
ind = 1:Ns;

n_step = 0;
while n_finished < Ns
    ind_finished = find(normv(A-Gv(v))<epsilon);
    dR(:,:,ind(ind_finished)) = expRot(v(:,ind_finished));
    ind = setdiff(ind,ind(ind_finished));
    n_finished = n_finished+length(ind_finished);
    
    v(:,ind_finished) = [];
    A(:,ind_finished) = [];
    v = v + alpha*permute(pagefun(@mtimes,pagefun(@inv,DGv(v)),permute(A-Gv(v),[1,3,2])),[1,3,2]);
    
    n_step = n_step+1;
end

R = gather(mulRot(R,dR));
    
M2 = -m*g*permute(cat(1,rho(2)*R(3,3,:)-rho(3)*R(3,2,:),...
    rho(3)*R(3,1,:)-rho(1)*R(3,3,:),...
    rho(1)*R(3,2,:)-rho(2)*R(3,1,:)),[1,3,2]);
x = gather(J^-1*(permute(pagefun(@mtimes,permute(dR,[2,1,3]),permute(J*x,[1,3,2])),[1,3,2]) + ...
    dt/2*permute(pagefun(@mtimes,permute(dR,[2,1,3]),permute(M,[1,3,2])),[1,3,2]) + ...
    dt/2*M2));

end


function [ f ] = invFourier( F, e, x, BR, Bx, L )

lmax = BR-1;

d = Wigner_d(e(2),lmax);

DR = exp(-1i*e(1)*(-lmax:lmax)').*d.*exp(-1i*e(3)*(-lmax:lmax));

Fx = permute(sum(DR.*F.*permute(1:2:2*lmax+1,[1,3,2]),[1,2,3]),[4,5,6,1,2,3]);
Fx = ifftshift(ifftshift(ifftshift(Fx,1),2),3);

Dx = exp(2*pi*1i*(-Bx:Bx-1)'*(x(1)/L+1/2)).*exp(2*pi*1i*(-Bx:Bx-1)*(x(2)/L+1/2)).*...
    exp(2*pi*1i*permute(-Bx:Bx-1,[1,3,2])*(x(3)/L+1/2))/(2*Bx)^3;

f = sum(Fx.*Dx,[1,2,3]);
f = real(f);

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
    ExvR = sum(permute(vR,[5,1,2,3,4]).*permute(x,[1,5,6,7,8,2,3,4]).*...
        permute(w,[1,3,4,2]).*permute(f(:,:,:,:,:,:),[7,8,1,2,3,4,5,6]),[3,4,5,6,7,8])*(L/2/Bx)^3;

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

