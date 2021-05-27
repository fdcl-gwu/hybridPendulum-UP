function [ stat, MFG ] = pendulum_reduced_hybrid( path, method, getc )

addpath('../rotation3d');
addpath('../matrix Fisher');
addpath('..');
addpath('mex');
addpath('discrete/mex');

if ~exist('method','var') || isempty(method)
    method = 'euler';
end

if ~(strcmpi(method,'euler') || strcmpi(method,'midpoint') || strcmpi(method,'RK2') || strcmpi(method,'RK4'))
    error('''method'' must be one of ''euler'',''midpoint'', ''RK2'', or ''RK4''');
end

if exist('path','var') && ~isempty(path)
    saveToFile = true;
else
    saveToFile = false;
end

if ~exist('getc','var') || isempty(getc)
    getc = false;
end

% parameters
J = 0.0152492;
rho = 0.0679878;
m = 1.85480;
g = 9.8;

dWall = 0.15;
h = 0.2;
r = 0.05;

epsilon = 1.0;
Hd = eye(2)*0.05;
Gd = Hd*Hd';

lambda_max = 100;
theta_t = 5*pi/180;
nD = int32(10);

% scaled parameters
tscale = sqrt(J/(m*g*rho));

% time
sf = 400;
T = 2;
Nt = T*sf+1;

% scaled parameters
dtt = 1/sf/tscale;
lambda_max = lambda_max*tscale;

% band limit
BR = 30;
Bx = 30;
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
L = 1.6*2;
x = zeros(2,2*Bx,2*Bx);
for i = 1:2*Bx
    for j = 1:2*Bx
        x(:,i,j) = [-L/2+L/(2*Bx)*(i-1);-L/2+L/(2*Bx)*(j-1)];
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

% Fourier transform of x
X = zeros(2*Bx,2*Bx,2);
for i = 1:2
    X(:,:,i) = fftn(x(i,:,:));
end

% J^-1*M(R)
mR = [permute(R(3,2,:,:,:),[1,3,4,5,2]);permute(-R(3,1,:,:,:),[1,3,4,5,2])];

% damping
b = [0.2;0.2];
bt = b*tscale;

% noise
H = eye(2)*1;
Ht = H*tscale^(3/2);
G = 0.5*(Ht*Ht.');

% lambda
[lambda,lambda_indR,PC] = getLambda_mex(R,x,dWall,h,r,theta_t,lambda_max,false);

% Omega_new
Omega_new = getOmega_mex(R,x,lambda_indR,epsilon,PC,'new');

% fc
try
    load(strcat('fcL',num2str(BR),num2str(Bx),'.mat'),'fcL');
    load(strcat('fcL',num2str(BR),num2str(Bx),'.mat'),'fcL_indx');
catch
    [fcL,fcL_indx] = getFcL_mex(x,Omega_new,lambda,Gd,nD);
    save(strcat('fcL',num2str(BR),num2str(Bx),'.mat'),'fcL','fcL_indx','-v7.3');
end

%% color settings
if getc
    % spherical grid
    Nt1 = 100;
    Nt2 = 50;
    theta1 = linspace(-pi,pi,Nt1);
    theta2 = linspace(0,pi,Nt2);
    s1 = cos(theta1)'.*sin(theta2);
    s2 = sin(theta1)'.*sin(theta2);
    s3 = repmat(cos(theta2),Nt1,1);

    % circular grid
    Nalpha = 40;
    alpha = linspace(-pi,pi-2*pi/Nalpha,Nalpha);

    % get Euler angles on grid
    e = get_Euler(reshape(s1,[],1),reshape(s2,[],1),reshape(s3,[],1),alpha);
end

%% initial conditions
S = diag([15,15,15]);
U = expRot([0,-2*pi/3,0]);
Miu = [0;0]*tscale;
Sigma = (2*tscale)^2*eye(2);

c = pdf_MF_normal(diag(S));

f = permute(exp(sum(U*S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,4,2,3]).*permute((x-Miu),...
    [4,1,2,3]).*Sigma^-1,[1,2])),[1,2,5,3,4])/c/sqrt((2*pi)^2*det(Sigma));

if saveToFile
    save(strcat(path,'/f1'),'f','-v7.3');
end

if getc
    c = pendulum_plot_getc(f,e,L);
    c = reshape(c,Nt1,Nt2,3);

    c = c/max(c(:));
    c1 = c(:,:,1);
    c2 = c(:,:,2);
    c3 = c(:,:,3);
    c = ones(size(c));
    c(:,:,1) = c(:,:,1)-c2-c3;
    c(:,:,2) = c(:,:,2)-c1-c3;
    c(:,:,3) = c(:,:,3)-c1-c2;

    if saveToFile
        save(strcat(path,'/c1'),'c');
    end
end

% initial Fourier transform
F = fftSO3R_mex('forward',f,d,w);

% pre-allocate memory
U = zeros(3,3,Nt);
V = zeros(3,3,Nt);
S = zeros(3,3,Nt);
Miu = zeros(2,Nt);
Sigma = zeros(2,2,Nt);
P = zeros(2,3,Nt);

ER = zeros(3,3,Nt);
Ex = zeros(2,Nt);
Varx = zeros(2,2,Nt);
EvR = zeros(3,Nt);
ExvR = zeros(2,3,Nt);
EvRvR = zeros(3,3,Nt);

[ER(:,:,1),Ex(:,1),Varx(:,:,1),EvR(:,1),ExvR(:,:,1),EvRvR(:,:,1),...
    U(:,:,1),S(:,:,1),V(:,:,1),P(:,:,1),Miu(:,1),Sigma(:,:,1)] = get_stat(f,R,x,w);

%% propagation
for nt = 1:Nt-1
    tic;
    
    % continuous propagation
    F = pendulum_propagate_reduced_den(F,f,X,mR,bt,G,dtt,L,u,d,w,method);
    f = fftSO3R_mex('backward',F,d);
    
    % discrete propagation
    df = pendulum_reduced_discrete_propagate(true,f,lambda,lambda_indR,fcL,fcL_indx);
    f = f + df*dtt;
    F = fftSO3R_mex('forward',f,d,w);
    
    % calculate statistics
    [ER(:,:,nt+1),Ex(:,nt+1),Varx(:,:,nt+1),EvR(:,nt+1),ExvR(:,:,nt+1),EvRvR(:,:,nt+1),...
        U(:,:,nt+1),S(:,:,nt+1),V(:,:,nt+1),P(:,:,nt+1),Miu(:,nt+1),Sigma(:,:,nt+1)]...
        = get_stat(f,R,x,w);
    
    if getc
        c = pendulum_plot_getc(f,e,L);
        c = reshape(c,Nt1,Nt2,3);
        
        c = c/max(c(:));
        c1 = c(:,:,1);
        c2 = c(:,:,2);
        c3 = c(:,:,3);
        c = ones(size(c));
        c(:,:,1) = c(:,:,1)-c2-c3;
        c(:,:,2) = c(:,:,2)-c1-c3;
        c(:,:,3) = c(:,:,3)-c1-c2;
        
        if saveToFile
            save(strcat(path,'/c',num2str(nt+1)),'c');
        end
    end
    
    if getc
        if rem(nt,4)==0
            if saveToFile
                save(strcat(path,'/f',num2str(nt+1)),'f','-v7.3');
            end
        end
    else
        if saveToFile
            save(strcat(path,'/f',num2str(nt+1)),'f','-v7.3');
        end
    end
    
    toc;
    disp(strcat('nt=',num2str(nt),' finished'));
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
    save(strcat(path,'/stat'),'stat');
    save(strcat(path,'/MFG'),'MFG');
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');
rmpath('..');
rmpath('mex');
rmpath('discrete/mex');

end


function [ ER, Ex, Varx, EvR, ExvR, EvRvR, U, S, V, P, Miu, Sigma ] = get_stat( f, R, x, w )

Bx = size(x,2)/2;
L = x(1,end,1,1)+x(1,2,1,1)-2*x(1,1,1,1);

fR = sum(f(:,:,:,:,:),[4,5])*(L/2/Bx)^2;
ER = sum(R.*permute(fR,[4,5,1,2,3]).*permute(w,[1,4,3,2,5]),[3,4,5]);

fx = permute(sum(f(:,:,:,:,:).*w,[1,2,3]),[1,4,5,2,3]);
Ex = sum(x.*fx,[2,3])*(L/2/Bx)^2;
Varx = sum(permute(x,[1,4,2,3]).*permute(x,[4,1,2,3]).*...
    permute(fx,[1,4,2,3]),[3,4,5])*(L/2/Bx)^2 - Ex*Ex.';

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
    
    ExvR = sum(permute(vR,[5,1,2,3,4]).*permute(w,[1,3,4,2]).*permute(f,[6,7,1,2,3,4,5]),[3,4,5]);
    ExvR = sum(permute(x,[1,4,5,6,7,2,3]).*ExvR,[5,6,7])*(L/2/Bx)^2;

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


function [ e ] = get_Euler( s1, s2, s3, alpha )

Ns = length(s1);
Na = length(alpha);

R = zeros(3,3,Ns,Na,3);
for ns = 1:Ns
    rref = [s1(ns);s2(ns);s3(ns)];

    for nd = 1:3
        Rref = eye(3);
        jk = setdiff(1:3,nd);
        Rref(:,nd) = rref;
        Rref(:,jk) = null(rref');
        D = eye(3);
        D(jk(1),jk(1)) = det(Rref);
        Rref = Rref*D;

        vref = zeros(3,Na);
        vref(nd,:) = alpha;
        R(:,:,ns,:,nd) = mulRot(Rref,expRot(vref));
    end
end

e = rot2eul(reshape(R,[3,3,Ns*Na*3]),'zyz');
e = reshape(e,[3,Ns,Na,3]);

end

