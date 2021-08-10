function [ stat, MFG, R_res, x_res ] = pendulum_MC_hybrid( useLambda, R, x )

addpath('../matrix Fisher');
addpath('../rotation3d');

if nargin<1 || isempty(useLambda)
    useLambda = false;
end

Ns = 1000000;

% parameters
J = diag([0.01436,0.01436,0.0003326]);

rho = [0;0;0.1];
m = 1.0642;
g = 9.8;

b = [0.2;0.2;0];
H = diag([1,1,0]);

dWall = 0.12;
h = 0.2;
r = 0.025;

epsilon = 0.8;
Hd = eye(2)*0.05;

if useLambda
    lambda_max = 100;
    theta_t = 5*pi/180;
end

% scaled parameters
Jt = J/J(1,1);
tscale = sqrt(J(1,1)/(m*g*rho(3)));

% collision parameters
theta0 = asin(dWall/sqrt(h^2+r^2)) - asin(r/sqrt(h^2+r^2));

% time
sf = 400;
T = 8;
Nt = T*sf+1;

% scaled time
dtt = 1/sf/tscale;

bt = b*tscale;
Ht = H*tscale^(3/2);

if useLambda
    lambda_max = lambda_max*tscale;
end

% initial conditions
if ~exist('R','var')
    S = diag([15,15,15]);
    U = expRot([0,-2*pi/3,0]);
    R = pdf_MF_sampling_gpu(U*S,Ns);
end

if ~exist('x','var')
    Miu = [0;0]*tscale;
    Sigma = (2*tscale)^2*eye(2);
    x = mvnrnd(Miu,Sigma,Ns)';
    x = [x;zeros(1,Ns)];
    x = gpuArray(x);
else
    x = tscale*x;
    Ns = size(x,2);
end

% statistics
MFG.U = zeros(3,3,Nt);
MFG.V = zeros(3,3,Nt);
MFG.S = zeros(3,3,Nt);
MFG.Miu = zeros(3,Nt);
MFG.Sigma = zeros(3,3,Nt);
MFG.P = zeros(3,3,Nt);

stat.ER = zeros(3,3,Nt);
stat.Ex = zeros(3,Nt);
stat.Varx = zeros(3,3,Nt);
stat.EvR = zeros(3,Nt);
stat.ExvR = zeros(3,3,Nt);
stat.EvRvR = zeros(3,3,Nt);

[stat.ER(:,:,1),stat.Ex(:,1),stat.Varx(:,:,1),stat.EvR(:,1),stat.ExvR(:,:,1),stat.EvRvR(:,:,1),...
        MFG.U(:,:,1),MFG.S(:,:,1),MFG.V(:,:,1),MFG.P(:,:,1),MFG.Miu(:,1),MFG.Sigma(:,:,1)] = ...
        get_stat(gather(R),gather(x),zeros(3));
    
% simulate
R_res = zeros(3,3,1000,Nt);
x_res = zeros(3,1000,Nt);
R_res(:,:,:,1) = gather(R(:,:,1:1000));
x_res(:,:,1) = gather(x(:,1:1000));

for nt = 1:Nt-1
    tic;
    
    % continuous
    [R,x] = LGVI(R,x,dtt,Jt,bt);
    x = x + sqrt(dtt)*Ht*[randn([1,Ns],'gpuArray');randn([1,Ns],'gpuArray');randn([1,Ns],'gpuArray')];
    
    % discrete
    b3 = permute(R(:,3,:),[1,3,2]);
    theta = asin(b3(1,:));
    
    omega = permute(pagefun(@mtimes,R,permute(x,[1,3,2])),[1,3,2]);
    PC = (h-r*tan(theta)).*b3 + r*sec(theta).*[1;0;0];
    vC = cross(omega,PC);
    
    if useLambda
        u = rand(1,Ns,'gpuArray');
        lambda = getLambda(lambda_max,theta_t,theta0,theta,vC);
        ind_jump = find(exp(-lambda*dtt) < u);
        N_jump = length(ind_jump);
    else
        ind_jump = find(theta>=theta0 & vC(1,:)>0);
        N_jump = length(ind_jump);
    end
    
    if N_jump > 0
        t = cat(1,zeros(1,N_jump),PC(3,ind_jump),-PC(2,ind_jump));
        t = t./sqrt(sum(t.^2,1));
        tB = permute(pagefun(@mtimes,pagefun(@transpose,R(:,:,ind_jump)),...
            permute(t,[1,3,2])),[1,3,2]);
        
        x(:,ind_jump) = x(:,ind_jump) - (1+epsilon)*sum(x(:,ind_jump).*tB,1).*tB;
        xi = randn(2,1,N_jump,'gpuArray');
        x(1:2,ind_jump) = x(1:2,ind_jump) + permute(pagefun(@mtimes,Hd,xi),[1,3,2]);
    end
    
    % statistics
    R_res(:,:,:,nt+1) = gather(R(:,:,1:1000));
    x_res(:,:,nt+1) = gather(x(:,1:1000));
    
    [stat.ER(:,:,nt+1),stat.Ex(:,nt+1),stat.Varx(:,:,nt+1),stat.EvR(:,nt+1),stat.ExvR(:,:,nt+1),stat.EvRvR(:,:,nt+1),...
        MFG.U(:,:,nt+1),MFG.S(:,:,nt+1),MFG.V(:,:,nt+1),MFG.P(:,:,nt+1),MFG.Miu(:,nt+1),MFG.Sigma(:,:,nt+1)] = ...
        get_stat(gather(R),gather(x),MFG.S(:,:,nt));
    
    toc;
end

rmpath('../matrix Fisher');
rmpath('../rotation3d');

end


function [ R, x ] = LGVI( R, x, dt, J, bt )

Ns = size(R,3);

x = gpuArray(x);
R = gpuArray(R);

M = -[permute(-R(3,2,:),[1,3,2]);permute(R(3,1,:),[1,3,2]);zeros(1,Ns)];
M = M - bt.*x;

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
v = gpuArray.ones(2,Ns)*1e-5;
v = [v;gpuArray.zeros(1,Ns)];

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

R = mulRot(R,dR);
    
M2 = -[permute(-R(3,2,:),[1,3,2]);permute(R(3,1,:),[1,3,2]);zeros(1,Ns)];
M2 = M2 - bt.*x;
x = J^-1*(permute(pagefun(@mtimes,permute(dR,[2,1,3]),permute(J*x,[1,3,2])),[1,3,2]) + ...
    dt/2*permute(pagefun(@mtimes,permute(dR,[2,1,3]),permute(M,[1,3,2])),[1,3,2]) + dt/2*M2);
x(3,:) = 0;

end


function [ lambda ] = getLambda(lambda_max,theta_t,theta0,theta,vC)

ind0 = theta<theta0-theta_t | vC(1,:)<=0;
indmax = theta>theta0+theta_t & vC(1,:)>0;
indmid = ~(ind0 | indmax);

lambda(ind0) = 0;
lambda(indmax) = lambda_max;
lambda(indmid) = lambda_max/2*sin(pi/(2*theta_t)*(theta(indmid)-theta0)) + lambda_max/2;

end


function [ ER, Ex, Varx, EvR, ExvR, EvRvR, U, S, V, P, Miu, Sigma ] = get_stat(R, x, S)

Ns = size(R,3);

ER = sum(R,3)/Ns;
Ex = sum(x,2)/Ns;
Varx = sum(permute(x,[1,3,2]).*permute(x,[3,1,2]),3)/Ns - Ex*Ex.';

[U,D,V] = psvd(ER);
s = pdf_MF_M2S(diag(D),diag(S));
S = diag(s);

Q = mulRot(U',mulRot(R,V));
vR = permute(cat(1,s(2)*Q(3,2,:)-s(3)*Q(2,3,:),...
    s(3)*Q(1,3,:)-s(1)*Q(3,1,:),s(1)*Q(2,1,:)-s(2)*Q(1,2,:)),[1,3,2]);

EvR = sum(vR,2)/Ns;
ExvR = sum(permute(x,[1,3,2]).*permute(vR,[3,1,2]),3)/Ns;
EvRvR = sum(permute(vR,[1,3,2]).*permute(vR,[3,1,2]),3)/Ns;

covxx = Varx;
covxvR = ExvR-Ex*EvR.';
covvRvR = EvRvR-EvR*EvR.';

P = covxvR*covvRvR^-1;
Miu = Ex-P*EvR;
Sigma = covxx-P*covxvR.'+P*(trace(S)*eye(3)-S)*P.';

end

