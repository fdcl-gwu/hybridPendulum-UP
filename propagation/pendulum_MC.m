function [ stat, MFG, R, x ] = pendulum_MC(  )

addpath('../matrix Fisher');
addpath('../rotation3d');

Ns = 100000;

% parameters
Jd = diag([1,2,3]);
J = trace(Jd)*eye(3)-Jd;

rho = [0;0;1];
m = 10;
g = 9.8;

% time
sf = 100;
T = 0.1;
Nt = T*sf+1;

% initial conditions
S = diag([4,4,4]);
U = expRot([pi*2/3,0,0]);
R = zeros(3,3,Ns,Nt);
R(:,:,:,1) = gather(pdf_MF_sampling_gpu(U*S,Ns));

Miu = [0;0;0];
Sigma = 1^2*eye(3);
x = zeros(3,Ns,Nt);
x(:,:,1) = mvnrnd(Miu,Sigma,Ns)';

% simulate
for nt = 1:Nt-1
%     dx = J^-1*(-cross(x(:,:,nt),J*x(:,:,nt))-...
%         m*g*cross(repmat(rho,1,Ns),permute(R(3,:,:,nt),[2,3,1])));
%     x(:,:,nt+1) = x(:,:,nt)+dx/sf;
%     R(:,:,:,nt+1) = mulRot(R(:,:,:,nt),expRot(x(:,:,nt)/sf));

    M = -m*g*cross(repmat(rho,1,Ns),permute(R(3,:,:,nt),[2,3,1]));
    dR = variation_dR(x(:,:,nt),M,1/sf,J);
    R(:,:,:,nt+1) = gather(mulRot(R(:,:,:,nt),dR));
    
    M2 = -m*g*cross(repmat(rho,1,Ns),permute(R(3,:,:,nt+1),[2,3,1]));
    x(:,:,nt+1) = gather(J^-1*(permute(pagefun(@mtimes,permute(dR,[2,1,3]),permute(J*x(:,:,nt),[1,3,2])),[1,3,2]) + ...
        1/sf/2*permute(pagefun(@mtimes,permute(dR,[2,1,3]),permute(M,[1,3,2])),[1,3,2]) + ...
        1/sf/2*M2));
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

for nt = 1:Nt
    stat.ER(:,:,nt) = sum(R(:,:,:,nt),3)/Ns;
    stat.Ex(:,nt) = sum(x(:,:,nt),2)/Ns;
    stat.Varx(:,:,nt) = sum(permute(x(:,:,nt),[1,3,2]).*...
        permute(x(:,:,nt),[3,1,2]),3)/Ns - stat.Ex(:,nt)*stat.Ex(:,nt).';
    
    [U,D,V] = psvd(stat.ER(:,:,nt));
    s = pdf_MF_M2S(diag(D),diag(S));
    
    MFG.U(:,:,nt) = U;
    MFG.V(:,:,nt) = V;
    MFG.S(:,:,nt) = diag(s);
    
    Q = mulRot(U',mulRot(R(:,:,:,nt),V));
    vR = permute(cat(1,s(2)*Q(3,2,:)-s(3)*Q(2,3,:),...
        s(3)*Q(1,3,:)-s(1)*Q(3,1,:),s(1)*Q(2,1,:)-s(2)*Q(1,2,:)),[1,3,2]);
    
    stat.EvR(:,nt) = sum(vR,2)/Ns;
    stat.ExvR(:,:,nt) = sum(permute(x(:,:,nt),[1,3,2]).*permute(vR,[3,1,2]),3)/Ns;
    stat.EvRvR(:,:,nt) = sum(permute(vR,[1,3,2]).*permute(vR,[3,1,2]),3)/Ns;
    
    covxx = stat.Varx(:,:,nt);
    covxvR = stat.ExvR(:,:,nt)-stat.Ex(:,nt)*stat.EvR(:,nt).';
    covvRvR = stat.EvRvR(:,:,nt)-stat.EvR(:,nt)*stat.EvR(:,nt).';
    
    MFG.P(:,:,nt) = covxvR*covvRvR^-1;
    MFG.Miu(:,nt) = stat.Ex(:,nt)-MFG.P(:,:,nt)*stat.EvR(:,nt);
    MFG.Sigma(:,:,nt) = covxx-MFG.P(:,:,nt)*covxvR.'+...
        MFG.P(:,:,nt)*(trace(MFG.S(:,:,nt))*eye(3)-MFG.S(:,:,nt))*MFG.P(:,:,nt).';
end

rmpath('../matrix Fisher');
rmpath('../rotation3d');

end


function [ dR ] = variation_dR( x, M, dt, J )

x = gpuArray(x);
M = gpuArray(M);

Ns = size(x,2);
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

tic;
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
toc;

end

