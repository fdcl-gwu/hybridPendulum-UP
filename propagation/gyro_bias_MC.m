function [ stat, MFG, R_res, x_res ] = gyro_bias_MC(  )

addpath('../rotation3d');
addpath('../matrix Fisher');

Ns = 1000000;

% time
sf = 100;
T = 1;
Nt = T*sf+1;

% initial samples
R_res = zeros(3,3,1000,Nt);
x_res = zeros(3,1000,Nt);

S = diag([4,4,4]);
R = pdf_MF_sampling(S,Ns);
R_res(:,:,:,1) = R(:,:,1:1000);

Miu = [0;0;0.2];
Sigma = 0.2^2*eye(3);
x = mvnrnd(Miu,Sigma,Ns)';
x_res(:,:,1) = x(:,1:1000);

% noise
H1 = diag([0.1,0.1,0.1]);
H2 = diag([0.05,0.05,0.05]);

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
for nt = 1:Nt-1
    tic;
    
    gyro_rw = mvnrnd([0,0,0],H1*H1'*sf,Ns)';
    bias_rw = mvnrnd([0,0,0],H2*H2'*sf,Ns)';
    
    R = mulRot(R,expRot((x+gyro_rw)/sf));
    x = x+bias_rw/sf;
    
    R_res(:,:,:,nt+1) = R(:,:,1:1000);
    x_res(:,:,nt+1) = x(:,1:1000);
    
    [stat.ER(:,:,nt+1),stat.Ex(:,nt+1),stat.Varx(:,:,nt+1),stat.EvR(:,nt+1),stat.ExvR(:,:,nt+1),stat.EvRvR(:,:,nt+1),...
        MFG.U(:,:,nt+1),MFG.S(:,:,nt+1),MFG.V(:,:,nt+1),MFG.P(:,:,nt+1),MFG.Miu(:,nt+1),MFG.Sigma(:,:,nt+1)] = ...
        get_stat(gather(R),gather(x),MFG.S(:,:,nt));
    
    toc;
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');

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

