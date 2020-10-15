function [ stat, MFG, R, x ] = gyro_bias_MC(  )

addpath('../rotation3d');
addpath('../matrix Fisher');

Ns = 1000000;

% time
sf = 10;
T = 1;
Nt = T*sf+1;

% initial samples
R = zeros(3,3,Ns,Nt);
x = zeros(3,Ns,Nt);

S = diag([4,4,4]);
R(:,:,:,1) = pdf_MF_sampling(S,Ns);

Miu = [0;0;0.2];
Sigma = 0.2^2*eye(3);
x(:,:,1) = mvnrnd(Miu,Sigma,Ns)';

% noise
H1 = diag([0.1,0.1,0.1]);
H2 = diag([0.05,0.05,0.05]);

% simulate
for nt = 1:Nt-1
    gyro_rw = mvnrnd([0,0,0],H1*H1'*sf,Ns)';
    bias_rw = mvnrnd([0,0,0],H2*H2'*sf,Ns)';
    
    R(:,:,:,nt+1) = mulRot(R(:,:,:,nt),expRot((x(:,:,nt)+gyro_rw)/sf));
    x(:,:,nt+1) = x(:,:,nt)+bias_rw/sf;
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
        MFG.P(:,:,nt)*(trace(MFG.S(:,:,nt)*eye(3))-MFG.S(:,:,nt))*MFG.P(:,:,nt).';
end

rmpath('../rotation3d');
rmpath('../matrix Fisher');

end

