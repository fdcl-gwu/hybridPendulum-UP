function [ R, x, Sigma ] = pendulum_MEKF_reduced( )

addpath('../matrix Fisher/','../rotation3d/');

% parameters
J = diag([0.01436,0.01436,0.0003326]);
rho = 0.1;
m = 1.0642;
g = 9.8;

b = [0.2;0.2;0];
H = eye(2)*1;

% scaled parameters
tscale = sqrt(J(1,1)/(m*g*rho));
Jt = J/J(1,1);
bt = b*tscale;
Ht = H*tscale^(3/2);

% time
sf = 400;
T = 8;
Nt = T*sf+1;

% scaled time
dtt = 1/sf/tscale;

Q = [zeros(3),zeros(3,2);zeros(2,3),Ht*Ht'];

% initial conditions
S = diag([15,15,15]);
U = expRot([0,-2*pi/3,0]);
Miu = [0;0]*tscale;
xSigma = (2*tscale)^2*eye(2);

Sigma0 = zeros(5,5);
Sigma0(1:3,1:3) = MF2Gauss(U,S,eye(3));
Sigma0(4:5,4:5) = xSigma;

% pre-allocate memory
R = zeros(3,3,Nt);
x = zeros(2,Nt);
Sigma = zeros(5,5,Nt);

R(:,:,1) = U;
x(:,1) = Miu;
Sigma(:,:,1) = Sigma0;

% propagate
for nt = 2:Nt
    % mean value propagation
    [R(:,:,nt),x(:,nt)] = LGVI(R(:,:,nt-1),x(:,nt-1),dtt,Jt,bt);

    % uncertainty propagation
    Sigma(:,:,nt) = Sigma_RK4(Sigma(:,:,nt-1),R(:,:,nt-1),x(:,nt-1),...
        R(:,:,nt), x(:,nt), bt, Jt, Q, dtt );
end

rmpath('../matrix Fisher/','../rotation3d/');

end


function [ Cov ] = MF2Gauss( U, S, V )

R0 = U*V';

Ns = 1000000;
R = pdf_MF_sampling_gpu(U*S*V',Ns);
R = gather(R);

dr = logRot(mulRot(R0',R),'v');
Cov = mean(permute(dr,[1,3,2]).*permute(dr,[3,1,2]),3);

end


function [ R, x ] = LGVI( R, x, dt, J, bt )

x = [x;0];

M = [R(3,2,:);-R(3,1,:);0];
M = M - bt.*x;

A = dt*J*x+dt^2/2*M;

% G
normv = @(v) sqrt(sum(v.^2,1));
Gv = @(v) sin(normv(v))/normv(v)*(J*v)+...
    (1-cos(normv(v)))/sum(v.^2,1)*cross(v,J*v);

% Jacobian of G
DGv = @(v) (cos(normv(v))*normv(v)-sin(normv(v)))/normv(v)^3*J*(v*v') +...
    sin(normv(v))/normv(v)*J + (sin(normv(v))*normv(v)-2*(1-cos(normv(v))))/normv(v)^4*cross(v,J*v)*v' +...
    (1-cos(normv(v)))/normv(v)^2*(-hat(J*v)+hat(v)*J);

% initializa Newton method
v = [1e-5;1e-5;0];

% tolerance
epsilon = 1e-7;

% step size
alpha = 1;

% Newton method
n_step = 0;
while normv(A-Gv(v)) >= epsilon
    v = v + alpha*DGv(v)^-1*(A-Gv(v));
    n_step = n_step+1;
end

dR = expRot(v);
R = mulRot(R,dR);
    
M2 = [R(3,2,:);-R(3,1,:);0];
M2 = M2 - bt.*x;
x = J^-1*(dR'*J*x + dt/2*dR'*M + dt/2*M2);
x = x(1:2);

end


function [ Sigma ] = Sigma_RK4( Sigma, R1, x1, R2, x2, bt, J, Q, dtt )

[Rh,xh] = LGVI(R1,x1,dtt/2,J,bt);

JR = @(R) [R(3,3),0,-R(3,1);0,R(3,3),-R(3,2)];

F1 = [-hat([x1;0]),eye(3,2);JR(R1),-diag(bt(1:2))];
K1 = F1*Sigma + Sigma*F1' + Q;

Fh = [-hat([xh;0]),eye(3,2);JR(Rh),-diag(bt(1:2))];
K2 = Fh*(Sigma+dtt/2*K1) + (Sigma+dtt/2*K1)*Fh' + Q;

K3 = Fh*(Sigma+dtt/2*K2) + (Sigma+dtt/2*K2)*Fh' + Q;

F2 = [-hat([x2;0]),eye(3,2);JR(R2),-diag(bt(1:2))];
K4 = F2*(Sigma+dtt*K3) + (Sigma+dtt*K3)*F2' + Q;

Sigma = Sigma + 1/6*dtt*(K1+2*K2+2*K3+K4);

end

