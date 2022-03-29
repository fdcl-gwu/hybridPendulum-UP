function [ R, x, Sigma ] = pendulum_MEKF_reduced( )

addpath('../matrix Fisher/','../rotation3d/');

% parameters
J = 0.01436;
rho = 0.1;
m = 1.0642;
g = 9.8;

b = [0.2;0.2];
H = eye(2)*1;

% scaled parameters
tscale = sqrt(J/(m*g*rho));
bt = b*tscale;
Ht = H*tscale^(3/2);

% time
sf = 3200*10;
T = 8;
Nt = T*sf+1;

% scaled time
dtt = 1/sf/tscale;

Q = Ht*Ht'*dtt;

% initial conditions
S = diag([15,15,15]);
U = expRot([0,-2*pi/3,0]);
Miu = [0;0]*tscale;
xSigma = (2*tscale)^2*eye(2);

Sigma0 = MFG2Gauss(Miu,xSigma,zeros(2,3),U,S,eye(3));

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
    R(:,:,nt) = R(:,:,nt-1)*expRot([x(:,nt-1);0]*dtt);
    x(:,nt) = x(:,nt-1) + ([R(3,2,nt-1);-R(3,1,nt-1)]...
        - [bt(1)*x(1,nt-1);bt(2)*x(2,nt-1)])*dtt;

    % uncertainty propagation
    F = [expRot([x(:,nt-1);0]*dtt)', dtt*eye(3,2);
        dtt*[R(3,3,nt-1),0,-R(3,1,nt-1);0,R(3,3,nt-1),-R(3,2,nt-1)], eye(2)-dtt*diag(bt)];
    Sigma(:,:,nt) = F*Sigma(:,:,nt)*F' + [zeros(3,5);zeros(2,3),Q];
end

rmpath('../matrix Fisher/','../rotation3d/');

end


function [ Cov ] = MFG2Gauss( Miu, Sigma, P, U, S, V )

R0 = U*V';

Ns = 100000;
[x,R] = pdf_MFG_sampling(Miu,Sigma,P,U,S,V,Ns);

Cov = zeros(3+length(Miu));
for i = 1:Ns
    dr = logRot(R0'*R(:,:,i),'v');
    dx = x(:,i)-Miu;
    Cov = Cov + [dr;dx]*[dr',dx']/Ns;
end

end

