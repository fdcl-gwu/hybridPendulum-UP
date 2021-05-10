clear;
close all;

addpath('../../rotation3d');

%% parameters
d = 1.5;
h = 2;
r = 0.4;

pc = sqrt(r^2+h^2);
rc = sqrt(pc^2-d^2);

J = diag([0.0152492,0.0152492,0.00380233]);

epsilon = 0.7;

%% collision config
% collision point
PC = [d;cos(pi/4)*rc;sin(pi/4)*rc];

% attitude
thetar = asin(d/pc) - asin(r/pc);
r3 = sin(thetar)*[1;0;0] + cos(thetar)*[0;PC(2);PC(3)]/rc;
omega = cross([0;0;1],r3);
alpha = asin(norm(omega));
K = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0]/norm(omega);
R = eye(3) + sin(alpha)*K + (1-cos(alpha))*K^2;

R = R*expRot([0,0,rand]);

% inertia tensor
I = R*J*R';

% linear velocity at collision point
t = cross(r3,[1;0;0]);
t = t/norm(t);
vC_old = 1*t + 1*cross(t,PC)/pc;

% angular velocity
A = [R(:,3)';0,PC(3),-PC(2);-PC(3),0,PC(1)];
b = [0;vC_old(1);vC_old(2)];
omega_old = A\b;

%% collision response
omega_new = omega_old - (1+epsilon)*omega_old'*t*t;
vC_new = cross(omega_new,PC);

% check law of restitution
dv = vC_new'*[1;0;0] + epsilon*vC_old'*[1;0;0]

% check angular momentum
domega = cross(omega_new-omega_old, I^-1*cross(PC,[1;0;0]))

rmpath('../../rotation3d');

