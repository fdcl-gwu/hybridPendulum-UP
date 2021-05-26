function [  ] = pendulum_reduced_collision( use_mex, noise, path )

addpath('../../rotation3d');
addpath('../../matrix Fisher');
addpath('../../');

if ~exist('use_mex','var') || isempty(use_mex)
    use_mex = false;
end

if ~exist('noise','var') || isempty(noise)
    noise = false;
end

if use_mex
    addpath('mex/sources/pendulum_reduced_collision/build');
    % addpath('mex');
end

% parameters
d = 1.5;
h = 2;
r = 0.4;

Nt = 10;
dt = 1/100;

J = 0.0152492;
rho = 0.0679878;
m = 1.85480;
g = 9.8;

tscale = sqrt(J/(m*g*rho));

epsilon = 0.7;
Hd = eye(2)*0.05;
Gd = Hd*Hd';

% band limit
BR = 15;
Bx = 15;

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

dx2 = (L/(2*Bx))^2;

%% lambda
lambda_max = 100;
theta_t = 5*pi/180;

r3 = permute(R(:,3,:,:,:),[1,3,4,5,2]);
theta0 = asin(d/sqrt(h^2+r^2)) - asin(r/sqrt(h^2+r^2));
theta = permute(asin(sum(r3.*[1;0;0],1)),[2,3,4,1]);
PC = (h-r*tan(permute(theta,[4,1,2,3]))).*r3 + r*sec(permute(theta,[4,1,2,3])).*[1;0;0];

ind_0 = theta < (theta0 - theta_t);
ind_max = theta > (theta0 + theta_t);
ind_mid = ~ (ind_0 | ind_max);
ind_n0 = find(~ind_0);

if use_mex
    if ~noise
        [lambda,lambda_indR,lambda_indx,PC] = getLambda_mex(R,x,d,h,r,...
            theta_t,lambda_max,true);
    else
        [lambda,lambda_indR,PC] = getLambda_mex(R,x,d,h,r,...
            theta_t,lambda_max,false);
%         [lambda,lambda_indR,lambda_indx,PC] = getLambda_mex(R,x,d,h,r,...
%             theta_t,lambda_max,true);
    end
else
    try 
        data = load('lambda');
        lambda = data.lambda;
    catch
        lambda(ind_0) = 0;
        lambda(ind_max) = lambda_max;
        lambda(ind_mid) = lambda_max/2*sin(pi/(2*theta_t)*(theta(ind_mid)-theta0))...
            + lambda_max/2;
        lambda = reshape(lambda,2*BR,2*BR,2*BR);

        lambda = repmat(lambda,1,1,1,2*Bx,2*Bx);
        for iR = 1:2*BR
            for jR = 1:2*BR
                for kR = 1:2*BR
                    if ind_0(iR,jR,kR)
                        continue;
                    else
                        for ix = 1:2*Bx
                            for jx = 1:2*Bx
                                omega = R(:,:,iR,jR,kR)*[x(:,ix,jx);0];
                                vC = cross(omega,PC(:,iR,jR,kR));
                                if vC'*[1;0;0] < 0
                                    lambda(iR,jR,kR,ix,jx) = 0;
                                end
                            end
                        end
                    end
                end
            end
        end

        save('lambda.mat','lambda');
    end
end

%% Omega_old
if ~noise
    if use_mex
        Omega_old = getOmega_mex(R,x,lambda_indR,epsilon,PC,'old');
    else
        try
            data = load('Omega_old');
            Omega_old = data.Omega_old;
        catch
            Omega_old = zeros(2,length(ind_n0),2*Bx,2*Bx);
            for nR = 1:length(ind_n0)
                indR = ind_n0(nR);
                t = cross(r3(:,indR),[1;0;0]);
                t = t/sqrt(sum(t.^2));

                for ix = 1:2*Bx
                    for jx = 1:2*Bx
                        omega = R(:,:,indR)*[x(1:2,ix,jx);0];
                        omega_old = omega - (1+epsilon)/epsilon*omega'*t*t;
                        vC_old = cross(omega_old,PC(:,indR));

                        if vC_old'*[1;0;0] < 0
                            Omega_old(:,nR,ix,jx) = [nan;nan];
                        else
                            Omega_old3 = R(:,:,indR)'*omega_old;
                            if sum(Omega_old3(1:2) < -L/2+1e-10 | Omega_old3(1:2) > L/2-L/(2*Bx)-1e-10)
                                Omega_old(:,nR,ix,jx) = [nan;nan];
                            else
                                Omega_old(:,nR,ix,jx) = Omega_old3(1:2);
                            end
                        end
                    end
                end
            end

            save('Omega_old.mat','Omega_old','-v7.3');
        end
    end
end

%% Omega_new
if noise
    if use_mex
        Omega_new = getOmega_mex(R,x,lambda_indR,epsilon,PC,'new');
    else
        try
            data = load('Omega_new');
            Omega_new = data.Omega_new;
        catch
            Omega_new = zeros(2,length(ind_n0),2*Bx,2*Bx);
            for nR = 1:length(ind_n0)
                indR = ind_n0(nR);
                t = cross(r3(:,indR),[1;0;0]);
                t = t/sqrt(sum(t.^2));

                for ix = 1:2*Bx
                    for jx = 1:2*Bx
                        omega = R(:,:,indR)*[x(1:2,ix,jx);0];
                        omega_new = omega - (1+epsilon)*omega'*t*t;

                        Omega_new3 = R(:,:,indR)'*omega_new;
                        Omega_new(:,nR,ix,jx) = Omega_new3(1:2);
                    end
                end
            end

            save('Omega_new.mat','Omega_new','-v7.3');
        end
    end
end

c_normal = 1/(2*pi*sqrt(det(Gd)));

%% mex pre-calculation
if use_mex
    if noise
        try
            load(strcat('fcL',num2str(BR),num2str(Bx),'.mat'),'fcL');
            load(strcat('fcL',num2str(BR),num2str(Bx),'.mat'),'fcL_indx');
        catch
            nD = int32(10);
            [fcL,fcL_indx] = getFcL_mex(x,Omega_new,lambda,Gd,nD);
            % [fcL,fcL_indx1,fcL_indx2,fcL_numx2] = getFcL_mex(x,Omega_new,lambda,lambda_indx,Gd);
            save(strcat('fcL',num2str(BR),num2str(Bx),'.mat'),'fcL','fcL_indx','-v7.3');
        end
    else
        [ind_interp,coeff_interp] = getIndRule_mex(x,Omega_old);
    end
end

%% initial conditions
S = diag([15,15,15]);
U = expRot([0,theta0,0]);
Miu = [0;0]*1;
Sigma = (2*tscale)^2*eye(2);

c = pdf_MF_normal(diag(S));

f = permute(exp(sum(U*S.*R,[1,2])),[3,4,5,1,2]).*...
    permute(exp(sum(-0.5*permute((x-Miu),[1,4,2,3]).*permute((x-Miu),...
    [4,1,2,3]).*Sigma^-1,[1,2])),[1,2,5,3,4])/c/sqrt((2*pi)^2*det(Sigma));

if exist('path','var') && ~isempty(path)
    save(strcat(path,'/f1'),'f');
end

%% propagate
for nt = 1:Nt
    tic;
    df = zeros(size(f));
    
    if use_mex
        if ~noise
            df = pendulum_reduced_discrete_propagate(false,f,lambda,...
                lambda_indR,lambda_indx,ind_interp,coeff_interp);
        else
            df = pendulum_reduced_discrete_propagate(true,f,lambda,...
                lambda_indR,fcL,fcL_indx);
        end
    else
        for nR = 1:length(ind_n0)
            indR = ind_n0(nR);
            [iR,jR,kR] = ind2sub([2*BR,2*BR,2*BR],indR);

            if ~noise
                fx = permute(f(iR,jR,kR,:,:),[4,5,1,2,3]);
                fx = fftshift(fftshift(fx,1),2);
                Fx = fft2(fx);
                Fx = fftshift(fftshift(Fx,1),2);

                fx_max = max(fx,[],'all');

                for ix = 1:2*Bx
                    for jx = 1:2*Bx
                        if isnan(Omega_old(1,nR,ix,jx))
                            df(iR,jR,kR,ix,jx) = -lambda(iR,jR,kR,ix,jx)*f(iR,jR,kR,ix,jx);
                        else
                            F_omega1 = exp(2*pi*1i*(-Bx:Bx-1)'*Omega_old(1,nR,ix,jx)/L);
                            F_omega2 = exp(2*pi*1i*(-Bx:Bx-1)*Omega_old(2,nR,ix,jx)/L);
                            f_g = sum(Fx.*F_omega1.*F_omega2,[1,2])/(2*Bx)^2;
                            f_g = real(f_g);

                            if f_g < fx_max/100
                                f_g = 0;
                            end

                            lambda_g = lambda(iR,jR,kR,:,:);
                            lambda_g = lambda_g(find(lambda_g ~= 0,1,'first'));

                            df(iR,jR,kR,ix,jx) = lambda_g*f_g - lambda(iR,jR,kR,ix,jx)*f(iR,jR,kR,ix,jx);
                        end
                    end
                end
            else
                for ix = 1:2*Bx
                    for jx = 1:2*Bx
                        dOmega = x(:,ix,jx) - permute(Omega_new(:,nR,:,:),[1,3,4,2]);
                        f_c = c_normal*exp(-1/2*permute(sum(Gd^-1.*permute(dOmega,...
                            [1,4,2,3]).*permute(dOmega,[4,1,2,3]),[1,2]),[3,4,1,2]));

                        df_in = sum(f_c.*permute(lambda(iR,jR,kR,:,:),[4,5,1,2,3]).*...
                            permute(f(iR,jR,kR,:,:),[4,5,1,2,3]),[1,2])*dx2;
                        df(iR,jR,kR,ix,jx) = df_in - lambda(iR,jR,kR,ix,jx)*f(iR,jR,kR,ix,jx);
                    end
                end
            end
        end
    end
    
    f = f + df*dt;
    
    if exist('path','var') && ~isempty(path)
        save(strcat(path,'/f',num2str(nt+1)),'f');
    end
    toc;
end

rmpath('../../rotation3d');
rmpath('../../matrix Fisher');
rmpath('../../');

if use_mex
    rmpath('mex/sources/pendulum_reduced_collision/build');
end

end

