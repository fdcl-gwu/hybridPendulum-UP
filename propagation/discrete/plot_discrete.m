function [  ] = plot_discrete( path )
close all;

%% parameters
d = 1.5;
h = 2;
r = 0.4;

Nt = 10;

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

%% choose an attitude
theta0 = asin(d/sqrt(h^2+r^2)) - asin(r/sqrt(h^2+r^2));
jR = find(beta > theta0,1,'first');
iR = 2;
kR = 3;
theta = beta(jR);

r3 = R(:,3,iR,jR,kR);
PC = (h-r*tan(theta))*r3 + r*sec(theta)*[1;0;0];

vC_normal = zeros(2*Bx,2*Bx);
for ix = 1:2*Bx
    for jx = 1:2*Bx
        omega = R(:,:,iR,jR,kR)*[x(:,ix,jx);0];
        vC = cross(omega,PC);
        vC_normal(ix,jx) = vC'*[1;0;0];
    end
end

%% plot
x1 = x(1,:,1);
x2 = permute(x(2,1,:),[1,3,2]);

M(Nt) = struct('cdata',[],'colormap',[]);
for nt = 1:Nt+1
    data = load(strcat(path,'/f',num2str(nt)));
    f = data.f;
    
    f_omega = permute(f(iR,jR,kR,:,:),[5,4,1,2,3]);
    fig = figure; hold on;
    surf(x1,x2,f_omega);
    surf(x1,x2,vC_normal','faceColor','r');
    
    view([60,45]);
    annotation('textbox','String','red: velocity into wall','Position',[0.23,0.68,0.26,0.064]);
    xlabel('$\Omega_1$','Interpreter','latex');
    ylabel('$\Omega_2$','Interpreter','latex');
    zlabel('Conditional density');
    
    M(nt) = getframe(fig);
end

v = VideoWriter('x2_noise.avi');
v.FrameRate = 10;
open(v);
writeVideo(v,M);
close(v);

end

