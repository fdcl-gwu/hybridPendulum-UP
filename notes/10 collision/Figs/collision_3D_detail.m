clear;
close all;

figure; hold on;

% wall
d = 1.5;
patch([d;d;d;d],[-2.5;-2.5;2.5;2.5],[-2.5;2.5;2.5;-2.5],'b','FaceColor','b',...
    'EdgeColor','none','FaceAlpha',0.2);
view(3);
axis equal;

% cylinder
h = 2;
r = 0.4;

Nt = 501;
theta = linspace(-pi,pi,Nt);
xcy = repmat(r*cos(theta),2,1);
ycy = repmat(r*sin(theta),2,1);
zcy = [zeros(1,Nt);-h*ones(1,Nt)];

% collision points
pc = sqrt(r^2+h^2);
rc = sqrt(pc^2-d^2);

xrc = ones(1,Nt)*d;
yrc = cos(theta)*rc;
zrc = sin(theta)*rc;

thetar = asin(d/pc) - asin(r/pc);

patch(xrc,yrc,zrc,'k','FaceColor','None');

% third config
xc = d;
yc = cos(-pi/4*3)*rc;
zc = sin(-pi/4*3)*rc;

t3 = sin(thetar)*[1;0;0] + cos(thetar)*[0;yc;zc]/rc;
omega = cross([0;0;-1],t3);
alpha = asin(norm(omega));
K = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0]/norm(omega);
R = eye(3) + sin(alpha)*K + (1-cos(alpha))*K^2;

M = R*([xcy(1,:),xcy(2,:);ycy(1,:),ycy(2,:);zcy(1,:),zcy(2,:)]);
xcy = [M(1,1:Nt);M(1,Nt+1:2*Nt)];
ycy = [M(2,1:Nt);M(2,Nt+1:2*Nt)];
zcy = [M(3,1:Nt);M(3,Nt+1:2*Nt)];

surf(xcy,ycy,zcy,'LineStyle','none','FaceColor','g','FaceAlpha',0.2);

patch(xcy(1,:),ycy(1,:),zcy(1,:),'g','FaceAlpha',0.2);
patch(xcy(2,:),ycy(2,:),zcy(2,:),'g','FaceAlpha',0.2);

% lines
plot3([0,d],[0,0],[0,0],'k');
plot3([0,t3(1)*h],[0,t3(2)*h],[0,t3(3)*h],'k');
plot3([d,xc],[0,yc],[0,zc],'k');
plot3([xc,t3(1)*h],[yc,t3(2)*h],[zc,t3(3)*h]);

xt(1) = xc;
yt(1) = yc+zc;
zt(1) = zc-yc;

xt(2) = xc;
yt(2) = yc-zc;
zt(2) = zc+yc;

plot3(xt,yt,zt,'k--');

view([-60,10]);
xticks([]);
yticks([]);
zticks([]);

set(gcf,'Units','Inches','Position',[15,5,4,4]);
set(gcf,'PaperSize',[4,4]);
set(gca,'Units','Inches','Position',[-0.3,0.2,4.52,3.56]);

print('collision_3D_detail','-dpdf','-painters');
