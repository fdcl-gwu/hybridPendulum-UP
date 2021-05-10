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

% first config
xc1 = d;
yc1 = cos(-pi/2)*rc;
zc1 = sin(-pi/2)*rc;

t3 = sin(thetar)*[1;0;0] + cos(thetar)*[0;yc1;zc1]/rc;
omega = cross([0;0;-1],t3);
alpha = asin(norm(omega));
K = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0]/norm(omega);
R = eye(3) + sin(alpha)*K + (1-cos(alpha))*K^2;

M = R*([xcy(1,:),xcy(2,:);ycy(1,:),ycy(2,:);zcy(1,:),zcy(2,:)]);
xcy1 = [M(1,1:Nt);M(1,Nt+1:2*Nt)];
ycy1 = [M(2,1:Nt);M(2,Nt+1:2*Nt)];
zcy1 = [M(3,1:Nt);M(3,Nt+1:2*Nt)];

surf(xcy1,ycy1,zcy1,'LineStyle','none','FaceColor','y','FaceAlpha',0.5,'EdgeAlpha',0.5);

patch(xcy1(1,:),ycy1(1,:),zcy1(1,:),'y','FaceAlpha',0.5,'EdgeAlpha',0.5);
patch(xcy1(2,:),ycy1(2,:),zcy1(2,:),'y','FaceAlpha',0.5,'EdgeAlpha',0.5);

% second config
xc2 = d;
yc2 = cos(-pi/4)*rc;
zc2 = sin(-pi/4)*rc;

t3 = sin(thetar)*[1;0;0] + cos(thetar)*[0;yc2;zc2]/rc;
omega = cross([0;0;-1],t3);
alpha = asin(norm(omega));
K = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0]/norm(omega);
R = eye(3) + sin(alpha)*K + (1-cos(alpha))*K^2;

M = R*([xcy(1,:),xcy(2,:);ycy(1,:),ycy(2,:);zcy(1,:),zcy(2,:)]);
xcy2 = [M(1,1:Nt);M(1,Nt+1:2*Nt)];
ycy2 = [M(2,1:Nt);M(2,Nt+1:2*Nt)];
zcy2 = [M(3,1:Nt);M(3,Nt+1:2*Nt)];

surf(xcy2,ycy2,zcy2,'LineStyle','none','FaceColor','r','FaceAlpha',0.5,'EdgeAlpha',0.5);

patch(xcy2(1,:),ycy2(1,:),zcy2(1,:),'r','FaceAlpha',0.5,'EdgeAlpha',0.5);
patch(xcy2(2,:),ycy2(2,:),zcy2(2,:),'r','FaceAlpha',0.5,'EdgeAlpha',0.5);

% third config
xc3 = d;
yc3 = cos(-pi/4*3)*rc;
zc3 = sin(-pi/4*3)*rc;

t3 = sin(thetar)*[1;0;0] + cos(thetar)*[0;yc3;zc3]/rc;
omega = cross([0;0;-1],t3);
alpha = asin(norm(omega));
K = [0,-omega(3),omega(2);omega(3),0,-omega(1);-omega(2),omega(1),0]/norm(omega);
R = eye(3) + sin(alpha)*K + (1-cos(alpha))*K^2;

M = R*([xcy(1,:),xcy(2,:);ycy(1,:),ycy(2,:);zcy(1,:),zcy(2,:)]);
xcy3 = [M(1,1:Nt);M(1,Nt+1:2*Nt)];
ycy3 = [M(2,1:Nt);M(2,Nt+1:2*Nt)];
zcy3 = [M(3,1:Nt);M(3,Nt+1:2*Nt)];

surf(xcy3,ycy3,zcy3,'LineStyle','none','FaceColor','g','FaceAlpha',0.5,'EdgeAlpha',0.5);

patch(xcy3(1,:),ycy3(1,:),zcy3(1,:),'g','FaceAlpha',0.5,'EdgeAlpha',0.5);
patch(xcy3(2,:),ycy3(2,:),zcy3(2,:),'g','FaceAlpha',0.5,'EdgeAlpha',0.5);

view([-60,10]);
xticks([]);
yticks([]);
zticks([]);

set(gcf,'Units','Inches','Position',[15,5,4,4]);
set(gcf,'PaperSize',[4,4]);
set(gca,'Units','Inches','Position',[-0.3,0.2,4.52,3.56]);

print('collision_3D','-dpdf','-painters');
