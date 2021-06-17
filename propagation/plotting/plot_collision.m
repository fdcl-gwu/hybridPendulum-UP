function [  ] = plot_collision( path, sf, slow )

% theta0
dWall = 0.15;
h = 0.2;
r = 0.05;

theta0 = asin(dWall/sqrt(h^2+r^2)) - asin(r/sqrt(h^2+r^2));

% spherical grid
Nt1 = 100;
Nt2 = 50;
theta1 = linspace(-pi,pi,Nt1);
theta2 = linspace(0,pi,Nt2);
s1 = cos(theta1)'.*sin(theta2);
s2 = sin(theta1)'.*sin(theta2);
s3 = repmat(cos(theta2),Nt1,1);

% wall grid
[yWall,zWall] = meshgrid(-1:1,-1:1);
xWall = sin(theta0)*ones(size(yWall));

% count number of piece of data
files = dir(path);
Nt = 0;
for i = 1:length(files)
    if strcmp(files(i).name(1),'c')
        ind_dot = strfind(files(i).name,'.');
        nt = str2double(files(i).name(2:ind_dot-1));
        if nt > Nt
            Nt = nt;
        end
    end
end

% plot
for nt = 1:Nt
    load(strcat(path,'/c',num2str(nt)),'c');

    f = figure('color','w');
    hold on;
    h_surf = surf(s1,s2,s3,c,'LineStyle','none','FaceColor','interp',...
        'FaceAlpha',0.8,'FaceLighting','gouraud');
    surf(xWall,yWall,zWall,'LineStyle','none','FaceColor',[0,0,0],...
        'FaceAlpha',0.3);
    myarrow([0,0,0],[1.2,0,0],'k',1,0.1,0.1*tand(15));
    myarrow([0,0,0],[0,-1.2,0],'k',1,0.1,0.1*tand(15));
    myarrow([0,0,0],[0,0,-1.2],'k',1,0.1,0.1*tand(15));
    
    h_light=light;
    material dull;
    h_light.Style='infinite';
    h_light.Position=[-1 0 1]*10;
    
    set(h_surf,'FaceLighting','gouraud','AmbientStrength',0.95,...
        'DiffuseStrength',0.05,'SpecularStrength',0,'BackFaceLighting','unlit');

    xlim([-1.2,1.2]);
    ylim([-1.2,1.2]);
    zlim([-1.2,1.2]);
    view([-0.1,-1,-0.3]);
    axis equal;
    axis off;

    annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),...
        'Position',[0.15,0.75,0.4,0.07],'LineStyle','none');

    M(nt) = getframe(f);
    close(f);
end

% generate video
v = VideoWriter(strcat(path,'/R_collide2.avi'));

if exist('slow','var')
    v.FrameRate = sf/slow;
else
    v.FrameRate = sf;
end
v.Quality = 100;
open(v);
writeVideo(v,M);
close(v);

end

