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

theta = asin(s1);
c_theta = zeros(Nt1,Nt2,3);
for nt1 = 1:Nt1
    for nt2 = 1:Nt2
        if theta(nt1,nt2) > theta0
            c_theta(nt1,nt2,:) = [0,0,0];
        else
            c_theta(nt1,nt2,:) = [1,1,1];
        end
    end
end

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
    c = load(strcat(path,'/c',num2str(nt)));
    c = c.c;
    
    f = figure; hold on;
    surf(s1,s2,s3,c_theta,'LineStyle','none');
    surf(s1*1.001,s2*1.001,s3*1.001,c,'LineStyle','none','FaceColor','interp',...
        'FaceAlpha',0.5);

    xlim([-1,1]);
    ylim([-1,1]);
    zlim([-1,1]);
    view([0,-1,-1]);
    axis equal;

    annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),'Position',[0.15,0.75,0.16,0.07]);

    M(nt) = getframe;
    close(f);
end

% generate video
v = VideoWriter(strcat(path,'/R_collide.avi'));

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

