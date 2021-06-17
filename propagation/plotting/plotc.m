function [  ] = plotc( path, sf, slow )

% spherical grid
Nt1 = 100;
Nt2 = 50;
theta1 = linspace(-pi,pi,Nt1);
theta2 = linspace(0,pi,Nt2);
s1 = cos(theta1)'.*sin(theta2);
s2 = sin(theta1)'.*sin(theta2);
s3 = repmat(cos(theta2),Nt1,1);

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
    
    f = figure;
    surf(s1,s2,s3,c,'LineStyle','none','FaceColor','interp');

    xlim([-1,1]);
    ylim([-1,1]);
    zlim([-1,1]);
    view([1,-1,0]);
    axis equal;

    annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),'Position',[0.15,0.75,0.16,0.07]);

    M(nt) = getframe;
    close(f);
end

% generate video
v = VideoWriter(strcat(path,'/R1.avi'));

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

