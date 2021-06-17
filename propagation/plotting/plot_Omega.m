function [  ] = plot_Omega( path, L, sf, slow )

% count number of piece of data
files = dir(path);
fnt = zeros(length(files),1);
N = 0;
for i = 1:length(files)
    if strcmp(files(i).name(1),'f') && ~isnan(str2double(files(i).name(2)))
        ind_dot = strfind(files(i).name,'.');
        nt = str2double(files(i).name(2:ind_dot-1));
        N = N+1;
        fnt(N) = nt;
    end
end

fnt(N+1:end) = [];
fnt = sort(fnt);

% grid
load(strcat(path,'/f1'),'f');
BR = size(f,1)/2;
Bx = size(f,4)/2;
x = linspace(-L/2,L/2-L/(2*Bx),2*Bx);
y = x;

% weight
beta = pi/(4*BR)*(2*(0:(2*BR-1))+1);
w = zeros(1,2*BR);
for j = 1:2*BR
    w(j) = 1/(4*BR^3)*sin(beta(j))*sum(1./(2*(0:BR-1)+1).*sin((2*(0:BR-1)+1)*beta(j)));
end

% plot
N = 0;
for nt = fnt'
    try
        load(strcat(path,'/fx',num2str(nt)),'fx');
        save_fx = false;
    catch
        if nt > 1
            load(strcat(path,'/f',num2str(nt)),'f');
        end
        fx = permute(sum(f.*w,[1,2,3]),[4,5,1,2,3]);
        save_fx = true;
    end
    
    fig = figure; hold on;
    surf(x,y,fx,'LineStyle','none');
    zlim([-0.5,4.5]);
    view([1,1,1]);
    
    annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),...
        'Position',[0.13,0.78,0.22,0.07],'LineStyle','none');
    
    N = N+1;
    M(N) = getframe(fig);
    close(fig);
    
    if save_fx
        save(strcat(path,'/fx',num2str(nt)),'fx');
    end
end

% generate video
v = VideoWriter(strcat(path,'/Omega.avi'));

if exist('slow','var')
    v.FrameRate = sf/(fnt(2)-fnt(1))/slow;
else
    v.FrameRate = sf;
end
v.Quality = 100;
open(v);
writeVideo(v,M);
close(v);

end

