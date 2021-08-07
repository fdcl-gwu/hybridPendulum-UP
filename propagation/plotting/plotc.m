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

%% get a common normalization for c
% integration weight
w = repmat(sin(theta2),Nt1,1);
    
try
    load(strcat(path,'c_normal.mat'),'c_normal');
catch
    % recover normalization for each c
    fR_max = zeros(Nt,1);
    for nt = 1:Nt
        load(strcat(path,'/c',num2str(nt)),'c');

        cp(:,:,1) = 0.5*(1+c(:,:,1)-c(:,:,2)-c(:,:,3));
        cp(:,:,2) = 0.5*(1+c(:,:,2)-c(:,:,1)-c(:,:,3));
        cp(:,:,3) = 0.5*(1+c(:,:,3)-c(:,:,1)-c(:,:,2));

        int_cp = mean(sum(cp.*w,[1,2]));
        fR_max(nt) = 1/int_cp;
    end

    % the common normalization
    c_normal = max(fR_max);
    save(strcat(path,'c_normal.mat'),'c_normal');
end

%% plot
for nt = 1:Nt
    load(strcat(path,'/c',num2str(nt)),'c');
    
    % normalize
    cp(:,:,1) = 0.5*(1+c(:,:,1)-c(:,:,2)-c(:,:,3));
    cp(:,:,2) = 0.5*(1+c(:,:,2)-c(:,:,1)-c(:,:,3));
    cp(:,:,3) = 0.5*(1+c(:,:,3)-c(:,:,1)-c(:,:,2));
    
    int_cp = mean(sum(cp.*w,[1,2]));
    fR_max(nt) = 1/int_cp;
    cp = cp*fR_max(nt)/c_normal;
    
    c(:,:,1) = 1-cp(:,:,2)-cp(:,:,3);
    c(:,:,2) = 1-cp(:,:,1)-cp(:,:,3);
    c(:,:,3) = 1-cp(:,:,1)-cp(:,:,2);
    
    % make the color more apparent
    c(c<0) = 0;
    c = c.^2.5;
    
    % plot
    f = figure;
    hold on;
    h_surf = surf(s1,s2,s3,c,'LineStyle','none','FaceColor','interp',...
        'FaceAlpha',0.8,'FaceLighting','gouraud');
    myarrow([0,0,0],[1.2,0,0],'k',1,0.1,0.1*tand(15));
    myarrow([0,0,0],[0,1.2,0],'k',1,0.1,0.1*tand(15));
    myarrow([0,0,0],[0,0,1.2],'k',1,0.1,0.1*tand(15));
    
    h_light=light;
    material dull;
    h_light.Style='infinite';
    h_light.Position=[-1 0 1]*10;
    
    set(h_surf,'FaceLighting','gouraud','AmbientStrength',0.95,...
        'DiffuseStrength',0.05,'SpecularStrength',0,'BackFaceLighting','unlit');

    xlim([-1.2,1.2]);
    ylim([-1.2,1.2]);
    zlim([-1.2,1.2]);
    view([-0.1,1,-0.3]);
    axis equal;
    axis off;

    annotation('textbox','String',strcat('time: ',num2str((nt-1)/sf),' s'),...
        'Position',[0.15,0.75,0.4,0.07],'LineStyle','none');
    annotation('textbox','String','$\vec{e}_1$','Interpreter','latex',...
        'Position',[0.23,0.44,0.06,0.06],'LineStyle','none');
    annotation('textbox','String','$\vec{e}_3$','Interpreter','latex',...
        'Position',[0.49,0.78,0.06,0.06],'LineStyle','none');
    
    title('Marginal attitude density');

    M(nt) = getframe(f);
    close(f);
end

% generate video
v = VideoWriter(strcat(path,'/R_continuous.avi'));

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

