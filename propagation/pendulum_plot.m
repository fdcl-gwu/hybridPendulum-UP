function [  ] = pendulum_plot( varargin )

close all;

addpath('..\rotation3d');

if (size(varargin{1},1)==3 && size(varargin{2},1)==3)
    isDensity = false;
    R = varargin{1};
else
    isDensity = true;
    path = varargin{1};
    L = varargin{2};
end

if isDensity
    files = dir(path);
    
    Nt = 0;
    for i = 1:length(files)
        if strcmp(files(i).name(1),'f')
            nt = str2double(files(i).name(2:3));
            if nt > Nt
                Nt = nt;
            end
        end
    end
    
    if Nt > 0
        f = load(strcat(path,'\f',num2str(1)));
        f = f.f;
        
        % SO(3) grid
        BR = size(f,1)/2;
        Bx = size(f,4)/2;
        R = RGrid(BR);
        
        % spherical grid
        Nt1 = 100;
        Nt2 = 50;
        theta1 = linspace(-pi,pi,Nt1);
        theta2 = linspace(0,pi,Nt2);
        s1 = cos(theta1)'.*sin(theta2);
        s2 = sin(theta1)'.*sin(theta2);
        s3 = repmat(cos(theta2),Nt1,1);
        
        % circular grid
        Ntheta = 40;
        theta = linspace(-pi,pi-2*pi/Ntheta,Ntheta);
        
        % 1d integration rule
        d_t = 0.17;
        ind_R = cell(Nt1,Nt2,3);
        theta_R = cell(Nt1,Nt2,3);
        for nt1 = 1:Nt1
            for nt2 = 1:Nt2
                rref = [s1(nt1,nt2);s2(nt1,nt2);s3(nt1,nt2)];
                
                for i = 1:3
                    ind_R{nt1,nt2,i} = find(sqrt(sum((rref-R(:,i,:)).^2))<d_t);
                    
                    if isempty(ind_R{nt1,nt2,i})
                        a = 1;
                    end
                    
                    jk = setdiff(1:3,i);
                    Rref(:,i) = rref;
                    Rref(:,jk) = null(rref');
                    D = eye(3);
                    D(jk(1),jk(1)) = det(Rref);
                    Rref = Rref*D;

                    v = logRot(mulRot(Rref',R(:,:,ind_R{nt1,nt2,i})),'v');
                    theta_R{nt1,nt2,i} = v(i,:);
                    
                    [theta_R{nt1,nt2,i},I] = sort(theta_R{nt1,nt2,i});
                    ind_R{nt1,nt2,i} = ind_R{nt1,nt2,i}(I);
                    
                    [theta_R{nt1,nt2,i},I] = unique(theta_R{nt1,nt2,i});
                    ind_R{nt1,nt2,i} = ind_R{nt1,nt2,i}(I);
                end
            end
        end
        
        for nt = 1:Nt
            if nt > 1
                f = load(strcat(path,'\f',num2str(nt)));
                f = f.f;
            end
            
            fR = sum(f,[4,5,6])*(L/(2*Bx))^3;
            
            c = zeros(Nt1,Nt2,3);
            for nt1 = 1:Nt1
                for nt2 = 1:Nt2
                    for i = 1:3
                        c(nt1,nt2,i) = sum(interp1(theta_R{nt1,nt2,i}',fR(ind_R{nt1,nt2,i}),theta,'nearest','extrap'))*(2*pi/Ntheta);
                    end
                end
            end
            
            f = figure;
            surf(s1,s2,s3,sum(c,3),'LineStyle','none','FaceColor','interp');
            
            xlim([-1,1]);
            ylim([-1,1]);
            zlim([-1,1]);
            view([1,-1,0]);
            axis equal;
            
            annotation('textbox','String',strcat('time: ',num2str((nt-1)/100),' s'),'Position',[0.15,0.85,0.16,0.07]);

            M(nt) = getframe;
            close(f);
        end
    end
else
    Nt = size(R,3);
    for nt = 1:Nt
        f = figure; hold on;
        plot3([0,R(1,1,nt)],[0,R(2,1,nt)],[0,R(3,1,nt)]);
        plot3([0,R(1,2,nt)],[0,R(2,2,nt)],[0,R(3,2,nt)]);
        plot3([0,R(1,3,nt)],[0,R(2,3,nt)],[0,R(3,3,nt)]);

        xlim([-1,1]);
        ylim([-1,1]);
        zlim([-1,1]);
        view([1,1,0]);

        annotation('textbox','String',strcat('time: ',num2str((nt-1)/100),' s'),'Position',[0.15,0.85,0.16,0.07]);

        M(nt) = getframe;
        close(f);
    end
end

v = VideoWriter('R1.avi');
v.FrameRate = 100;
v.Quality = 100;
open(v);
writeVideo(v,M);
close(v);

rmpath('..\rotation3d');

end


function [ R ] = RGrid( BR )

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

R = reshape(R,3,3,(2*BR)^3);

end

