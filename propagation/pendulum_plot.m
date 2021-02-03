function [  ] = pendulum_plot( varargin )

close all;

addpath('..\rotation3d');

if (size(varargin{1},1)==3 && size(varargin{1},2)==3)
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
        
        % 3d interpolation
        d_threshold = 0.4;
        
        try
            ind_R = load(strcat(path,'\ind_R'));
            ind_R = ind_R.ind_R;
            v_R = load(strcat(path,'\v_R'));
            v_R = v_R.v_R;
        catch
            [ind_R,v_R] = interpInd(R,s1,s2,s3,theta,d_threshold);
            save(strcat(path,'\ind_R'),'ind_R');
            save(strcat(path,'\v_R'),'v_R');
        end
        
        parfor nt = 1:Nt
            f = load(strcat(path,'\f',num2str(nt)));
            f = f.f;
            
            if ~isa(f,'double')
                f = double(f);
            end
            
            fR = sum(f,[4,5,6])*(L/(2*Bx))^3;
            fR = reshape(fR,1,[]);
            
            c = zeros(Nt1,Nt2,3);
            for nt1 = 1:Nt1
                for nt2 = 1:Nt2
                    for i = 1:3
                        for ntheta = 1:Ntheta
                            v = v_R{nt1,nt2,ntheta,i};
                            ind = ind_R{nt1,nt2,ntheta,i};
                            if size(v,1) > 1
                                f_interp = scatteredInterpolant(v',fR(ind_R{nt1,nt2,ntheta,i})');
                                c(nt1,nt2,i) = c(nt1,nt2,i) + f_interp(zeros(1,size(v,1)));
                            else
                                [v,I] = sort(v);
                                ind = ind(I);
                                c(nt1,nt2,i) = c(nt1,nt2,i) + interp1(v,fR(ind),0);
                            end
                        end
                    end
                end
            end
            
            c = c*(2*pi/Ntheta);
            
            f = figure;
            surf(s1,s2,s3,sum(c,3),'LineStyle','none','FaceColor','interp');
            
            xlim([-1,1]);
            ylim([-1,1]);
            zlim([-1,1]);
            view([1,-1,0]);
            axis equal;
            
            annotation('textbox','String',strcat('time: ',num2str((nt-1)/100),' s'),'Position',[0.15,0.75,0.16,0.07]);

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


function [ ind_R, v_R ] = interpInd( R, s1, s2, s3, theta, d )

Nt1 = size(s1,1);
Nt2 = size(s1,2);
Ntheta = size(theta,2);

ind_R = cell(Nt1,Nt2,Ntheta,3);
v_R = cell(Nt1,Nt2,Ntheta,3);

for i = 1:3
    vref = [0;0;0];
    vref(i) = 1;

    for nt1 = 1:Nt1
        for nt2 = 1:Nt2
            rref = [s1(nt1,nt2);s2(nt1,nt2);s3(nt1,nt2)];

            Rref = eye(3);
            jk = setdiff(1:3,i);
            Rref(:,i) = rref;
            Rref(:,jk) = null(rref');
            D = eye(3);
            D(jk(1),jk(1)) = det(Rref);
            Rref = Rref*D;
            Rref = mulRot(Rref,expRot(vref.*theta));

            for ntheta = 1:Ntheta
                v = logRot(mulRot(Rref(:,:,ntheta)',R),'v');
                ind_R{nt1,nt2,ntheta,i} = find(sqrt(sum(v.^2)) < d);
                v = v(:,ind_R{nt1,nt2,ntheta,i});
                            
                v_isunique = false(3,1);
                for j = 1:3
                    v_isunique(j) = max(v(j,:))-min(v(j,:)) < 1e-14;
                end

                v(v_isunique,:) = [];
                v_R{nt1,nt2,ntheta,i} = v;
            end
        end
    end
end

end

