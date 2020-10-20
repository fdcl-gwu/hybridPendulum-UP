function [  ] = pendulum_plot( R )

close all;

figure;

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

v = VideoWriter('R2.avi');
v.FrameRate = 100;
v.Quality = 100;
open(v);
writeVideo(v,M);
close(v);

end

