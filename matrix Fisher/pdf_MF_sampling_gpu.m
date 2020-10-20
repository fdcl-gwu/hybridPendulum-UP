function [ R ] = pdf_MF_sampling_gpu( F, N )

[U,S,V]=psvd(F);

% Bingham sampling
B=zeros(4,4);
B(1:3,1:3)=2*S-trace(S)*eye(3);
B(4,4)=trace(S);

lamB=eig(B);
A=-B;
lamA=-lamB;
min_lamA=min(lamA);
lamA=lamA-min_lamA;
A=A-min_lamA*eye(4);

funb = @(b) 1/(b+2*lamA(1))+1/(b+2*lamA(2))+1/(b+2*lamA(3))+1/(b+2*lamA(4))-1;

tol = optimoptions('fsolve','TolFun', 1e-8, 'TolX', 1e-8,'display','off');
[b,err,exitflag]=fsolve(funb,1,tol);
if exitflag ~= 1
    disp([err exitflag]);
end

W=eye(4)+2*A/b;
Mstar=exp(-(4-b)/2)*(4/b)^2;

x=gpuArray.zeros(4,N);
nx=0;
nxi=0;

while nx < N
    xi = mvnrnd(zeros(4,1),inv(W),N-nx)';
    xi = xi./sqrt(sum(xi.^2));
    nxi = nxi+N-nx;
    
    pstar_Bing = exp(-permute(sum(sum(permute(xi,[1,3,2])...
        .*permute(xi,[3,1,2]).*A,1),2),[3,2,1]));
    pstar_ACGD = permute(sum(sum(permute(xi,[1,3,2])...
        .*permute(xi,[3,1,2]).*W,1),2),[3,2,1]).^(-2);
    u = gpuArray.rand(N-nx,1);
    
    valid = find(u < (pstar_Bing ./ (Mstar*pstar_ACGD)));
    x(:,nx+1:nx+length(valid)) = xi(:,valid);
    nx = nx+length(valid);
end

R = permute(x(4,:).^2-sum(x(1:3,:).*x(1:3,:),1),[1,3,2]).*eye(3)...
    + 2*permute(x(1:3,:),[1,3,2]).*permute(x(1:3,:),[3,1,2])...
    + 2*permute(x(4,:),[1,3,2]).*hat(x(1:3,:));
R = mulRot(mulRot(U,R,false),V',false);

end
