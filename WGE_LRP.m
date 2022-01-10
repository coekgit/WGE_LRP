function [Z_t,P_t,H_t,obj] = WGE_LRP(X,Projection_Num,option)
% X = double(NewTrain_DAT);
% y = double(NewTest_DAT(:,1));
% X  =  X./( repmat(sqrt(sum(X.*X)), [size(X,1),1]) );
% y  =  y./( repmat(sqrt(sum(y.*y)), [size(y,1),1]) );


warning('off','all')

iter = option.iter;
threshold= option.threshold;
rho = option.rho;
lambda1 = option.lambda1;
lambda2 = option.lambda2;
lambda3 = option.lambda3;
lambda4 = option.lambda4;
max_rho = option.max_rho;
dim = size(X,1);
num = size(X,2);

options = [];
options.NeighborMode = 'KNN';
options.k = 0;
options.WeightMode = 'Cosine';
W = constructW(X',options);
W = W + eye(size(W));
W = W./sum(W,2);
D = diag(sum(W,2));


%%
% H_t = rand(dim,Projection_Num);  
% Z_t = eye(num,num);
% U_t = eye(num,num);
options = [];
options.ReducedDim = Projection_Num;
[P_t,~] = PCA1(X', options);

H_t = P_t;
% H_t = max(0,H_t);


Z_t = W;
U_t = W;
Y_t = H_t'*X*Z_t;
E_t = zeros(dim,num);
% Y_t = rand(Projection_Num,num);
% P_t = rand(dim,Projection_Num);


C_1_t = zeros(num,num);
C_2_t = zeros(dim,num);
C_3_t = zeros(Projection_Num,num);

obj=zeros(1,iter);

for t = 1:iter    
%     t
%     H_t1 = inv(lambda2*diag(1./sum(H_t'))+2**X*W*X'+rho*X*Z_t*Z_t'*X')*(2**X*W*X'*P_t+rho*X*Z_t*(Y_t-1/rho*C_3_t)');
    H_t1 = (2*lambda2*eye(size(H_t,1))+rho*X*Z_t*Z_t'*X'+X*(D+D')*X'+lambda4*(X*X'))\(rho*X*Z_t*(Y_t-1/rho*C_3_t)'+2*X*W'*X'*P_t+2*lambda4*X*Y_t');
%     H_t1 = max(0,H_t1);
    Z_t1 = (X'*H_t1*H_t1'*X+eye(num))\(X'*H_t1*(Y_t-1/rho*C_3_t)+(U_t-1/rho*C_1_t));
%     Z_t1+1/rho*C_1_t
%     W = Z_t1;
    es = lambda1/rho;
    temp_U = Z_t1+C_1_t/rho;
    [uu,ss,vv] = svd(temp_U,'econ');
    ss = diag(ss);
    SVP = length(find(ss>es));
    if SVP>1
        ss = ss(1:SVP)-es;
    else
        SVP = 1;
        ss = 0;
    end
    U_t1 = uu(:,1:SVP)*diag(ss)*vv(:,1:SVP)';  
%     rank(U_t1)
%     rank(U_t1)
%     E_t1 = inv(lambda3*diag(1./sum(E_t'))+rho*eye(num))*(X'-Y_t'*P_t'+C_2_t');
%     E_t1 = inv(lambda3*1./sqrt(sum(E_t.*E_t,2)+eps)+rho*eye(num))*(X'-Y_t'*P_t'+C_2_t');
    tempV = X - P_t*Y_t + 1/rho*C_2_t;
    E_t1 = solve_l1l2(tempV,lambda3/rho);
    Y_t1 = 0.5*1/(rho+lambda4)*(rho*(P_t'*(X-E_t1+1/rho*C_2_t)+H_t1'*X*Z_t1)+C_3_t+2*lambda4*H_t1'*X);
%     [U,S,V] = svd(2**X*W*X'*H_t1-rho*(X-E_t1'+1/rho*C_2_t)*Y_t1');
    [U,~,V] = svd(2*X*W*X'*H_t1+rho*(X-E_t1+1/rho*C_2_t)*Y_t1','econ');
    P_t1 = U*V';
    C_1_t1 = C_1_t + rho *(Z_t1-U_t1);
    C_2_t1 = C_2_t + rho *(X - P_t1*Y_t1 - E_t1);
    C_3_t1 = C_3_t + rho *(H_t1'*X*Z_t1-Y_t1);
%     rho = min(1.01*rho,rho_max);
    %%
    
%     LL1 = norm(Z_t1-Z_t,'fro');
%     LL2 = norm(U_t1-U_t,'fro');
%     LL3 = norm(Y_t1-Y_t,'fro');
%     LL4 = norm(H_t1-H_t,'fro');
%     LL5 = norm(P_t1-P_t,'fro');
%     SLSL = max(max(max(max(LL1,LL2),LL3),LL4),LL5);
%     if SLSL*rho/norm(X,'fro') < 0.1
        rho = min(1.1*rho,max_rho);
%     end
%     SLSL*rho/norm(X,'fro')
    H_t = H_t1;
    Z_t = Z_t1;
    U_t = U_t1;
    Y_t = Y_t1;
    P_t = P_t1;
    C_1_t = C_1_t1;
    C_2_t = C_2_t1;
    C_3_t = C_3_t1;
    E_t = E_t1;

    obj(t) = (trace(X*D*X'-2*X*W*X'*H_t1*P_t1')+trace(H_t1'*X*(D+lambda4*eye(size(D)))*X'*H_t1)+lambda2*norm(H_t1,'fro')^2+lambda1*sum(ss)+lambda3*sum(sqrt(sum(E_t1.*E_t1,2)+eps))-2*lambda4*trace(H_t1'*X*Y_t1')+lambda4*trace(Y_t1*Y_t1'))/norm(X,'fro');  
end

end