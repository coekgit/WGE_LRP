function [Z_t,Q_t,N_t,obj] = nnlrp(X,Projection_Num,option)

warning('off','all')

iter = option.iter;
threshold= option.threshold;
mu = option.mu;
lambda1 = option.lambda1;
lambda2 = option.lambda2;
lambda3 = option.lambda3;
% lambda4 = option.lambda4;
max_mu = option.max_mu;
dim = size(X,1);
num = size(X,2);




%%
% H_t = rand(dim,Projection_Num);  
% Z_t = eye(num,num);
% U_t = eye(num,num);
options = [];
options.ReducedDim = Projection_Num;
[P_t,~] = PCA1(X', options);
% H_t = max(0,H_t);

options = [];
options.NeighborMode = 'KNN';
options.k = 0;
options.WeightMode = 'Cosine';
W = constructW(X',options);
W = W + eye(size(W));
W = W./sum(W,2);
D = diag(sum(W,2));

Q_t =P_t;
Z_t = W;
U_t = W;
B_t = eye(num);
N_t = B_t;
V_t = N_t*Z_t;
% Y_t = rand(Projection_Num,num);
% P_t = rand(dim,Projection_Num);


C_1_t = zeros(dim,num);
C_2_t = zeros(num,num);
C_3_t = zeros(num,num);
C_4_t = zeros(num,num);

obj=zeros(1,iter);

for t = 1:iter    
    t
%     N_t
    Z_t1 = inv(N_t'*N_t+eye(num))*(N_t'*V_t+U_t+1/mu*(N_t'*C_2_t-C_3_t));
    U_t1 = SolveMatrix(lambda1,mu,Z_t1+C_3_t/mu,'nuclear');
    N_t1 = (V_t*Z_t1'+B_t+1/mu*(C_2_t*Z_t1'+C_4_t))*inv(Z_t1*Z_t1'+eye(num));
    B_t1 = SolveMatrix(lambda2,mu,N_t1+C_4_t/mu,'1');
    B_t1 = max(0,B_t1);
    
    
%     norm(Q_t)
    
    S = X*W*X'*Q_t+X*V_t'*X'*Q_t-1/mu*C_1_t*V_t'*X'*Q_t;
    [U,~,V] = svd(S,'econ');
    P_t1 = U*V'; 
    V_t1 = inv(X'*Q_t*Q_t'*X+eye(num))*(X'*Q_t*P_t1'*X-N_t1*Z_t1+1/mu*(X'*Q_t*P_t1'*C_1_t+C_2_t));
    Q_t1 = inv(X*D*X'+X*V_t1*V_t1'*X')*(X*W'*X'*P_t1+X*V_t1*X'*P_t1-1/mu*X*V_t1*C_1_t'*P_t1);
    a = [norm(Z_t)
    norm(U_t)
    norm(N_t)
    norm(B_t)
    norm(P_t)
    norm(Q_t)
    norm(V_t)]

    C_1_t1 = C_1_t + mu *(X - P_t1*Q_t1'*X*V_t1);
    C_2_t1 = C_2_t + mu *(V_t1-N_t1*Z_t1);
    C_3_t1 = C_3_t + mu *(Z_t1-U_t1);
    C_4_t1 = C_4_t + mu *(N_t1-B_t1);
    
    W = B_t1;
    D = diag(sum(W,2));
    

%     N_t1 = N_t1 - diag(diag(N_t1));
 
%     obj(t) = (trace(X*D*X'-2*X*W*X'*H_t1*P_t1')+trace(H_t1'*X*(D+lambda4*eye(size(D)))*X'*H_t1)+lambda2*norm(H_t1,'fro')^2+lambda1*sum(ss)+lambda3*sum(sqrt(sum(E_t1.*E_t1,2)+eps))-2*lambda4*trace(H_t1'*X*Y_t1')+lambda4*trace(Y_t1*Y_t1'))/norm(X,'fro');  
    obj(t) = lambda1*norm(U_t1)+lambda2*norm(N_t1,1)+lambda3*norm(Q_t1,2)-2*trace(X*W*X'*Q_t1*P_t1')+trace(Q_t1'*X*D*X'*Q_t1);
%     fprintf('第%d次实验rho为：%f,obj为：%f\n',t,rho, obj(t))

    mu = min(1.01*mu,max_mu);
    U_t = U_t1;
    B_t = B_t1;
    Z_t = Z_t1;
    P_t = P_t1;
    Q_t = Q_t1;
    V_t = V_t1;
    N_t = N_t1;
    C_1_t = C_1_t1;
    C_2_t = C_2_t1;
    C_3_t = C_3_t1;
    C_4_t = C_4_t1;


end

end