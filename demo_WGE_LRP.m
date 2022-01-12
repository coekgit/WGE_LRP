close all;
clear all;
clc;
tic
name = 'AR_32x32.mat';
load (name);
fea = double(fea);
sele_num = 4;
Eigen_NUM=105;
Projection_Num =100;

%ADMM option
option.iter = 200;
option.threshold = 0.001;
option.rho = 1;
option.alpha = 1;
option.lambda1 = 1e3;
option.lambda2 = 1e-5;
option.lambda3 = 1e3;
option.lambda4 = 1e-4;
option.max_rho = 1e8;


nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];
for i = 1:nnClass
  num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end
%%------------------select training samples and test samples--------------%%
Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass    
    idx      = find(gnd==j);
    randIdx  = randperm(num_Class(j));
    Train_Ma = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];            % select select_num samples per class for training
    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  % select remaining samples per class for test
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';                       % transform to a sample per column
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]); 

label = unique(Train_Lab);
Y = bsxfun(@eq, Train_Lab, label');
Y = double(Y)';
X = Train_Ma;


%%
Tr_DAT = Train_Ma;
Tt_DAT = Test_Ma;

[disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,Eigen_NUM);
tr_dat  =  disc_set'*Tr_DAT;
tt_dat  =  disc_set'*Tt_DAT;

tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [Eigen_NUM,1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [Eigen_NUM,1]) );

%%
Train_SET = tr_dat;
Test_SET  = tt_dat;

lambda = 1e-2;
[Z_t,P_t,H_t,obj] = WGE_LRP(Train_SET,Projection_Num,option);
W = H_t';

Train_Maa = W*Train_SET;
Test_Maa  = W*Test_SET;

acc = my_KNN(Train_Maa,Train_Lab,Test_Maa,Test_Lab)
plot(log(obj))
toc
