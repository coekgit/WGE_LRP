function [acc] = my_KNN(X,X_label,Y,Y_label)

Train_Maa = X;
Test_Maa = Y;
Train_Lab = X_label;
Test_Lab = Y_label;

Train_Maa = Train_Maa./repmat(sqrt(sum(Train_Maa.^2)),[size(Train_Maa,1) 1]);
Test_Maa  = Test_Maa./repmat(sqrt(sum(Test_Maa.^2)),[size(Test_Maa,1) 1]);    

Mdl2= fitcknn(Train_Maa', Train_Lab,'Distance','euclidean','NumNeighbors',1);
[class_test] = predict(Mdl2,Test_Maa'); 

acc = sum(Test_Lab == class_test)/length(Test_Lab)*100;


end