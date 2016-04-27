load('inputs.mat');
[y1,X1]=learnKmeans(y_train,X_train_cnn_std);
[y,X]=solveImbalaceData(y1,X1,'RandomOverSampling');
gamma=0.06;
C=[17.78,17.78,17.78,17.78];
% Initialize Predictors Structures
beta0=zeros(1,4);
alpha=zeros(size(X,1),4);
% Create Data Set for 1vsRest, 2vsRest,3vsRest and 4vsRest
yTrBinSch=splitClass(y,-1);   
Kern=rbfKernel(X,X,gamma);
% Compute Optimal Parameters
for i=1:4
    [alpha(:,i),beta0(i)]=SMO(Kern,yTrBinSch(:,i),C(i));
end
yPredTrain=multiClassPredSVM(yTrBinSch,X,X,alpha,beta0,'rbfKernel',gamma,1);
berErrorCVtrMulti=ber(y,yPredTrain,'Multiclass')

load('inputs_result.mat');
load('test.mat');
X_test_cnn=test.X_cnn;
[~,XTe]=featureEngineering(y,X_test_cnn,1,0,'None','CNN');        
% Standartize Training Set
X_test_cnn_std=standartscore(XTe);
yPred=multiClassPredSVM(yTrBinSch,X,X_test_cnn_std,alpha,beta0,'rbfKernel',0.06,1);

load('train.mat');
X_train_cnn=train.X_cnn;
%[~,XTr]=featureEngineering(y,X_train_cnn,1,0,'None','CNN');        
% Standartize Training Set
%X_train_cnn_std=standartscore(XTr);
yPredTrain=multiClassPredSVM(yTrBinSch,X,X_train_cnn_std,alpha,beta0,'rbfKernel',0.06,1);
berErrorCVtrMulti=ber(train.y,yPredTrain,'Multiclass')
save('inputs_result.mat','X_train_cnn_std','X_test_cnn_std','y_train');

