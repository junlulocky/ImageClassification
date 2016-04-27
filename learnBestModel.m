%% Compute Predictions by Learning the Best Model
function [yPredBinClass,yPredMultiClass]=learnBestModel(yTrain,XTrain,XTest,V) 
%% Initialization Details
% Number of Classes
Class=length(unique(yTrain));

%% Create HyperParameters Structures 
% HyperParameters Array
gamma=0.06;
C=[17.78,17.78,17.78,17.78];

%% Run Algorithm
% Perform Feature Extraction on the Training Set
[yTrain,XTrain,~]=featureEngineering(yTrain,XTrain,0,1,'RandomUnderSampling','CNN');
% Project XTest in XTraining Basis
XTest=XTest*V;
% Standartize Training Set
XTrain=standartscore(XTrain);
XTest=standartscore(XTest);
% Initialize Predictors Structures
beta0=zeros(1,Class);
alpha=zeros(size(XTrain,1),Class);
% Create Data Set for 1vsRest, 2vsRest,3vsRest and 4vsRest
yTrBinSch=splitClass(yTrain,-1);       
% Compute Kernel
Kern=rbfKernel(XTrain, XTrain,gamma);
% Compute Optimal Parameters
for i=1:Class
    [alpha(:,i),beta0(i)]=SMO(Kern,yTrBinSch(:,i),C(i));
end
% Class Prediction
% Compute Sample Test Error
yPredMultiClass=multiClassPredSVM(yTrBinSch,XTrain,XTest,alpha,beta0,'rbfKernel',gamma,1);
yPredBinClass=size(yPredMultiClass');
yPredBinClass(yPredMultiClass~=4)=1;
yPredBinClass(yPredMultiClass==4)=0;
end