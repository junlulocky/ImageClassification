%% Learning Estimation of the Distribution of the Test Error
function []=learnDistribTestError(y,X) 
%% Initialization Details
% Number of Classes
Class=length(unique(y));
% Create Waitbar
h = waitbar(0,'Please wait...');

%% Create Error, Seed, HyperParameters and Predictors Structures 
% Seed Array
seed=randsample(1000,5);
% HyperParameters Array
gamma=0.06;
C=[17.78,17.78,17.78,17.78];
% Initialize Error Structures
MulticlassBerError=zeros(length(seed),1);
BinBerError=zeros(length(seed),1);

%% Run Algorithm
for s=1:length(seed)
    % Set Seed
    setSeed(s);
    % Randomly Pick Training and Validation Set
    [yTrain,yValid,XTrain,XValid]=splitTrainValid(y,X);
    % Perform Feature Extraction on the Training Set
    [yTrain,XTrain]=featureEngineering(yTrain,XTrain,0,1,'RandomOverSampling','CNN');
    % Standartize Training and Validation Set
    XTrain=standartscore(XTrain);
    XValid=standartscore(XValid);

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
    yPred=multiClassPredSVM(yTrBinSch,XTrain,XValid,alpha,beta0,'rbfKernel',gamma,1);
    MulticlassBerError(s)=ber(yValid,yPred,'Multiclass');
    BinBerError(s)=ber(yValid,yPred,'Binary');

    % Update Waitbar
    waitbar(s / length(seed));
end
% Close Waitbar
close(h);  

display(MulticlassBerError);
display(BinBerError);

%% Ilustrate Error Distribution
fig = boxplot([MulticlassBerError, BinBerError]);
h = legend(findall(gca,'Tag','Box'), {'1 Multiclass Classification','2 Binary Classification'});
set(h,'Fontsize',20);
xlabel('Best Model','FontSize', 15);
ylabel('Predicted Test BER','FontSize', 15);

end