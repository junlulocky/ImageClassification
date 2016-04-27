%% Learning Logistic Regression OneVsAll
function [berErrorTrMulti,berErrorTeMulti,berErrorTrBin,berErrorTeBin]=learnLogisticRegressionOnevsAll(y,X,Basis,Deg)
%% Create Training and Test Sets
% Create Sets
[~,D]=size(X);
% Number of Classes
Class=length(unique(y));
%% Create Training Parameters
% Number of Folds
K=4;
% Create Waitbar
h = waitbar(0,'Please wait...');

%% Create Error, Seed and Predictors Structures 
% Seed Array
seed=randsample(1000,25);
% Initialize Error Structures
berErrorCVtrMulti=zeros(K,1);
berErrorCVteMulti=zeros(K,1);
meanberErrorCVtrMulti=zeros(length(seed),1);
meanberErrorCVteMulti=zeros(length(seed),1);
berErrorCVtrBin=zeros(K,1);
berErrorCVteBin=zeros(K,1);
meanberErrorCVtrBin=zeros(length(seed),1);
meanberErrorCVteBin=zeros(length(seed),1);

% Initialize Predictors Structures
beta=zeros(D,Class);

%% Run Algorithm
for s=1:length(seed)

    % Set Seed
    setSeed(s);
    % Data Set    
    idx=randperm(size(X,1));
    y=y(idx);
    X(:,:)=X(idx,:);
    
    % split data in K fold 
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
 
    % Choose Basis Function
    switch Basis
        case 'PolyBasis'
            X=polybasis(X,Deg);
        case 'GaussianBasis'
            X=gaussianBasis(X);
    end
    
    % K-fold cross validation
    for k = 1:K
        % get k'th subgroup in test, others in train
        idxTe = idxCV(k,:);
        idxTr = idxCV([1:k-1 k+1:end],:);
        idxTr = idxTr(:);
        yTe = y(idxTe);
        XTe = X(idxTe,:);
        yTr = y(idxTr);
        XTr = X(idxTr,:);
       
        
        % Create Data Set for 1vsRest, 2vsRest,3vsRest and 4vsRest
        yTrBinSch=splitClass(yTr,0); 
        % Compute Optimal Parameters 
        for i=1:Class               
            beta(:,i)=logisticRegression(yTrBinSch(:,i), XTr);
        end
        % Class Prediction for Multiclass and Binary
        % Sample Cross Validation Error
        yPred=multiClassPredLogistReg(XTr,beta);
        berErrorCVtrMulti(k)=ber(yTr,yPred,'Multiclass');
        berErrorCVtrBin(k)=ber(yTr,yPred,'Binary');
        yPred=multiClassPredLogistReg(XTe,beta);
        berErrorCVteMulti(k)=ber(yTe,yPred,'Multiclass');
        berErrorCVteBin(k)=ber(yTe,yPred,'Binary');
        
    end
    % Cross Validation Error for each seed
    meanberErrorCVtrMulti(s)=mean(berErrorCVtrMulti);
    meanberErrorCVteMulti(s)=mean(berErrorCVteMulti);
    meanberErrorCVtrBin(s)=mean(berErrorCVtrBin);
    meanberErrorCVteBin(s)=mean(berErrorCVteBin);
    
    % Update Waitbar
    waitbar(s / length(seed))
end    
% Close Waitbar
close(h);    

%% Compute Test and Training Error
display(meanberErrorCVtrMulti);
display(meanberErrorCVteMulti);
display(meanberErrorCVtrBin);
display(meanberErrorCVteBin);
berErrorTrMulti=mean(meanberErrorCVtrMulti);
berErrorTeMulti=mean(meanberErrorCVteMulti);
berErrorTrBin=mean(meanberErrorCVtrBin);
berErrorTeBin=mean(meanberErrorCVteBin);
end
