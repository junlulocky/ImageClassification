%% Learning Logistic Regression
function [berErrorTr,berErrorTe]=learnLogisticRegression(y,X,Basis,Deg)
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

%% Create Error and Seed Structures 
% Seed Array
seed=randsample(1000,25);
% Initialize Error Structures
berErrorCVtr=zeros(K,1);
berErrorCVte=zeros(K,1);
meanberErrorCVtr=zeros(length(seed),1);
meanberErrorCVte=zeros(length(seed),1);

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
        yTrBinSchTr=yTrBinSch(:,Class);
        yTrBinSch=splitClass(yTe,0);  
        yTrBinSchTe=yTrBinSch(:,Class);
       
        beta = logisticRegression(yTrBinSchTr, XTr);  

        % Sample Cross Validation Error
        yPred=binClassPredLogistReg(XTr,beta);
        berErrorCVtr(k)=ber(yTrBinSchTr,yPred);
        yPred=binClassPredLogistReg(XTe,beta);
        berErrorCVte(k)=ber(yTrBinSchTe,yPred);
               
    end
    % Cross Validation Error for each seed
    meanberErrorCVtr(s)=mean(berErrorCVtr);
    meanberErrorCVte(s)=mean(berErrorCVte);

    assignin('base','berErrorCVtr',berErrorCVtr);
    assignin('base','berErrorCVte',berErrorCVte);
    
    % Update Waitbar
    waitbar(s / length(seed))
end    
% Close Waitbar
close(h);    

%% Compute Test and Training Error
berErrorTr=meanberErrorCVtr;
berErrorTe=meanberErrorCVte;

end