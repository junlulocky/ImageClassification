%% Learning Penalizes Logistic Regression 1vsAll
function [berErrorTr,berErrorTe]=learnPenLogisticRegression(y_train,X_train)
%% Create Training and Test Sets
% Number of samples per class
Nc=2000;
% Samples of Label 4
setLabel4=find(y_train==4);
% Samples of Label 1,2,3
setLabelnot4=find(y_train~=4);
% Create Sets
[~,D]=size(X_train);

%% Create Training Parameters
% Number of Folds
K=3;
% Create Waitbar
h = waitbar(0,'Please wait...');

%% Create Error, Seed and Lambda Structures 
% Seed Array
seed=randsample(1000,3);
% Lambda Array
lambda=linspace(10^-1,10^4,15);
% Initialize Error Structures
berErrorCVtr=zeros(K,1);
berErrorCVte=zeros(K,1);
meanberErrorCVtr=zeros(length(seed),length(lambda));
meanberErrorCVte=zeros(length(seed),length(lambda));

% Create Waitbar
%h = waitbar(0,'Please wait...');

%% Run Algorithm
for s=1:length(seed)
    % Set Seed
    setSeed(s);
    
    % Create Sets
    X=zeros(2*Nc,D);
    y=zeros(2*Nc,1);
    
    % Create Classes 0 and 1 from the Sets
    idx0=randsample(length(setLabelnot4),Nc);
    idx1=randsample(length(setLabel4),Nc);
    X(1:Nc,:)=X_train(setLabelnot4(idx0),:);
    y(1:Nc)=0;
    X(Nc+1:end,:)=X_train(setLabel4(idx1),:);
    y(Nc+1:end)=1;
    
    % Data Set
    idx=randperm(2*Nc);
    y=y(idx);
    X(:,:)=X(idx,:);
    
    % split data in K fold 
    N = size(y,1);
    idx = randperm(N);
    Nk = floor(N/K);
    for k = 1:K
        idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
    end
    
    % choose for polynomial basis degree
    X=polybasis(X,7);
    
    %% Find Minimum Lambda
    for l=1:length(lambda)
        
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

            beta = penLogisticRegression(yTr, XTr,lambda(l));  

            % Sample Cross Validation Error
            yPred=binClassPredLogistReg(XTr,beta);
            berErrorCVtr(k)=ber(yTr,yPred);
            yPred=binClassPredLogistReg(XTe,beta);
            berErrorCVte(k)=ber(yTe,yPred);

        end
    % Cross Validation Error for each seed
    meanberErrorCVtr(s,l)=mean(berErrorCVtr);
    meanberErrorCVte(s,l)=mean(berErrorCVte);
    
    display(l);
    display(meanberErrorCVtr(s,l));
    display(meanberErrorCVte(s,l));
    
    % Update Waitbar
    waitbar(l / length(lambda))
    end
    
    assignin('base','meanberErrorCVtr',berErrorCVtr);
    assignin('base','meanberErrorCVte',berErrorCVte);
    
end

%% Compute Test and Training Error
[~,index]=min(mean(meanberErrorCVte));
beta=penlogisticRegression(yTr, XTr,lambda(index));
yPred=binClassPredLogistReg(XTr,beta);
berErrorTr=ber(yTr,yPred);
yPred=binClassPredLogistReg(XTe,beta);
berErrorTe=ber(yTe,yPred);

% Close Waitbar
close(h);    

end