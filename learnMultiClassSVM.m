%% Learning MultiClass SVM 
function [berErrorTrMulti,berErrorTeMulti,berErrorTrBin,berErrorTeBin]=learnMultiClassSVM(y,X,kernelfunc,Deg)
%% Create Training Parameters
% Number of Classes
Class=length(unique(y));
% Standartize the data
X=standartscore(X);
% Number of Folds
K=4;
% Create Waitbar
h = waitbar(0,'Please wait...');

%% Create Error, Seed, HyperParameters and Predictors Structures 
% Seed Array
seed=randsample(1000,1);
% HyperParameters Array
C=[logspace(-3,1.25,10);logspace(-3,1.25,10); logspace(-3,1.25,10); logspace(-3,1.25,10)]';
gamma=logspace(1,-3,10);
% Initialize Error Structures
berErrorCVtrMulti=zeros(K,1);
berErrorCVteMulti=zeros(K,1);
meanberErrorCVtrMulti=zeros(length(seed),1);
meanberErrorCVteMulti=zeros(length(seed),1);
berErrorCVtrBin=zeros(K,1);
berErrorCVteBin=zeros(K,1);
meanberErrorCVtrBin=zeros(length(seed),1);
meanberErrorCVteBin=zeros(length(seed),1);

%% Run Algorithm
for s=1:length(seed)
    % Set Seed
    setSeed(s);

    for gd=1:length(gamma)
        for c=1:size(C,1)

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
                
                % Initialize Predictors Structures
                beta0=zeros(1,Class);
                alpha=zeros(size(XTr,1),Class);
                % Create Data Set for 1vsRest, 2vsRest,3vsRest and 4vsRest
                yTrBinSch=splitClass(yTr,-1);                
                % Compute Kernel 
                switch kernelfunc
                    case 'linearKernel'
                        Kern=linearKernel(XTr, XTr);
                    case 'rbfKernel'
                        Kern=rbfKernel(XTr, XTr,gamma(gd));
                    case 'polyKernel'
                        Kern=polyKernel(XTr, XTr,Deg);
                end
                
                % Compute Optimal Parameters
                for i=1:Class
                    [alpha(:,i),beta0(i)]=SMO(Kern,yTrBinSch(:,i),C(c,i));
                end
                
                % Class Prediction
                % Sample Cross Validation Error
                yPred=multiClassPredSVM(yTrBinSch,XTr,XTr,alpha,beta0,kernelfunc,gamma(gd),Deg);
                berErrorCVtrMulti(k)=ber(yTr,yPred,'Multiclass');
                berErrorCVtrBin(k)=ber(yTr,yPred,'Binary');
                yPred=multiClassPredSVM(yTrBinSch,XTr,XTe,alpha,beta0,kernelfunc,gamma(gd),Deg);
                berErrorCVteMulti(k)=ber(yTe,yPred,'Multiclass');
                berErrorCVteBin(k)=ber(yTe,yPred,'Binary');
                     
            end
            % Cross Validation Error for each seed and hyperparameter
            meanberErrorCVtrMulti(s)=mean(berErrorCVtrMulti);
            meanberErrorCVteMulti(s)=mean(berErrorCVteMulti);
            meanberErrorCVtrBin(s)=mean(berErrorCVtrBin);
            meanberErrorCVteBin(s)=mean(berErrorCVteBin);
        end
        % Update Waitbar
        waitbar(gd / length(gamma));
    end
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