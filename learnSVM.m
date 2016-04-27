%% Learning SVM 
function [berErrorTr,berErrorTe]=learnSVM(y_train,X_train,kernelfunc,Deg)
%% Create Training and Test Sets
% Standartize the data
X_train=standartscore(X_train);
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

%% Create Error, Seed and HyperParameters Structures 
% Seed Array
seed=randsample(1000,1);
% HyperParameters Array
gamma=0;
C=0.001;
% Initialize Error Structures
berErrorCVtr=zeros(K,1);
berErrorCVte=zeros(K,1);
meanberErrorCVtr=zeros(length(seed),length(C));
meanberErrorCVte=zeros(length(seed),length(C));

%% Run Algorithm
for s=1:length(seed)
    % Set Seed
    setSeed(s);
    
    % Create Sets
    X=zeros(2*Nc,D);
    y=zeros(2*Nc,1);
    
    % Create Classes -1 and 1 from the Sets
    idx0=randsample(length(setLabelnot4),Nc);
    idx1=randsample(length(setLabel4),Nc);
    X(1:Nc,:)=X_train(setLabelnot4(idx0),:);
    y(1:Nc)=-1;
    X(Nc+1:end,:)=X_train(setLabel4(idx1),:);
    y(Nc+1:end)=1;
    
    for gd=1:length(gamma)
        for c=1:length(C)

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
                [alpha,beta0]=SMO(Kern,yTr,C(c));
                
                % Class Prediction
                % Sample Cross Validation Error
                yPred=binClassPredSVM(yTr,XTr,XTr,alpha,beta0,'linearKernel',gamma(gd),Deg);
                berErrorCVtr(k)=ber(yTr,yPred);
                yPred=binClassPredSVM(yTr,XTr,XTe,alpha,beta0,'linearKernel',gamma(gd),Deg);
                berErrorCVte(k)=ber(yTe,yPred);
            end
            % Cross Validation Error for each seed and hyperparameter
            meanberErrorCVtr(gd,c)=mean(berErrorCVtr);
            meanberErrorCVte(gd,c)=mean(berErrorCVte);

            % Update Waitbar
            waitbar(c / length(C))
        end
    end
end    
% Close Waitbar
close(h);    

%% Compute Test and Training Error
      
display(meanberErrorCVte);
display(meanberErrorCVtr);

berErrorTr=meanberErrorCVtr;
berErrorTe=meanberErrorCVte;

%berErrorTr=mean(meanberErrorCVtr);
%berErrorTe=mean(meanberErrorCVte);
end