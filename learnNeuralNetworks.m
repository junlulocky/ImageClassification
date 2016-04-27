%% Learn Neural Networks
function [berErrorTrMulti,berErrorTeMulti,berErrorTrBin,berErrorTeBin]=learnNeuralNetworks(y,X)
%% Create Training Parameters
% Number of Folds
K=5;
% Standartize the data
X=standartscore(X);
% Create Waitbar
h = waitbar(0,'Please wait...');
%% Create Error and Seed Structures 
% Seed Array
seed=randsample(1000,25);
numNeuron=14;
% Initialize Error Structures
berErrorCVtrMulti=zeros(K,1);
berErrorCVteMulti=zeros(K,1);
meanberErrorCVtrMulti=zeros(length(seed),length(numNeuron));
meanberErrorCVteMulti=zeros(length(seed),length(numNeuron));
berErrorCVtrBin=zeros(K,1);
berErrorCVteBin=zeros(K,1);
meanberErrorCVtrBin=zeros(length(seed),length(numNeuron));
meanberErrorCVteBin=zeros(length(seed),length(numNeuron));

%% Run Algorithm
for s=1:length(seed)

    for i=1:length(numNeuron)

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

            %% Create Neural Networks Setup
            nn=nnsetup([size(X,2) 4]); 
            nn.ativation_function='sigmoid';
            nn.output='softmax';
            nn.learningRate=0.1;
            opts.numepochs=15;
            opts.batchsize=20;

            % Split into 1-K Scheme
            yTrBinSch=splitClass(yTr,0);                
            % Compute Optimal Parameters 
            nn=nntrain(nn,XTr,yTrBinSch,opts);       

            % Class Prediction
            % Sample Cross Validation Error
            yPred=nnpredict(nn,XTr);
            berErrorCVtrMulti(k)=ber(yTr,yPred,'Multiclass');
            berErrorCVtrBin(k)=ber(yTr,yPred,'Binary');
            yPred=nnpredict(nn,XTe);
            berErrorCVteMulti(k)=ber(yTe,yPred,'Multiclass');
            berErrorCVteBin(k)=ber(yTe,yPred,'Binary');               

        end
        % Cross Validation Error for each seed
        meanberErrorCVtrMulti(s,i)=mean(berErrorCVtrMulti);
        meanberErrorCVteMulti(s,i)=mean(berErrorCVteMulti);
        meanberErrorCVtrBin(s,i)=mean(berErrorCVtrBin);
        meanberErrorCVteBin(s,i)=mean(berErrorCVteBin);
    end
    % Update Waitbar
    waitbar(s/length(seed));
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