%% Solver to Imbalanced Training Set
function [y,X]=solveImbalaceData(y_train,X_train,SolveImbalance)
%%
% Procedures: 
% (1) - Random Under Sampling: - Randomly let go data points from the
% majority classes so that they meet the number of samples of the minority 
% class.
% (2) - Random Over Sampling: - Randomly duplicate data points from the
% minority classes so that they meet the number of samples of the majority 
% class.
% (3) - Algorithmic Over Sampling: - Algorithmicly create virtual samples
% of the minority classes against the majority class so that their samples
% number meet. Use of extended version of SMOTE, ADASYN.

% Samples of Label 1
setLabel1=find(y_train==1);
% Samples of Label 2
setLabel2=find(y_train==2);
% Samples of Label 3
setLabel3=find(y_train==3);
% Samples of Label 4
setLabel4=find(y_train==4);

switch SolveImbalance
    case 'RandomUnderSampling'
        % UnderSampling of Classes 3, 4
        % Number of Samples per Class
        Nc=850; %600   
        % Create Sets
        X=zeros(4*Nc,size(X_train,2));
        y=zeros(4*Nc,1);
        % Create Classes 1,2,3,4 from the Sets
        idx1=randsample(length(setLabel1),Nc);
        idx2=randsample(length(setLabel2),Nc);
        idx3=randsample(length(setLabel3),Nc);
        idx4=randsample(length(setLabel4),Nc);
        X(1:Nc,:)=X_train(setLabel1(idx1),:);
        y(1:Nc)=1;
        X(Nc+1:2*Nc,:)=X_train(setLabel2(idx2),:);
        y(Nc+1:2*Nc)=2;
        X(2*Nc+1:3*Nc,:)=X_train(setLabel3(idx3),:);
        y(2*Nc+1:3*Nc)=3;
        X(3*Nc+1:end,:)=X_train(setLabel4(idx4),:);
        y(3*Nc+1:end)=4;
    case 'RandomOverSampling'
        % Over Sampling of Classes 1, 2 and 3 
        % Number of Samples per Class
        Nc=2000; % 1700
        % Create Sets
        X=zeros(4*Nc,size(X_train,2));
        y=zeros(4*Nc,1);
        % Create Classes 1,2,3,4 from the Sets
        idx1=randi(length(setLabel1),Nc,1);
        idx2=randi(length(setLabel2),Nc,1);
        idx3=randi(length(setLabel3),Nc,1);
        idx4=randi(length(setLabel4),Nc,1);
        X(1:Nc,:)=X_train(setLabel1(idx1),:);
        y(1:Nc)=1;
        X(Nc+1:2*Nc,:)=X_train(setLabel2(idx2),:);
        y(Nc+1:2*Nc)=2;
        X(2*Nc+1:3*Nc,:)=X_train(setLabel3(idx3),:);
        y(2*Nc+1:3*Nc)=3;
        X(3*Nc+1:end,:)=X_train(setLabel4(idx4),:);
        y(3*Nc+1:end)=4;
    case 'AlgorithmicOverSampling'
        % Algorithmic OverSampling to meet Class 4
        % Create Sets
        X=[];
        y=[];
        % Create Classes 1,2,3,4 from the Sets
        % Class 1
        [tmp,~]=ADASYN([X_train(setLabel1,:);X_train(setLabel4,:);],[ones(length(setLabel1),1);zeros(length(setLabel4),1)],[],[],[],true); %% Attention to Standart!
        X=[X;X_train(setLabel1,:);tmp];
        y=[y;ones(length(setLabel1)+size(tmp,1),1)];  
        % Class 2
        [tmp,~]=ADASYN([X_train(setLabel2,:);X_train(setLabel4,:);],[ones(length(setLabel2),1);zeros(length(setLabel4),1)],[],[],[],true); %% Attention to Standart!
        X=[X;X_train(setLabel2,:);tmp];
        y=[y;2*ones(length(setLabel2)+size(tmp,1),1)];   
        % Class 3
        [tmp,~]=ADASYN([X_train(setLabel3,:);X_train(setLabel4,:);],[ones(length(setLabel3),1);zeros(length(setLabel4),1)],[],[],[],true); %% Attention to Standart!
        X=[X;X_train(setLabel3,:);tmp];
        y=[y;3*ones(length(setLabel3)+size(tmp,1),1)];   
        % Class 4
        X=[X;X_train(setLabel4,:)];
        y=[y;4*ones(length(setLabel4),1)];
        
end   

end
