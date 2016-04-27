%% Split Training Set into Training and Validation Set
function [yTrain,yValid,XTrain,XValid]=splitTrainValid(y,X)
% Randomize and get Indices
N=size(X,1);
Ntr=round(0.75*N);
idx=randperm(N);
X(:,:)=X(idx,:);
y=y(idx);
% Perform Assingments
XTrain=X(1:Ntr,:);
yTrain=y(1:Ntr);
XValid=X(Ntr+1:N,:);
yValid=y(Ntr+1:N,:);
end