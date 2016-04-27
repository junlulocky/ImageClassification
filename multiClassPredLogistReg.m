%% Predicted Class for MultiClass Logistic Regression
function y_pred=multiClassPredLogistReg(X,beta)
% predict the output due to the probability 
h=zeros(size(X,1),size(beta,2));
for k=1:size(beta,2)
    h(:,k)=exp(X*beta(:,k))./(1+exp(X * beta(:,k)));
end
[~,y_pred]=max(h,[],2);
end