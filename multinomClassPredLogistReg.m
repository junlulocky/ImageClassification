%% Predicted Class for Multinomial Logistic Regression
function y_pred=multinomClassPredLogistReg(X,beta)
% predict the output due to the probability 
h=exp(X*beta)./repmat(sum(exp(X*beta),2),1,4);
[~,y_pred]=max(h,[],2);
end