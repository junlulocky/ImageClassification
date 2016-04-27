%% Class Prediction for Binary Class SVM
function y_pred=binClassPredSVM(yTr,XTr,XTe,alpha,beta0,kernelfunc,gamma,D)
% Compute Support Vectors
SV_idx = find(alpha>0);
% Identify Support Vectors in Data Set
XTr_SV = XTr(SV_idx, :);
y_SV = yTr(SV_idx);
alpha_SV = alpha(SV_idx);
% Compute Prediction
switch kernelfunc
    case 'linearKernel'
        K_pred=linearKernel(XTe, XTr_SV);
    case 'rbfKernel'
        K_pred=rbfKernel(XTe, XTr_SV,gamma);
    case 'polyKernel'
        K_pred=polyKernel(XTe, XTr_SV,D);
end
y_pred = K_pred*(alpha_SV.*y_SV)+beta0;
% Compute Predicted Class
y_pred(y_pred>0)=1;
y_pred(y_pred<0)=-1;
end