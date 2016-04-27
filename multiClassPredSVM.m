%% Class Prediction for MultiClass SVM
function y_pred=multiClassPredSVM(yTr,XTr,XTe,alpha,beta0,kernelfunc,gamma,D)
% Initialization
K=size(alpha,2);
h=zeros(size(XTe,1),K);
for k=1:K
    % Compute Support Vectors
    SV_idx = find(alpha(:,k)>0);
    % Identify Support Vectors in Data Set
    XTr_SV = XTr(SV_idx, :);
    y_SV = yTr(SV_idx, k);
    alpha_SV = alpha(SV_idx,k);
    % Compute Prediction
    switch kernelfunc
        case 'linearKernel'
            K_pred=linearKernel(XTe, XTr_SV);
        case 'rbfKernel'
            K_pred=rbfKernel(XTe, XTr_SV,gamma);
        case 'polyKernel'
            K_pred=polyKernel(XTe, XTr_SV,D);
    end
    h(:,k)=K_pred*(alpha_SV.*y_SV)+beta0(k);
end
% Compute Predicted Class
[~,y_pred]=max(h,[],2);
end