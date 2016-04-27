%% Feature Engineering
function [y,X,V]=featureEngineering(y,X,flagPCA,flagOutlier,flagSolveImbalanced,flagFeature)
%% PCA Fatorization - Reducing Dimensionality
if flagPCA==1    
switch flagFeature
    case 1
        % LMSVD for CNN Features
        opts.maxit = 300;
        opts.tol = 1e-8;
        fprintf('\nPlease Wait. Computing SVD Decomposition for CNN Features takes a while\n');
        X=double(X);
        [U,S,V]=lmsvd(X,50,opts);       
    case 2
        % Matlab PCA for HOG Features
        [U,S,V]=svd(X);
end
% Select the first 50 Principal Components
X=U(:,1:50)*S(1:50,1:50);   
else
V=0;    
end
%% Standartize Training Data
X=standartscore(X);
%% Unsupervised Learning - Outlier Removal
if flagOutlier==1
[y,X]=learnKmeans(y,X);
end
%% Solve Imbalance Data
if ~strcmp('None',flagSolveImbalanced)
[y,X]=solveImbalaceData(y,X,flagSolveImbalanced);
end

clearvars -except y X V
end
