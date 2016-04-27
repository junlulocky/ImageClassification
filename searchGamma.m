%% Heurastic Estimate for HyperParameter Gamma of RBF Kernel 
function gamma=searchGamma(X)
% Input - Data Input Matrix Standartized 
% Output - Rought Estimate of HyperParameter Gamma
m = size(X,1);
n = floor(0.5 * m);
index = ceil(rand(1,n) * m )';
index2 = ceil(rand(1,n) * m)';
temp = X(index,:) - X(index2,:);
dist = sum((temp .* temp)')';
dist = dist(dist ~= 0);
gamma = (quantile(dist ,[0.9 0.5 0.1]) .^ -1);
end