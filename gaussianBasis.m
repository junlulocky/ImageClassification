%% Gaussian Basis Function
function phi=gaussianBasis(X)
% Function polybasis:
% Input: X input data
% Output: phi p
% Process: Express X input data in a Gaussian Basis Function of mean 0 and 
% unit variance
phi=exp(-X.^2);
end
