%% Polynomial Kernel 
function K=polyKernel(X1,X2,D)
% Input: Matrixes X1, X1; Degree of the Kernel
% Output: Polynomial Kernel K(X1,X2)
tmp=linearKernel(X1,X2);
K=(1+tmp).^D;
end