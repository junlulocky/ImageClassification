%% Radial Basis Function Kernel 
function K=rbfKernel(X1,X2,gamma)
% Compute Radial Basis Function Kernel
aux=pdist2(X1,X2);
K=exp(-gamma*aux);
end