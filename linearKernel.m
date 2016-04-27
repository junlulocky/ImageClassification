function [ K ]=linearKernel(X1,X2)
%LINEAR_KERNEL Build a linear kernel.
K=X1*X2';
end