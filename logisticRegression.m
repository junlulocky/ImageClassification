%% Logistic Regression
function [beta] = logisticRegression(Y,tX)
% Logistic regression using stochastic gradient descent
m=length(Y);
% Choose Parameters
maxIters=500;
batchsize=20;
beta = zeros(size(tX,2),1);
alpha=1/10^2.3;
% Analysis
L=zeros(maxIters,1);

for i=1:maxIters
   % Random Permutation
   idx=randperm(m);
   x=tX(idx,:);
   y=Y(idx);
   xcopy=x;
   ycopy=y; 
   for j=1:floor(m/batchsize)          
       % Pick Example
       x=xcopy((j-1)*batchsize+1:j*batchsize,:);
       y=ycopy((j-1)*batchsize+1:j*batchsize);
       % Calculate the cost and gradient descent for mini batch 
       tmp = exp(x * beta)./(1+exp(x * beta));
       gradient = -x'*(y - tmp)/batchsize;
       % Update beta
       beta=beta - gradient * alpha;
   end
   % Update alpha
   if i<75
      alpha=1/(10+i)^2.3;
   else
      alpha=1/(20+i)^2.4;
   end    
   %pause;   
   tmp = exp(tX * beta)./(1+exp(tX * beta));
   L(i)= sum(1/m*(-Y.*log(tmp)-(1-Y).*log(1-tmp)));
   %display(L(i));
   %display(i);
end
%figure;
%plot(1:maxIters,L);
end
