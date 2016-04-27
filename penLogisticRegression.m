%% Penalized Logistic Regression
function beta = penLogisticRegression(Y,tX,lambda)
% Penalized Logistic regression using stochastic gradient descent
m=length(Y);
% Choose Parameters
maxIters=200;
batchsize=20;
beta = zeros(size(tX,2),1);
alpha=1/10000000000^2.5;
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
       % Calculate the cost and gradient descent 
       tmp=exp(x*beta)./(1+exp(x*beta));
       betasquared=beta.^2;
       %L = sum(1/m*(-y.*log(tmp)-(1-y).*log(1-tmp))) + lambda /(2*m) * sum(betasquared(2:size(beta),1)) ;
       % Add the lamdba to penalize the gradient descent
       gradient = -x'*(y - tmp)/batchsize;
       gradient(2:size(beta),1)=gradient(2:size(beta),1)+(lambda/batchsize*beta(2:size(beta),1));
       % Update beta
       beta=beta-gradient*alpha;
   end
    % Update alpha
   if i<75
      alpha=1/(10000000000+i)^2.5;
   else
      alpha=1/(10000000000+i)^2.6;
   end
   %pause;
   
   tmp = exp(tX * beta)./(1+exp(tX * beta));
   L = sum(1/m*(-Y.*log(tmp)-(1-Y).*log(1-tmp))) + lambda /(2*m) * sum(betasquared(2:size(beta),1)) ;
   %display(L);
   %display(i);
   
end
%figure;
%plot(1:maxIters,L);
end