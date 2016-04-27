%% Multiclass Logistic Regression
function beta=multinomLogisticRegression(y,tX)
% Multiclass Logistic regression using stochastic gradient descent
% Change Targe Value Coding Scheme
Y=splitClass(y,0);
% Assign Problem's Dimensions
m=size(Y,1);
K=size(Y,2);
% Choose Parameters
maxIters=500;
batchsize=20;
beta = zeros(size(tX,2),K);
alpha=1/10^1.7;
% Analysis
L=zeros(maxIters,1);
for i=1:maxIters   
   % Random Permutation
   idx=randperm(m);
   x=tX(idx,:);
   y=Y(idx,:);
   xcopy=x;
   ycopy=y;
   for j=1:floor(m/batchsize)          
       % Pick Example
       x=xcopy((j-1)*batchsize+1:j*batchsize,:);
       y=ycopy((j-1)*batchsize+1:j*batchsize,:);
       % Calculate the cost and gradient descent for mini batch 
       tmp=exp(x*beta)./repmat(sum(exp(x*beta),2),1,4);
       gradient=-x'*(y-tmp)/(batchsize*K);    
       % Update beta
       beta=beta - gradient * alpha;
   end
   % Update alpha
    if i<75
        alpha=1/(10+i)^1.8;
    else
        alpha=1/(10+i)^1.9;
    end    
   %pause;
   
   tmp = exp(tX*beta)./repmat(sum(exp(tX*beta),2),1,4);
   Lmat=-(1/(m*K))*(Y.*(log(tmp)));
   L(i)= sum(Lmat(:));
   %display(L(i));
   %display(i);

end

end
