function Y=splitClass(y,negTarget)
% Function Split Class
% Input: y Matrix with 1:K coding scheme
% Output: Y Matrix with Positive Target One; Negative Targe to Define
K=length(unique(y));
Y=zeros(size(y,1),K)+negTarget;
for k=1:K
    Y(y==k,k)=1;
end
end
