function X_rescaled = standartscore(X)
% Function standartscore:
% Input: X input data
% Output: X input data normalised
% Process: Performs standart score normalization over input data

for i=1:size(X,2)
    meanX = mean(X(:,i));
    tmp1 = X(:,i)-meanX;
    stdX = std(X(:,i));
    tmp2 = tmp1 ./ stdX;
    X_rescaled(:,i)=tmp2;
end


end

