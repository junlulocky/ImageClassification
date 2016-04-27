function [Ln,r,Mu] = kmeansUpdate(X, MuOld)
% update r and Mu given X and Mu
% X is DxN data
% Mu is DxK mean vector
% r is 1xN responsibility vector, e.g. r = [1,2,1] for 2 clusters 3 data points
% Ln is 1xN minimum distance to its center for each point n

  % initialize
  K = size(MuOld,2);
  [D,N] = size(X);
  r = zeros(1,N);
  Mu = zeros(D,K);
  Ln = zeros(1,N);

clust=0;
% compute r
for n=1:N
  min=+inf;
  for k=1:K
      dist=norm(X(:,n)-MuOld(:,k))^2;
      if dist<min
          min=dist;
          clust=k;
      end
  end
  Ln(n)=min;
  r(n)=clust;
end   

% compute Mu for each k
for k=1:K
    index = find(r==k);
    Mu(:,k)=mean(X(:,index),2);
end

