%% Learn K Means for Outliers Removal
function [y_train,X_train]=learnKmeans(y_train,X_train)
%% Create Clustering Data
setLabelnot4=find(y_train~=4);
X=X_train(setLabelnot4,:);
y=y_train(setLabelnot4,:);
%% Initilize K-Means Parameters
MuOld=[mean(X_train(y_train==1,:)',2) mean(X_train(y_train==2,:)',2) mean(X_train(y_train==4,:)',2)];
maxIters=10;
L=zeros(1,maxIters);
Lold=inf;

%% Run Algorithm
for i = 1:maxIters
  % update R and Mu
  [Ln, r, Mu] = kmeansUpdate(X',MuOld);

  % average distance over all n
  L(i) = mean(Ln);
  %fprintf('\n%d .4%f\n', i, L(i));

  % convergence
  if (L(i)>Lold)
      fprintf('The Algorithm is not convering');
      break;
  end

  % new mean is the old mean now
  MuOld = Mu;
  Lold = L(i);
  
end
r=r';

y_train(setLabelnot4(y~=r))=[];
X_train(setLabelnot4(y~=r),:)=[];
end