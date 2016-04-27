%% Balanced Error Rate - BER
function error=ber(y,y_pred,task)
% Compute Balanced Error Rate
% Input: Y True Values, Y Prediction (class labels)
% Output: Balanced Error Rate
if strcmp(task,'Binary')
    y(y~=4)=0;
    y_pred(y_pred~=4)=0;
    y(y==4)=1;
    y_pred(y_pred==4)=1;
end    

% Initialize Parameters
class=sort(unique(y));
numclass=length(class);
bertable=zeros(numclass,numclass);

% Compute Balanced Error Table; i for Truth, j for Prediction
for i=1:(numclass)
    for j=1:(numclass)
        search=(y_pred == class(i) & y == class(j));
        bertable(i,j)=length(find(search));
    end
end

% Compute Balanced Error
error=0;
for i=1:numclass
    den=sum(bertable(i,:));
    error=error+(den-bertable(i,i))/den;
end
error=error*(1/numclass);

%display(y_pred);
%display(y);
%pause;
end
