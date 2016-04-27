function y_pred= binClassPredLogistReg(X,beta)
% predict the output due to the probability
y_pred=zeros(size(X,1),1);
h=exp(X * beta)./(1+exp(X * beta));
for i=1:size(h,1)
    if h(i,1)>0.5
        y_pred(i,1)=1;
    end
end

end