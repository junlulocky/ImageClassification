function phi =polybasis(X,M)
% Function polybasis:
% Input: X input data, M degree of polynomial
% Output: phi p
% Process: Express X input data in a Polynomial basis function up to degree

D=size(X,2);
N=size(X,1);
phi=zeros(N,M*D);
for j=1:M
    phi(:,(1+(j-1)*D):(j*D))=X.^j;
end

end

